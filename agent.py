import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from config import (
    MODEL_PATH,
    ACTION_DIM,
    REWARD_TILE_MILESTONES,
    PPO_LR,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_CLIP_EPS,
    PPO_EPOCHS,
    PPO_MINI_BATCH_SIZE,
    PPO_ENTROPY_COEF,
    PPO_ENTROPY_COEF_END,
    PPO_ENTROPY_DECAY_UPDATES,
    PPO_VALUE_COEF,
    PPO_MAX_GRAD_NORM,
    PPO_TARGET_KL,
    PPO_N_STEPS,
    PPO_N_ENVS,
    N,
    LOOKAHEAD_EVAL_DEPTH,
    LOOKAHEAD_EVAL_BEAM,
    LOOKAHEAD_EVAL_HEURISTIC_WEIGHT,
    LOOKAHEAD_EVAL_VALUE_WEIGHT,
)
from env import Game2048Env
from model import ActorCritic


class RolloutBuffer:
    """标准 on-policy rollout buffer, 支持多环境并行采集。"""

    def __init__(self, n_steps, n_envs, obs_shape):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.logprobs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.action_masks = np.zeros((n_steps, n_envs, ACTION_DIM), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value, action_mask):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.action_masks[i] = action_mask
        self.ptr += 1

    def ready(self):
        return self.ptr >= self.n_steps

    def reset(self):
        self.ptr = 0

    def flatten(self):
        n = self.n_steps * self.n_envs
        return {
            "obs": self.obs.reshape(n, *self.obs.shape[2:]),
            "actions": self.actions.reshape(n),
            "logprobs": self.logprobs.reshape(n),
            "rewards": self.rewards.reshape(n),
            "dones": self.dones.reshape(n),
            "values": self.values.reshape(n),
            "action_masks": self.action_masks.reshape(n, ACTION_DIM),
        }


class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=PPO_LR, eps=1e-5)

        self.n_envs = PPO_N_ENVS
        self.n_steps = PPO_N_STEPS
        self.buffer = RolloutBuffer(self.n_steps, self.n_envs, (N, N))
        self.envs = [Game2048Env() for _ in range(self.n_envs)]

        self.update_count = 0
        self.last_update = (0.0, 0.0, 0.0)
        self._bootstrap_values = np.zeros(self.n_envs, dtype=np.float32)
        self._bootstrap_dones = np.zeros(self.n_envs, dtype=np.float32)

        # 兼容旧接口
        self.training_stage = 0

    def _stage_name(self):
        return "ppo"

    def _state_to_board(self, state):
        state_arr = np.asarray(state, dtype=np.int64)
        board = np.zeros_like(state_arr, dtype=np.int64)
        non_zero = state_arr > 0
        board[non_zero] = (1 << state_arr[non_zero]).astype(np.int64)
        return board

    def _board_to_state(self, board):
        state = np.zeros_like(board, dtype=np.int64)
        non_zero = board > 0
        state[non_zero] = np.log2(board[non_zero]).astype(np.int64)
        return state

    def _value_of_board(self, board):
        state = self._board_to_state(board).astype(np.float32)
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.net(state_t)
        return float(value.item())

    def _rollout_score(self, board, depth, beam_width):
        base_state = self._board_to_state(board).astype(np.float32)
        base_state_t = torch.from_numpy(base_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, base_value = self.net(base_state_t)
        if depth <= 0:
            return float(base_value.item())

        candidates = []
        for action in range(ACTION_DIM):
            next_board, merge_score = Game2048Env._move_board(board, action)
            if next_board is None:
                continue
            candidates.append((merge_score, action, next_board))

        if not candidates:
            return float(base_value.item())

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_child = candidates[: max(1, int(beam_width))]

        best_score = -1e18
        for immediate, _, next_board in best_child:
            child_score = immediate + PPO_GAMMA * self._rollout_score(next_board, depth - 1, beam_width)
            if child_score > best_score:
                best_score = child_score

        return float(LOOKAHEAD_EVAL_VALUE_WEIGHT * base_value.item() + best_score)

    def reset_envs(self):
        states = np.stack([env.reset() for env in self.envs], axis=0).astype(np.float32)
        return states

    def _entropy_coef(self):
        if PPO_ENTROPY_DECAY_UPDATES <= 0:
            return PPO_ENTROPY_COEF
        t = min(1.0, self.update_count / PPO_ENTROPY_DECAY_UPDATES)
        return PPO_ENTROPY_COEF + (PPO_ENTROPY_COEF_END - PPO_ENTROPY_COEF) * t

    @staticmethod
    def _mask_from_env(env):
        return env.get_action_mask()

    def _get_action_masks(self, states):
        """从 log2 状态重建各环境动作掩码。"""
        masks = np.zeros((len(states), ACTION_DIM), dtype=np.float32)
        for i, state in enumerate(states):
            board = np.zeros((N, N), dtype=np.int64)
            nz = state > 0
            board[nz] = (1 << state[nz].astype(np.int64))
            for action in range(ACTION_DIM):
                if Game2048Env._preview_board(board, action) is not None:
                    masks[i, action] = 1.0
        return masks

    def _act_with_masks(self, states, masks, evaluate=False):
        states_t = torch.from_numpy(states).float().to(self.device)
        masks_t = torch.from_numpy(masks).float().to(self.device)

        self.net.eval()
        with torch.no_grad():
            logits, values = self.net(states_t, action_mask=masks_t)
            dist = Categorical(logits=logits)
            if evaluate:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = dist.sample()
            logprobs = dist.log_prob(actions)

        return (
            actions.cpu().numpy(),
            logprobs.cpu().numpy(),
            values.cpu().numpy(),
        )

    def _select_action_lookahead(self, state):
        board = self._state_to_board(state)
        candidates = []
        for action in range(ACTION_DIM):
            next_board, merge_score = Game2048Env._move_board(board, action)
            if next_board is None:
                continue
            score = merge_score + self._rollout_score(
                next_board,
                LOOKAHEAD_EVAL_DEPTH - 1,
                LOOKAHEAD_EVAL_BEAM,
            )
            candidates.append((score, action))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return int(candidates[0][1])

    def act_batch(self, states, evaluate=False):
        """批量选动作, 训练时采样, 评估时贪心。"""
        masks = self._get_action_masks(states)
        actions, logps, values = self._act_with_masks(states, masks, evaluate=evaluate)
        return actions, logps, values, masks

    def select_action(self, state, evaluate=False):
        """单环境接口, 供 UI / 评估脚本使用。"""
        state_np = np.asarray(state, dtype=np.float32)
        if evaluate:
            lookahead_action = self._select_action_lookahead(state_np)
            if lookahead_action is not None:
                actions, logps, values, _ = self.act_batch(state_np[np.newaxis], evaluate=True)
                actions[0] = lookahead_action
                return int(actions[0]), float(logps[0]), float(values[0])

        actions, logps, values, _ = self.act_batch(state_np[np.newaxis], evaluate=evaluate)
        return int(actions[0]), float(logps[0]), float(values[0])

    def get_valid_actions(self, state):
        """兼容 UI: 从 log2 状态返回合法动作。"""
        board = np.zeros((N, N), dtype=np.int64)
        state = np.asarray(state)
        nz = state > 0
        board[nz] = (1 << state[nz].astype(np.int64))
        valid = []
        for action in range(ACTION_DIM):
            if Game2048Env._preview_board(board, action) is not None:
                valid.append(action)
        return valid

    def collect_rollout(self, states):
        """采集 n_steps × n_envs 条 transition。"""
        self.buffer.reset()
        ep_returns = []
        ep_lengths = []
        ep_max_tiles = []
        cur_return = np.zeros(self.n_envs, dtype=np.float32)
        cur_length = np.zeros(self.n_envs, dtype=np.int32)
        dones = np.zeros(self.n_envs, dtype=np.float32)

        for _ in range(self.n_steps):
            action_masks = np.stack([env.get_action_mask() for env in self.envs], axis=0)
            actions, logps, values = self._act_with_masks(states, action_masks, evaluate=False)

            next_states = np.zeros_like(states)
            rewards = np.zeros(self.n_envs, dtype=np.float32)
            dones = np.zeros(self.n_envs, dtype=np.float32)

            for i, env in enumerate(self.envs):
                ns, reward, done, _ = env.step(int(actions[i]))
                next_states[i] = ns
                rewards[i] = reward
                dones[i] = float(done)
                cur_return[i] += reward
                cur_length[i] += 1

                if done:
                    ep_returns.append(float(env.score))
                    ep_lengths.append(int(cur_length[i]))
                    ep_max_tiles.append(int(env.board.max()))
                    next_states[i] = env.reset()
                    cur_return[i] = 0.0
                    cur_length[i] = 0

            self.buffer.add(states, actions, logps, rewards, dones, values, action_masks)
            states = next_states.astype(np.float32)

        # bootstrap
        with torch.no_grad():
            states_t = torch.from_numpy(states).float().to(self.device)
            masks_t = torch.from_numpy(self._get_action_masks(states)).float().to(self.device)
            _, last_values = self.net(states_t, action_mask=masks_t)
            last_values = last_values.cpu().numpy()

        stats = {
            "ep_returns": ep_returns,
            "ep_lengths": ep_lengths,
            "ep_max_tiles": ep_max_tiles,
            "last_values": last_values,
            "last_dones": dones,
        }
        return states, stats

    def _compute_gae(self, rewards, values, dones, last_values, last_dones):
        """标准 GAE-Lambda, 正确处理 episode 边界。"""
        n_steps, n_envs = rewards.shape
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros(n_envs, dtype=np.float32)

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones
            else:
                next_values = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1]

            delta = rewards[t] + PPO_GAMMA * next_values * next_non_terminal - values[t]
            last_gae = delta + PPO_GAMMA * PPO_GAE_LAMBDA * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self):
        """标准 PPO 更新。"""
        flat = self.buffer.flatten()
        n = flat["obs"].shape[0]

        adv, ret = self._compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            self._bootstrap_values,
            self._bootstrap_dones,
        )
        adv_flat = adv.reshape(-1)
        ret_flat = ret.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        obs_t = torch.from_numpy(flat["obs"]).float().to(self.device)
        actions_t = torch.from_numpy(flat["actions"]).long().to(self.device)
        old_logps_t = torch.from_numpy(flat["logprobs"]).float().to(self.device)
        adv_t = torch.from_numpy(adv_flat).float().to(self.device)
        ret_t = torch.from_numpy(ret_flat).float().to(self.device)
        masks_t = torch.from_numpy(flat["action_masks"]).float().to(self.device)
        old_values_t = torch.from_numpy(flat["values"]).float().to(self.device)

        batch_size = n
        mini_batch = min(PPO_MINI_BATCH_SIZE, batch_size)
        entropy_coef = self._entropy_coef()

        self.net.train()
        pi_losses, vf_losses, entropies = [], [], []

        for _ in range(PPO_EPOCHS):
            stop_early = False
            idx = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch):
                mb = idx[start : start + mini_batch]

                logits, new_values = self.net(obs_t[mb], action_mask=masks_t[mb])
                dist = Categorical(logits=logits)
                new_logps = dist.log_prob(actions_t[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logps - old_logps_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                v_pred = new_values.view(-1)
                v_old = old_values_t[mb].view(-1)
                v_clipped = v_old + (v_pred - v_old).clamp(-PPO_CLIP_EPS, PPO_CLIP_EPS)
                value_loss = torch.max(
                    F.mse_loss(v_pred, ret_t[mb]),
                    F.mse_loss(v_clipped, ret_t[mb]),
                )

                loss = policy_loss + PPO_VALUE_COEF * value_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), PPO_MAX_GRAD_NORM)
                self.optimizer.step()

                if PPO_TARGET_KL > 0:
                    approx_kl = (old_logps_t[mb] - new_logps).mean().item()
                    if approx_kl > PPO_TARGET_KL:
                        stop_early = True
                        break

                pi_losses.append(float(policy_loss.item()))
                vf_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

            if stop_early:
                break

        self.update_count += 1
        self.last_update = (
            float(np.mean(pi_losses)) if pi_losses else 0.0,
            float(np.mean(vf_losses)) if vf_losses else 0.0,
            float(np.mean(entropies)) if entropies else 0.0,
        )
        return self.last_update

    def collect_and_update(self):
        states = self.reset_envs()
        states, stats = self.collect_rollout(states)
        self._bootstrap_values = stats["last_values"]
        self._bootstrap_dones = stats["last_dones"]
        losses = self.update()
        return stats, losses

    # ── 兼容旧 train.py 接口 ──
    def start_episode(self):
        pass

    def store_transition(self, *args, **kwargs):
        pass

    def finalize_episode(self, max_tile):
        return 0

    def maybe_advance_curriculum(self, *args, **kwargs):
        return False

    def maybe_update(self):
        return None

    def reset_rollout(self):
        self.buffer.reset()

    def load_model(self, path=MODEL_PATH):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.net.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.update_count = int(ckpt.get("update_count", 0))
            print(f"Model loaded from {path}")
        except Exception:
            print("No previous model found, starting fresh.")

    def save_model(self, path=MODEL_PATH):
        torch.save(
            {
                "model_state": self.net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "update_count": self.update_count,
            },
            path,
        )
        print(f"Model saved to {path}")

    def evaluate(self, n_episodes=20, max_steps=1024):
        """贪心评估, 返回统计信息。"""
        env = Game2048Env()
        scores, max_tiles, wins = [], [], 0
        milestone_hits = {threshold: 0 for threshold in REWARD_TILE_MILESTONES}
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < max_steps:
                action, _, _ = self.select_action(state, evaluate=True)
                state, _, done, _ = env.step(action)
                steps += 1
            scores.append(env.score)
            mt = int(env.board.max())
            max_tiles.append(mt)
            for threshold in milestone_hits:
                if mt >= threshold:
                    milestone_hits[threshold] += 1
            if mt >= 2048:
                wins += 1
        return {
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "mean_max_tile": float(np.mean(max_tiles)),
            "best_max_tile": int(np.max(max_tiles)),
            "win_rate": wins / n_episodes,
            "tile_128_rate": milestone_hits.get(128, 0) / n_episodes,
            "tile_256_rate": milestone_hits.get(256, 0) / n_episodes,
            "tile_512_rate": milestone_hits.get(512, 0) / n_episodes,
            "tile_1024_rate": milestone_hits.get(1024, 0) / n_episodes,
            "tile_2048_rate": milestone_hits.get(2048, 0) / n_episodes,
        }
