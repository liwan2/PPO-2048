"""Part 2: 启发式搜索算法 — Expectimax + 启发式评估函数"""

import math
import random
import numpy as np
from config import N, ACTION_DIM, ANCHOR_CORNER
from env import Game2048Env


def heuristic_evaluate(board, weights=None):
    """启发式评估函数"""
    if weights is None:
        weights = {
            "empty": 10.0, "monotonicity": 5.0, "smoothness": 2.0,
            "corner": 4.0, "edge": 2.0, "max_value": 1.0, "merge_potential": 3.0,
        }
    n = board.shape[0]
    log_board = np.zeros_like(board, dtype=np.float64)
    nz = board > 0
    log_board[nz] = np.log2(board[nz])
    score = 0.0

    empty_count = int(np.sum(board == 0))
    score += weights.get("empty", 10.0) * empty_count

    mono_score = 0.0
    for r in range(n):
        for c in range(n - 1):
            if log_board[r, c] > 0 and log_board[r, c + 1] > 0:
                mono_score -= abs(log_board[r, c] - log_board[r, c + 1])
    for c in range(n):
        for r in range(n - 1):
            if log_board[r, c] > 0 and log_board[r + 1, c] > 0:
                mono_score -= abs(log_board[r, c] - log_board[r + 1, c])
    score += weights.get("monotonicity", 5.0) * mono_score

    smooth_score = 0.0
    for r in range(n):
        for c in range(n - 1):
            if log_board[r, c] > 0 and log_board[r, c + 1] > 0:
                smooth_score -= abs(log_board[r, c] - log_board[r, c + 1])
            elif board[r, c] == 0 and board[r, c + 1] > 0:
                smooth_score -= 1.0
    for c in range(n):
        for r in range(n - 1):
            if log_board[r, c] > 0 and log_board[r + 1, c] > 0:
                smooth_score -= abs(log_board[r, c] - log_board[r + 1, c])
            elif board[r, c] == 0 and board[r + 1, c] > 0:
                smooth_score -= 1.0
    score += weights.get("smoothness", 2.0) * smooth_score

    if ANCHOR_CORNER == "bottom_left":
        corners = [(n - 1, 0)]
    elif ANCHOR_CORNER == "bottom_right":
        corners = [(n - 1, n - 1)]
    elif ANCHOR_CORNER == "top_left":
        corners = [(0, 0)]
    else:
        corners = [(0, n - 1)]
    max_val = int(board.max())
    for ar, ac in corners:
        if int(board[ar, ac]) == max_val and max_val > 0:
            score += weights.get("corner", 4.0) * np.log2(max_val)

    edge_positions = set()
    last = n - 1
    for idx in range(n):
        edge_positions.add((0, idx)); edge_positions.add((last, idx))
        edge_positions.add((idx, 0)); edge_positions.add((idx, last))
    max_positions = list(zip(*np.where(board == max_val)))
    for pos in max_positions:
        if pos in edge_positions and max_val > 0:
            score += weights.get("edge", 2.0) * np.log2(max_val); break

    if max_val > 0:
        score += weights.get("max_value", 1.0) * np.log2(max_val)

    merge_potential = 0.0
    for r in range(n):
        for c in range(n - 1):
            if board[r, c] > 0 and board[r, c] == board[r, c + 1]:
                merge_potential += np.log2(board[r, c]) * 2
    for c in range(n):
        for r in range(n - 1):
            if board[r, c] > 0 and board[r, c] == board[r + 1, c]:
                merge_potential += np.log2(board[r, c]) * 2
    score += weights.get("merge_potential", 3.0) * merge_potential
    return score


def _get_spawn_states(board, n_empty_sample=4):
    """随机生成后继状态"""
    empty = list(zip(*np.where(board == 0)))
    if not empty:
        return [(board.copy(), 1.0)]
    if len(empty) > n_empty_sample:
        empty = random.sample(empty, n_empty_sample)
    results = []
    per_cell_prob = 1.0 / max(1, len(list(zip(*np.where(board == 0)))))
    for r, c in empty:
        for val, prob in [(2, 0.9), (4, 0.1)]:
            nb = board.copy(); nb[r, c] = val
            results.append((nb, prob * per_cell_prob))
    return results


def expectimax_search(board, depth, weights=None):
    """Expectimax 搜索"""
    best_action, best_score = None, -math.inf
    for action in range(ACTION_DIM):
        new_board, merge_score = Game2048Env._move_board(board, action)
        if new_board is None:
            continue
        value = _expectimax_value(new_board, depth - 1, weights) + merge_score * 0.25
        if value > best_score:
            best_score, best_action = value, action
    return best_action, best_score


def _expectimax_value(board, depth, weights=None):
    if depth <= 0:
        return heuristic_evaluate(board, weights)
    empty = list(zip(*np.where(board == 0)))
    if not empty:
        return heuristic_evaluate(board, weights)
    best = -math.inf
    for action in range(ACTION_DIM):
        new_board, merge_score = Game2048Env._move_board(board, action)
        if new_board is None:
            continue
        spawn_states = _get_spawn_states(new_board, n_empty_sample=2)
        expected = sum(prob * _expectimax_value(sb, depth - 1, weights) for sb, prob in spawn_states)
        val = expected + merge_score * 0.25
        if val > best:
            best = val
    if best == -math.inf:
        return heuristic_evaluate(board, weights)
    return best


class HeuristicSearchAgent:
    """启发式搜索智能体"""
    def __init__(self, search_depth=3, weights=None, use_expectimax=True, verbose=False):
        self.search_depth = search_depth
        self.weights = weights if weights else {
            "empty": 10.0, "monotonicity": 5.0, "smoothness": 2.0,
            "corner": 4.0, "edge": 2.0, "max_value": 1.0, "merge_potential": 3.0,
        }
        self.use_expectimax = use_expectimax
        self.verbose = verbose

    def get_weights_dict(self):
        return self.weights.copy()

    def select_action(self, state):
        board = np.zeros((N, N), dtype=np.int64)
        nz = state > 0
        board[nz] = (1 << state[nz].astype(np.int64))
        valid = [a for a in range(ACTION_DIM) if Game2048Env._preview_board(board, a) is not None]
        if not valid:
            return None, 0.0, {"valid": False}
        if len(valid) == 1:
            return valid[0], 0.0, {"valid": True, "forced": True}
        if self.use_expectimax:
            best_action, best_score = expectimax_search(board, self.search_depth, self.weights)
        else:
            best_action, best_score = None, -math.inf
            for action in valid:
                nb, ms = Game2048Env._move_board(board, action)
                sc = heuristic_evaluate(nb, self.weights) + ms * 0.25
                if sc > best_score:
                    best_score, best_action = sc, action
        if best_action is None:
            best_action = valid[0]
        return best_action, best_score, {"valid": True}


def evaluate_heuristic_agent(agent, n_episodes=50, max_steps=2048):
    """评估启发式搜索智能体"""
    env = Game2048Env()
    scores, max_tiles, steps_list, wins = [], [], [], 0
    for _ in range(n_episodes):
        state = env.reset(); done = False; st = 0
        while not done and st < max_steps:
            action, _, _ = agent.select_action(state)
            if action is None: break
            _, _, done, _ = env.step(action); st += 1
        scores.append(env.score)
        max_tiles.append(int(env.board.max()))
        steps_list.append(env.steps)
        if int(env.board.max()) >= 2048: wins += 1
    stats = {"mean_score": float(np.mean(scores)),"max_score": float(np.max(scores)),
             "mean_max_tile": float(np.mean(max_tiles)),"best_max_tile": int(np.max(max_tiles)),
             "mean_steps": float(np.mean(steps_list)),"win_rate": wins / n_episodes}
    for ms in [128, 256, 512, 1024, 2048]:
        stats[f"tile_{ms}_rate"] = sum(1 for mt in max_tiles if mt >= ms) / n_episodes
    return stats


def run_heuristic_eval():
    print("=" * 60)
    print("Part 2: 启发式搜索算法评估")
    print("=" * 60)
    for depth in [1, 2, 3]:
        agent = HeuristicSearchAgent(search_depth=depth, use_expectimax=True)
        stats = evaluate_heuristic_agent(agent, n_episodes=50)
        print(f"\nDepth={depth} | 50 episodes:")
        for k, v in stats.items():
            if "rate" in k:
                print(f"  {k}: {v*100:.1f}%")
            else:
                print(f"  {k}: {v:.1f}")

if __name__ == "__main__":
    run_heuristic_eval()
