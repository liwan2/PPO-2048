"""Part 4: 监督学习 — 用启发式搜索 / PPO 冠军模型采集数据训练神经网络"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import N, ACTION_DIM, SUPERVISED_MODEL_PATH
from env import Game2048Env


# ── 数据集采集 ────────────────────────────────────────────────────────

def collect_data(agent, n_episodes=50000, max_steps=1024):
    """使用给定智能体采集 (棋盘状态, 动作) 标记数据"""
    env = Game2048Env()
    states, actions = [], []
    action_counts = np.zeros(ACTION_DIM, dtype=np.int64)
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action, _, _ = agent.select_action(state)
            if action is None:
                break
            states.append(state.copy())
            actions.append(action)
            action_counts[action] += 1
            _, _, done, _ = env.step(action)
            steps += 1
        if (ep + 1) % 100 == 0:
            print(f"  Collected {ep + 1}/{n_episodes} episodes...")
    total = len(actions)
    dist = {['Up','Down','Left','Right'][i]: f'{c/total*100:.1f}%'
            for i, c in enumerate(action_counts)}
    print(f"  Action distribution: {dist}")
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)


def collect_data_from_ppo(model_paths=None, n_episodes=50000, max_steps=1024):
    """使用 PPO 冠军模型采集 (棋盘状态, 动作) 标记数据。

    Args:
        model_paths: PPO 模型权重文件列表, 交替使用
        n_episodes:  采集局数
        max_steps:   每局最大步数

    Returns:
        (states, actions)  numpy arrays
    """
    from agent import Agent

    if model_paths is None:
        model_paths = [
            "championship_model/2048_ppo_best_2.0.pth",
            "championship_model/2048_ppo_best_1.0.pth",
        ]

    env = Game2048Env()
    states, actions = [], []
    action_counts = np.zeros(ACTION_DIM, dtype=np.int64)
    agents = []

    for mdl_path in model_paths:
        if os.path.exists(mdl_path):
            agent = Agent()
            try:
                ckpt = torch.load(mdl_path, map_location="cpu", weights_only=False)
                if "model_state" in ckpt:
                    agent.net.load_state_dict(ckpt["model_state"])
                else:
                    agent.net.load_state_dict(ckpt)
            except Exception as e:
                print(f"  Skipping {mdl_path}: {e}")
                continue
            agent.net.eval()
            agents.append(agent)
            print(f"  Loaded PPO model from {mdl_path}")

    # 没有冠军模型时回退到启发式搜索
    if not agents:
        print("  No PPO models found, falling back to heuristic search")
        from cross import HeuristicAgent
        agents.append(HeuristicSearchAgent(search_depth=2, use_expectimax=True))

    for ep in range(n_episodes):
        agent_obj = agents[ep % len(agents)]
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            if isinstance(agent_obj, Agent):
                action, _, _ = agent_obj.select_action(state, evaluate=True)
            else:
                action, _, _ = agent_obj.select_action(state)
            if action is None:
                break
            states.append(state.copy())
            actions.append(action)
            action_counts[action] += 1
            _, _, done, _ = env.step(action)
            steps += 1
        if (ep + 1) % 100 == 0:
            print(f"  Collected {ep + 1}/{n_episodes} episodes...")

    total = len(actions)
    dist = {['Up','Down','Left','Right'][i]: f'{c/total*100:.1f}%'
            for i, c in enumerate(action_counts)}
    print(f"  Action distribution: {dist}")
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)


# ── 数据集封装 ────────────────────────────────────────────────────────

class GameDataset(Dataset):
    """2048 (state, action) 数据集。"""
    def __init__(self, states, actions):
        self.states = torch.from_numpy(states).float()
        self.actions = torch.from_numpy(actions).long()

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        s = self.states[idx].view(-1) / 16.0   # 归一化到 [0, 1]
        return s, self.actions[idx]


# ── 监督学习网络 ──────────────────────────────────────────────────────

class SupervisedNet(nn.Module):
    """三层 MLP 分类网络, 输入 4x4 棋盘 log2 编码, 输出 4 动作 logits。"""
    def __init__(self):
        super().__init__()
        in_features = N * N
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, ACTION_DIM)

    def forward(self, x):
        feat = self.net(x)
        return self.classifier(feat)

    @torch.no_grad()
    def predict(self, state, valid_mask=None):
        """单步推理, 返回动作 (已屏蔽非法动作)。"""
        self.eval()
        s = torch.from_numpy(state).float().view(1, -1) / 16.0
        logits = self(s)
        if valid_mask is not None:
            mask_t = torch.from_numpy(valid_mask).float()
            logits = logits.masked_fill(mask_t <= 0, -1e8)
        return int(logits.argmax(dim=-1).item())


# ── 训练 ──────────────────────────────────────────────────────────────

def train_supervised(model, train_loader, val_loader, epochs=20,
                     lr=1e-3, device="cpu"):
    """训练监督学习分类模型。

    Returns:
        float: 最佳验证准确率 (%)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad()
            logits = model(states)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * states.size(0)
            _, preds = logits.max(1)
            train_correct += (preds == actions).sum().item()
            train_total += states.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                logits = model(states)
                loss = criterion(logits, actions)
                val_loss += loss.item() * states.size(0)
                _, preds = logits.max(1)
                val_correct += (preds == actions).sum().item()
                val_total += states.size(0)

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100
        print(f"Epoch {epoch}/{epochs} | "
              f"TrainLoss: {train_loss/train_total:.4f} | "
              f"TrainAcc: {train_acc:.2f}% | "
              f"ValAcc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        scheduler.step()

    return best_val_acc


# ── SupervisedAgent (包装 SupervisedNet 为统一接口) ──────────────────

class SupervisedAgent:
    """包装 SupervisedNet, 提供与 HeuristicSearchAgent / Agent 一致的接口。"""
    def __init__(self, net):
        self.net = net
        self.net.eval()

    def select_action(self, state):
        board = np.zeros((N, N), dtype=np.int64)
        nz = state > 0
        board[nz] = (1 << state[nz].astype(np.int64))

        valid = [a for a in range(ACTION_DIM)
                 if Game2048Env._preview_board(board, a) is not None]
        if not valid:
            return None, 0.0, {"valid": False}
        if len(valid) == 1:
            return valid[0], 0.0, {"valid": True, "forced": True}

        mask = np.zeros(ACTION_DIM, dtype=np.float32)
        for v in valid:
            mask[v] = 1.0
        action = self.net.predict(state, valid_mask=mask)
        if action not in valid:
            action = valid[0]
        return action, 0.0, {"valid": True}

    @classmethod
    def load(cls, path=None):
        p = path or SUPERVISED_MODEL_PATH
        net = SupervisedNet()
        if os.path.exists(p):
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            if "model_state" in ckpt:
                net.load_state_dict(ckpt["model_state"])
            else:
                net.load_state_dict(ckpt)
            print(f"Loaded supervised model from {p}")
        net.eval()
        return cls(net)


# ── 评估辅助 ──────────────────────────────────────────────────────────

def evaluate_supervised_model(model, n_episodes=50, max_steps=1024):
    """评估监督学习模型在 2048 上的游戏表现。"""
    from cross import evaluate_agent
    agent = SupervisedAgent(model)
    stats = evaluate_agent(agent, n_episodes=n_episodes)
    return stats


# ── 训练入口 ──────────────────────────────────────────────────────────

def _train_from_data(states, actions, epochs=15):
    """共用训练流水线: 划分数据集 → 训练 → 评估。"""
    indices = np.random.permutation(len(states))
    split = int(len(indices) * 0.8)
    train_s, train_a = states[indices[:split]], actions[indices[:split]]
    val_s, val_a = states[indices[split:]], actions[indices[split:]]
    print(f"  训练集: {len(train_s)} | 验证集: {len(val_s)}")

    train_loader = DataLoader(GameDataset(train_s, train_a),
                              batch_size=128, shuffle=True)
    val_loader = DataLoader(GameDataset(val_s, val_a),
                            batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    model = SupervisedNet()
    best_acc = train_supervised(model, train_loader, val_loader,
                                epochs=epochs, device=device)
    print(f"\n  最佳验证准确率: {best_acc:.2f}%")
    return model


def run_supervised_training():
    """方法 A: 用启发式搜索数据训练监督学习模型。"""
    print("=" * 60)
    print("Part 4: 监督学习 — 方法 A (启发式搜索数据)")
    print("=" * 60)

    print("\n[Step 1] 用启发式搜索智能体采集数据...")
    from cross import HeuristicAgent
    teacher = HeuristicSearchAgent(search_depth=2, use_expectimax=True)
    states, actions = collect_data(teacher, n_episodes=50000)
    print(f"  采集完成: {len(states)} 条样本")

    print("\n[Step 2] 训练...")
    model = _train_from_data(states, actions, epochs=15)

    print("\n[Step 3] 评估游戏表现...")
    stats = evaluate_supervised_model(model, n_episodes=50)
    for k, v in stats.items():
        if "rate" in k:
            print(f"  {k}: {v*100:.1f}%")
        else:
            print(f"  {k}: {v:.1f}")

    torch.save({"model_state": model.state_dict()}, SUPERVISED_MODEL_PATH)
    print(f"\n  模型已保存至 {SUPERVISED_MODEL_PATH}")
    return model


def run_supervised_training_from_championship():
    """方法 B: 用 PPO 冠军模型数据训练监督学习模型。"""
    print("=" * 60)
    print("Part 4b: 监督学习 — 方法 B (PPO 冠军模型数据)")
    print("=" * 60)

    print("\n[Step 1] 用冠军模型采集数据...")
    states, actions = collect_data_from_ppo(n_episodes=50000)
    print(f"  采集完成: {len(states)} 条样本")

    print("\n[Step 2] 训练...")
    model = _train_from_data(states, actions, epochs=15)

    print("\n[Step 3] 评估游戏表现...")
    stats = evaluate_supervised_model(model, n_episodes=50)
    for k, v in stats.items():
        if "rate" in k:
            print(f"  {k}: {v*100:.1f}%")
        else:
            print(f"  {k}: {v:.1f}")

    torch.save({"model_state": model.state_dict()}, SUPERVISED_MODEL_PATH)
    print(f"\n  模型已保存至 {SUPERVISED_MODEL_PATH}")
    return model


def run_all_supervised():
    """依次运行两种监督学习训练。"""
    print("\n" + "=" * 60)
    print("监督学习综合训练")
    print("=" * 60)
    print("\n--- 方法 A: 基于启发式搜索 ---")
    model_a = run_supervised_training()
    print("\n--- 方法 B: 基于 PPO 冠军模型 ---")
    model_b = run_supervised_training_from_championship()
    return model_a, model_b


if __name__ == "__main__":
    run_supervised_training()
