# 2048 — 人工智能实验

基于 **Python + PyTorch + Pygame** 的 2048 游戏 AI 项目，综合运用了多种人工智能方法。

## 项目结构

| 文件 | 说明 |
|------|------|
| config.py | 超参数与配置（PPO、奖励函数、UI 尺寸等） |
| env.py | 2048 游戏环境（四方向滑动、合并、计分） |
| model.py | Actor-Critic 神经网络模型 |
| agent.py | PPO Agent（含 RolloutBuffer、GAE、训练循环） |
| train.py | PPO 强化学习训练（断点续训） |
| cross.py | **启发式搜索 + 遗传算法**（合并实现） |
| supervised_train.py | **监督学习**（支持两种数据源训练） |
| ui.py | Pygame 可视化（3 种模型一键切换） |
| main.py | 集成菜单，统一入口 |
| run_short.py | 快速验证 PPO（50 次更新） |
| championship_model/ | PPO 冠军模型权重 |
| 2048_ppo_model.pth | PPO 训练 checkpoint |
| 2048_ppo_best.pth | PPO 最佳模型 |
| 2048_supervised_model.pth | 监督学习训练结果 |
| legacy/ | 早期测试脚本 |

## 快速开始

### 环境依赖

- Python 3.9+
- PyTorch
- Pygame
- NumPy

用已有的虚拟环境激活:

`bash
# 如使用项目自带的虚拟环境
.\.2048ve\Scripts\Activate.ps1
`

### 运行

`bash
python main.py
`

交互菜单支持:

1. **UI 调用** — 可视化界面，可切换启发式搜索（cross.py）/ 监督学习 / PPO 三种模型
2. **PPO 训练 (断点续训)** — 加载已有 checkpoint 继续训练，没有则从头开始
3. **监督学习 — 启发式搜索数据** — 用启发式搜索采集数据训练监督网络
4. **监督学习 — PPO 冠军模型数据** — 用冠军模型采集轨迹训练监督网络
5. **启发式算法 + 遗传算法优化** — 先评估启发式搜索表现，再用遗传算法优化权重
6. **退出**

### 单独运行

```bash
# 训练 PPO
python train.py

# 启动 UI (模型选择)
python ui.py

# 启发式评估 + GA 优化
python cross.py

# 监督学习
python supervised_train.py

# 监督学习 — PPO 冠军模型数据
python -c "import supervised_train; supervised_train.run_supervised_training_from_championship()"
```

## 实现的方法

### 启发式搜索 + 遗传算法 (cross.py)
启发式评估函数基于 6 个核心特征（空格数、平滑度、单调性、角落布局、可合并对、蛇形有序度）及 3 个交叉特征，结合 **Expectimax** 搜索（默认深度 3）。
使用遗传算法（锦标赛选择、两点交叉、单点变异）自动优化 9 维权重参数。

### 监督学习 (Supervised Learning)
训练 **SupervisedNet**（3 层 MLP: 256→128→64→4）进行决策。支持两种数据源:
- **方法 A**: 以启发式搜索的决策为标签
- **方法 B**: 使用 PPO 冠军模型（championship_model/）采集轨迹

三种模型可在 UI 中一键切换对比。

### PPO 强化学习 (Reinforcement Learning)
使用 **Proximal Policy Optimization** 从零开始训练，含多环境并行采样、GAE-Lambda、熵衰减、早停等机制。

## 配置

所有可调参数集中在 config.py，包括:

- 游戏规则（棋盘大小、获胜条件）
- PPO 超参数（学习率、GAE λ、clip ε、mini-batch 大小等）
- 奖励函数权重（空格、单调性、平滑度、角落、合并等）
- 训练监控（评估频率、早停条件、模型保存策略）
- UI 布局

## 许可

仅用于教学与学习目的。
