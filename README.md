# 2048 — 人工智能课程实验

基于 **Python + PyTorch + Pygame** 的 2048 游戏 AI 项目，综合运用了多种人工智能方法。

## 项目结构

| 文件 | 说明 |
|------|------|
| env.py | 2048 游戏环境（四方向滑动、合并、计分） |
| config.py | 超参数与配置（PPO、奖励函数、UI 尺寸等） |
| model.py | Actor-Critic 神经网络模型 |
| agent.py | PPO Agent（含 RolloutBuffer、GAE、训练循环） |
| 	rain.py | PPO 强化学习训练入口 |
| 
un_short.py | PPO 快速验证（50 次更新） |
| heuristic_search.py | 启发式搜索与评估（Part 2） |
| evo_optimize.py | 遗传算法优化启发式参数（Part 3） |
| supervised_train.py | 监督学习训练（Part 4） |
| ui.py | Pygame 可视化界面（支持人机对战与 AI 自动演示） |
| main.py | 集成菜单，统一入口 |
| legacy/ | 早期测试脚本（bench、diagnostic） |
| championship_model/ | 锦标赛模型权重（需配合代码加载） |

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

1. **训练 PPO** — 强化学习训练（Part 5）
2. **启动可视化界面** — Pygame 图形界面，可手动操作或观看 AI 演示
3. **评估启发式搜索** — 搜索算法的性能评估（Part 2）
4. **遗传算法优化** — 进化算法优化启发式参数（Part 3）
5. **监督学习训练** — 监督学习训练（Part 4）
6. **快速验证** — 50 次 PPO 更新快速查看效果

### 单独运行

`bash
# 训练 PPO
python train.py

# 启动 UI
python ui.py

# 启发式搜索评估
python heuristic_search.py

# 遗传算法优化
python evo_optimize.py

# 监督学习训练
python supervised_train.py
`

## 实现的方法

- **Part 2 — 启发式搜索**: 基于棋盘特征（空格数、单调性、平滑度、角落/边缘布局）的评估函数 + 前向搜索。
- **Part 3 — 遗传算法**: 使用进化策略自动搜索最优的启发式权重组合。
- **Part 4 — 监督学习**: 以启发式搜索的决策为标签，训练神经网络模仿专家行为。
- **Part 5 — PPO 强化学习**: 使用 Proximal Policy Optimization 从零开始训练，含多环境并行采样、GAE、熵衰减、早停等机制。

## 配置

所有可调参数集中在 config.py，包括:

- 游戏规则（棋盘大小、获胜条件）
- PPO 超参数（学习率、GAE λ、clip ε、mini-batch 大小等）
- 奖励函数权重（空格、单调性、平滑度、角落、合并等）
- 训练监控（评估频率、早停条件、模型保存策略）
- UI 布局

## 许可

仅用于教学与学习目的。
