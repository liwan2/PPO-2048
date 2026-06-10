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

0. **退出**
1. **训练 PPO** — 强化学习训练
2. **启动可视化界面** — 支持 3 种模型切换（启发式搜索 / 监督学习 / PPO）
3. **评估启发式搜索** — 搜索算法的性能评估（模型 1）
4. **遗传算法优化** — 进化算法优化启发式参数（模型 2）
5. **监督学习 — 启发式搜索数据** — 用启发式搜索策略采集数据训练（模型 3a）
6. **监督学习 — PPO 冠军模型数据** — 用强化学习冠军模型采集轨迹训练（模型 3b）
7. **快速验证** — 50 次 PPO 更新快速查看效果

### 单独运行

```bash
# 训练 PPO
python train.py

# 启动 UI (模型选择)
python ui.py

# 启发式搜索评估
python heuristic_search.py

# 遗传算法优化
python evo_optimize.py

# 监督学习 — 启发式搜索数据
python supervised_train.py

# 监督学习 — PPO 冠军模型数据
python -c "import supervised_train; supervised_train.run_supervised_training_from_championship()"
```

## 实现的方法

### 模型 1: 启发式搜索 (Heuristic Search)
基于棋盘特征（空格数、单调性、平滑度、角落/边缘布局、合并潜力）的评估函数，结合 **Expectimax** 前向搜索（默认深度 3），对每个合法动作后的棋盘状态评分，选择最优动作。

### 模型 2: 遗传算法优化 (Genetic Algorithm)
在启发式搜索策略的基础上，将评估函数中的 7 个权重参数作为优化对象，使用遗传算法（锦标赛选择、均匀交叉、高斯变异）自动搜索更优参数组合。

### 模型 3: 监督学习 (Supervised Learning)
训练 **SupervisedNet** 神经网络（3 层 MLP: 256→128→64→4）进行 2048 游戏决策。支持两种数据来源:
- **方法 A**: 以启发式搜索策略的决策为标签，训练神经网络模仿专家行为
- **方法 B**: 使用 PPO 冠军模型（championship_model/）采集高质量轨迹进行训练

监督学习模型提供 select_action 接口，可在可视化界面中与启发式搜索、PPO 模型切换对比。

### 模型 4: PPO 强化学习 (Reinforcement Learning)
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
