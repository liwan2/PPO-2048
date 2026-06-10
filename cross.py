import random
import copy
from typing import List, Tuple

# ===================== 1. 2048 游戏核心引擎 =====================
class Game2048:
    def __init__(self):
        self.size = 4
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.score = 0
        # 初始化棋盘，随机生成两个初始方块
        self.add_random_tile()
        self.add_random_tile()

    def reset(self):
        """重置游戏"""
        self.board = [[0]*self.size for _ in range(self.size)]
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        """在空位随机生成 2(90%) 或 4(10%)"""
        empty_cells = [
            (i, j) for i in range(self.size)
            for j in range(self.size) if self.board[i][j] == 0
        ]
        if not empty_cells:
            return
        x, y = random.choice(empty_cells)
        self.board[x][y] = 2 if random.random() < 0.9 else 4

    def _compress(self, row: List[int]) -> List[int]:
        """单行压缩：去除空位，靠左排列"""
        new_row = [num for num in row if num != 0]
        new_row += [0] * (self.size - len(new_row))
        return new_row

    def _merge(self, row: List[int]) -> Tuple[List[int], int]:
        """单行合并相邻相同数字，返回新行 + 本次合并得分"""
        row = self._compress(row)
        add_score = 0
        for i in range(self.size - 1):
            if row[i] != 0 and row[i] == row[i+1]:
                row[i] *= 2
                add_score += row[i]
                row[i+1] = 0
        row = self._compress(row)
        return row, add_score

    def move_left(self) -> bool:
        """向左移动，返回是否发生有效移动"""
        moved = False
        for i in range(self.size):
            old_row = self.board[i].copy()
            new_row, s = self._merge(self.board[i])
            if new_row != old_row:
                moved = True
                self.board[i] = new_row
                self.score += s
        return moved

    def move_right(self) -> bool:
        """向右移动"""
        moved = False
        for i in range(self.size):
            old_row = self.board[i].copy()
            rev_row = self.board[i][::-1]
            new_row, s = self._merge(rev_row)
            new_row = new_row[::-1]
            if new_row != old_row:
                moved = True
                self.board[i] = new_row
                self.score += s
        return moved

    def move_up(self) -> bool:
        """向上移动"""
        moved = False
        for j in range(self.size):
            col = [self.board[i][j] for i in range(self.size)]
            old_col = col.copy()
            new_col, s = self._merge(col)
            if new_col != old_col:
                moved = True
                self.score += s
                for i in range(self.size):
                    self.board[i][j] = new_col[i]
        return moved

    def move_down(self) -> bool:
        """向下移动"""
        moved = False
        for j in range(self.size):
            col = [self.board[i][j] for i in range(self.size)]
            old_col = col.copy()
            rev_col = col[::-1]
            new_col, s = self._merge(rev_col)
            new_col = new_col[::-1]
            if new_col != old_col:
                moved = True
                self.score += s
                for i in range(self.size):
                    self.board[i][j] = new_col[i]
        return moved

    def move(self, direction: int) -> bool:
        """统一移动接口：0=上,1=右,2=下,3=左"""
        if direction == 0:
            return self.move_up()
        elif direction == 1:
            return self.move_right()
        elif direction == 2:
            return self.move_down()
        elif direction == 3:
            return self.move_left()
        return False

    def is_game_over(self) -> bool:
        """判断游戏是否结束：无空位 且 无相邻可合并数字"""
        # 检查是否有空位
        for row in self.board:
            if 0 in row:
                return False
        # 检查水平相邻
        for i in range(self.size):
            for j in range(self.size-1):
                if self.board[i][j] == self.board[i][j+1]:
                    return False
        # 检查垂直相邻
        for j in range(self.size):
            for i in range(self.size-1):
                if self.board[i][j] == self.board[i+1][j]:
                    return False
        return True

    def get_board(self) -> List[List[int]]:
        """获取棋盘副本（防止篡改原棋盘）"""
        return copy.deepcopy(self.board)

    def print_board(self):
        """打印棋盘"""
        for row in self.board:
            print(f"{row} | Score: {self.score}")
        print("-" * 30)

# ===================== 2. 启发式评价函数 & 特征提取 =====================
# 5个核心启发特征（权重由遗传算法优化）
# 特征1: 空格数量 (越多越好)
# 特征2: 平滑度 (相邻格子差值越小越好，取负值)
# 特征3: 单调性 (行/列单调递增/递减，越大越好)
# 特征4: 最大值在角落 (高分奖励)
# 特征5: 相邻可合并方块数 (越多越好)

def count_empty(board: List[List[int]]) -> int:
    """特征1：统计空格数"""
    cnt = 0
    size = len(board)
    for i in range(size):
        for j in range(size):
            if board[i][j] == 0:
                cnt += 1
    return cnt

def calc_smoothness(board: List[List[int]]) -> float:
    """特征2：平滑度，相邻格子差值之和（越小越好，最终取负）"""
    smooth = 0.0
    size = len(board)
    for i in range(size):
        for j in range(size):
            val = board[i][j]
            if val == 0:
                continue
            # 右邻
            if j + 1 < size and board[i][j+1] != 0:
                smooth -= abs(val - board[i][j+1])
            # 下邻
            if i + 1 < size and board[i+1][j] != 0:
                smooth -= abs(val - board[i+1][j])
    return smooth

def calc_monotonicity(board: List[List[int]]) -> float:
    """特征3：单调性，行/列单调趋势得分"""
    size = len(board)
    mono = 0.0
    # 行单调性
    for row in board:
        for j in range(size-1):
            if row[j] >= row[j+1]:
                mono += row[j] - row[j+1]
    # 列单调性
    for j in range(size):
        for i in range(size-1):
            if board[i][j] >= board[i+1][j]:
                mono += board[i][j] - board[i+1][j]
    return mono

def max_in_corner(board: List[List[int]]) -> int:
    """特征4：最大值是否在四个角落（是则加分）"""
    size = len(board)
    max_val = max(max(row) for row in board)
    corners = [board[0][0], board[0][size-1], board[size-1][0], board[size-1][size-1]]
    return 1 if max_val in corners else 0

def count_mergeable(board: List[List[int]]) -> int:
    """特征5：相邻可合并方块数量"""
    cnt = 0
    size = len(board)
    # 水平
    for i in range(size):
        for j in range(size-1):
            if board[i][j] == board[i][j+1] and board[i][j] != 0:
                cnt += 1
    # 垂直
    for j in range(size):
        for i in range(size-1):
            if board[i][j] == board[i+1][j] and board[i][j] != 0:
                cnt += 1
    return cnt

def calc_snake_order(board: List[List[int]]) -> float:
    """特征6：蛇形有序度 (S型单调递减矩阵匹配)"""
    # 构造更偏向于经典蛇形摆法的权值矩阵
    SNAKE_MATRIX = [
        [32768, 16384, 8192, 4096],
        [256,   512,  1024, 2048],
        [128,    64,    32,   16],
        [1,      2,     4,    8]
    ]
    score = 0.0
    for i in range(4):
        for j in range(4):
            score += board[i][j] * SNAKE_MATRIX[i][j]
    return score

def evaluate_board(board: List[List[int]], weights: List[float]) -> float:
    """
    启发式评价函数：加权求和得到棋盘状态总分
    :param board: 4x4棋盘
    :param weights: 9维权重 
    :return: 状态得分（越高状态越优）
    """
    # 基础特征
    f1 = count_empty(board)
    f2 = calc_smoothness(board)
    f3 = calc_monotonicity(board)
    f4 = max_in_corner(board)
    f5 = count_mergeable(board)
    f6 = calc_snake_order(board)
    
    # 组合交叉特征
    f7 = f1 * f2  # (空格数 × 平滑度)
    f8 = f2 * f3  # (平滑度 × 单调性)
    f9 = f4 * f3  # (最大值角落 × 单调性)

    # 加权计算总分
    score = (
        weights[0] * f1
        + weights[1] * f2
        + weights[2] * f3
        + weights[3] * f4
        + weights[4] * f5
        + weights[5] * f6
        + weights[6] * f7
        + weights[7] * f8
        + weights[8] * f9
    )
    return score
# ===================== 3. 期望极大算法 (Expectimax AI 决策) =====================
def simulate_move(board: List[List[int]], direction: int) -> Tuple[List[List[int]], bool]:
    """模拟一次移动，返回新棋盘 + 是否有效移动"""
    game = Game2048()
    game.board = copy.deepcopy(board)
    moved = game.move(direction)
    return game.get_board(), moved

def search_best_move(
    board: List[List[int]],
    weights: List[float],
    depth: int = 2
) -> int:
    """
    Expectimax 搜索入口：遍历4个方向，寻找期望得分最高的动作
    :return: 最优方向 0上/1右/2下/3左
    """
    dirs = [0, 1, 2, 3]
    best_dir = -1
    best_score = -float('inf')

    for d in dirs:
        new_board, moved = simulate_move(board, d)
        if not moved:
            continue  # 跳过无效移动

        # 玩家移动后交由环境生成随机方块（Chance Node）
        current_score = expectimax_chance(new_board, weights, depth - 1)
        if current_score > best_score:
            best_score = current_score
            best_dir = d
            
    return best_dir if best_dir != -1 else 0

def expectimax_max(board: List[List[int]], weights: List[float], depth: int) -> float:
    """Max Node：玩家回合，计算所有合法移动中的最高期望得分"""
    if depth == 0:
        return evaluate_board(board, weights)

    best_score = -float('inf')
    moved_any = False

    for d in [0, 1, 2, 3]:
        new_board, moved = simulate_move(board, d)
        if moved:
            moved_any = True
            # 交给环境层计算这种移动后的期望得分
            score = expectimax_chance(new_board, weights, depth - 1)
            if score > best_score:
                best_score = score

    if not moved_any:
        return evaluate_board(board, weights)
        
    return best_score

def expectimax_chance(board: List[List[int]], weights: List[float], depth: int, max_sample: int = 4) -> float:
    """Chance Node：环境回合，穷举所有可能的空位生成 2 或 4，按概率加权求和"""
    size = len(board)
    empty_cells = [(i, j) for i in range(size) for j in range(size) if board[i][j] == 0]
    
    if not empty_cells or depth == 0:
        return evaluate_board(board, weights)

    expected_score = 0.0
    num_empty = len(empty_cells)
    
    # 空格太多时随机采样，避免搜索树爆炸（depth=0 时全部评估）
    if num_empty > max_sample and depth > 0:
        empty_cells = random.sample(empty_cells, max_sample)
    
    # 根据2048规则计算生成的概率权重
    prob_2 = 0.9 / num_empty
    prob_4 = 0.1 / num_empty

    for x, y in empty_cells:
        # 分支1：生成 2
        board[x][y] = 2
        expected_score += prob_2 * expectimax_max(board, weights, depth)
        
        # 分支2：生成 4
        board[x][y] = 4
        expected_score += prob_4 * expectimax_max(board, weights, depth)
        
        # 原地回溯还原空位状态，避免昂贵的深拷贝操作
        board[x][y] = 0

    return expected_score

# ===================== 4. 遗传算法 GA 优化权重参数 =====================
# 超参数（可自行调整）
POPULATION_SIZE = 20    # 种群规模
GENERATIONS = 3        # 进化代数
CROSS_RATE = 0.7        # 交叉概率
MUTATE_RATE = 0.1       # 变异概率
WEIGHT_RANGE = (-10.0, 10.0)  # 权重取值范围
CHROMOSOME_LEN = 9      # 染色体长度 = 5个特征权重

def create_individual() -> List[float]:
    """创建单个个体（随机5维权重）"""
    return [random.uniform(*WEIGHT_RANGE) for _ in range(CHROMOSOME_LEN)]

def init_population() -> List[List[float]]:
    """初始化种群"""
    return [create_individual() for _ in range(POPULATION_SIZE)]

def run_ai_game(weights: List[float], max_steps: int = 1000) -> int:
    """
    用一组权重运行AI玩一局2048，返回最终得分（适应度）
    """
    game = Game2048()
    step = 0
    while not game.is_game_over() and step < max_steps:
        board = game.get_board()
        best_dir = search_best_move(board, weights, depth=3)
        if not game.move(best_dir):
            break  # 停止无效移动导致的死循环
        game.add_random_tile()
        step += 1
    return game.score

def fitness(individual: List[float], test_times: int = 3) -> float:
    """
    适应度函数：单个体运行多局游戏，取平均得分
    得分越高，适应度越高
    """
    total = 0
    for _ in range(test_times):
        total += run_ai_game(individual)
    return total / test_times

def select(pop: List[List[float]], fit_list: List[float]) -> List[List[float]]:
    """锦标赛选择：选择优秀个体，避免早熟收敛"""
    new_pop = []
    for _ in range(len(pop)):
        # 随机选3个个体，保留最优
        candidates = random.sample(list(zip(pop, fit_list)), 3)
        candidates.sort(key=lambda x: x[1], reverse=True)
        new_pop.append(copy.deepcopy(candidates[0][0]))
    return new_pop

def cross(parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
    """两点交叉"""
    if random.random() < CROSS_RATE:
        p1 = random.randint(1, CHROMOSOME_LEN - 1)
        p2 = random.randint(p1, CHROMOSOME_LEN - 1)
        child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
        child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(individual: List[float]) -> List[float]:
    """单点变异：小幅扰动权重"""
    if random.random() < MUTATE_RATE:
        idx = random.randint(0, CHROMOSOME_LEN - 1)
        # 在原权重附近随机扰动
        delta = random.uniform(-1.0, 1.0)
        individual[idx] += delta
        # 限制权重范围
        individual[idx] = max(WEIGHT_RANGE[0], min(WEIGHT_RANGE[1], individual[idx]))
    return individual

def genetic_algorithm() -> Tuple[List[float], List[float]]:
    """遗传算法主流程，返回最优权重 + 每代最优适应度"""
    pop = init_population()
    best_fit_history = []
    best_ind = None

    for gen in range(GENERATIONS):
        # 1. 计算所有个体适应度
        fit_list = [fitness(ind) for ind in pop]
        current_best_idx = fit_list.index(max(fit_list))
        current_best_fit = fit_list[current_best_idx]
        current_best_ind = pop[current_best_idx]

        # 记录历史最优
        if best_ind is None or current_best_fit > max(best_fit_history):
            best_ind = current_best_ind.copy()
        best_fit_history.append(current_best_fit)

        print(f"第{gen+1}代 | 最优得分: {current_best_fit:.2f} | 最优权重: {current_best_ind}")

        # 2. 选择
        new_pop = select(pop, fit_list)
        # 3. 交叉
        for i in range(0, POPULATION_SIZE - 1, 2):
            new_pop[i], new_pop[i+1] = cross(new_pop[i], new_pop[i+1])
        # 4. 变异
        for i in range(POPULATION_SIZE):
            new_pop[i] = mutate(new_pop[i])

        pop = new_pop

    return best_ind, best_fit_history

# ===================== 5. 测试运行函数 =====================
def test_fixed_weights_ai():
    """测试固定权重的启发式AI"""
    print("===== 测试固定权重 AI =====")
    # 人工预设一组初始权重 (加入衍生与交叉项后，扩展至9个权重)
    fixed_weights = [-5.476019308259945, -9.922720223591403, 7.284119005539409, -6.1761117007068655, 6.681805197965257, -2.902172783121964, -3.3344140748100166, 5.131427690395551, -3.1109349782006657]
    game = Game2048()
    step = 0
    while not game.is_game_over() and step < 800:
        board = game.get_board()
        best_dir = search_best_move(board, fixed_weights, depth=3)
        if not game.move(best_dir):
            break
        game.add_random_tile()
        step += 1
        if step % 100 == 0:
            game.print_board()
    print(f"游戏结束，最终得分: {game.score}")
    game.print_board()

def test_optimized_ai(best_weights: List[float]):
    """测试遗传算法优化后的权重AI"""
    print("\n===== 测试遗传算法优化后权重 AI =====")
    game = Game2048()
    step = 0
    while not game.is_game_over() and step < 2000:
        board = game.get_board()
        best_dir = search_best_move(board, best_weights, depth=3)
        if not game.move(best_dir):
            break
        game.add_random_tile()
        step += 1
        if step % 100 == 0:
            game.print_board()
    print(f"游戏结束，最终得分: {game.score}")
    game.print_board()

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 1. 先测试人工固定权重的AI
    test_fixed_weights_ai()

    # 2. 运行遗传算法，优化评价函数权重
    print("\n========== 开始遗传算法优化权重 ==========")
    optimal_weights, fit_history = genetic_algorithm()
    print(f"\n进化完成！全局最优权重: {optimal_weights}")

    # 3. 使用优化后的最优权重测试AI
    test_optimized_ai(optimal_weights)
# ===================== 6. 接口适配 (与 UI / supervised_train 兼容) =====================
import numpy as np
from config import N

DEFAULT_WEIGHTS = [-5.476019308259945, -9.922720223591403, 7.284119005539409, -6.1761117007068655, 6.681805197965257, -2.902172783121964, -3.3344140748100166, 5.131427690395551, -3.1109349782006657]
#[2.0, 1.5, 1.0, 3.0, 2.5, 0.001, 0.5, 0.1, 0.5]


class HeuristicAgent:

    def __init__(self, weights=None, depth=3):
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS.copy()
        self.depth = depth

    def get_weights(self):
        return self.weights.copy()

    def select_action(self, state):
        # 解码：log2 numpy state → List[List[int]]
        board = [[0] * 4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                v = int(state[r, c])
                if v > 0:
                    board[r][c] = 1 << v

        # 注意：内部 Game2048 使用的方向编码为 0=上,1=右,2=下,3=左
        # 而 env.Game2048Env 使用的编码为 0=上,1=下,2=左,3=右
        # 需要在返回给外部（UI/env）前做映射。
        game_to_env = [0, 3, 1, 2]

        valid = [d for d in range(4) if simulate_move(board, d)[1]]
        if not valid:
            return None, 0.0, {"valid": False}
        if len(valid) == 1:
            return game_to_env[valid[0]], 0.0, {"valid": True, "forced": True}

        action = search_best_move(board, self.weights, self.depth)
        if action not in valid:
            action = valid[0]
        return game_to_env[action], 0.0, {"valid": True}


def evaluate_agent(agent, n_episodes=50, max_steps=2048):
    """使用 env.Game2048Env 评估任意 agent 的游戏表现。"""
    from env import Game2048Env
    env = Game2048Env()
    scores, max_tiles, steps_list, wins = [], [], [], 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        st = 0
        while not done and st < max_steps:
            action, _, _ = agent.select_action(state)
            if action is None:
                break
            _, _, done, _ = env.step(action)
            st += 1
        scores.append(env.score)
        max_tiles.append(int(env.board.max()))
        steps_list.append(env.steps)
        if int(env.board.max()) >= 2048:
            wins += 1
    stats = {"mean_score": float(np.mean(scores)), "max_score": float(np.max(scores)),
             "mean_max_tile": float(np.mean(max_tiles)), "best_max_tile": int(np.max(max_tiles)),
             "mean_steps": float(np.mean(steps_list)), "win_rate": wins / n_episodes}
    for ms in [128, 256, 512, 1024, 2048]:
        stats[f"tile_{ms}_rate"] = sum(1 for mt in max_tiles if mt >= ms) / n_episodes
    return stats


def run_heuristic_eval():
    """评估不同深度下的启发式搜索表现"""
    print("=" * 60)
    print("启发式搜索算法评估")
    print("=" * 60)
    for depth in [1, 2, 3]:  # depth=3 太慢，仅供评估非UI使用
        agent = HeuristicAgent(depth=depth)
        stats = evaluate_agent(agent, n_episodes=50)
        print(f"\\nDepth={depth} | 50 episodes:")
        for k, v in stats.items():
            if "rate" in k:
                print(f"  {k}: {v*100:.1f}%")
            else:
                print(f"  {k}: {v:.1f}")


def run_ga_optimization():
    """运行遗传算法优化权重 + 对比评估"""
    print("=" * 60)
    print("遗传算法优化启发式参数")
    print("=" * 60)

    best_weights, history = genetic_algorithm()
    print(f"\\n优化完成！最优权重: {best_weights}")

    print("\\n对比优化前后效果...")
    default_agent = HeuristicAgent(depth=3)
    default_stats = evaluate_agent(default_agent, n_episodes=30)
    print(f"\\n默认权重  | MeanScore={default_stats['mean_score']:.1f} | "
          f"MaxTile={default_stats['mean_max_tile']:.0f} | "
          f"512Rate={default_stats['tile_512_rate']*100:.1f}%")

    opt_agent = HeuristicAgent(weights=best_weights, depth=3)
    opt_stats = evaluate_agent(opt_agent, n_episodes=30)
    print(f"GA优化后  | MeanScore={opt_stats['mean_score']:.1f} | "
          f"MaxTile={opt_stats['mean_max_tile']:.0f} | "
          f"512Rate={opt_stats['tile_512_rate']*100:.1f}%")
    return best_weights