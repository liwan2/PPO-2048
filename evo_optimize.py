"""Part 3: 进化计算 — 遗传算法优化启发式搜索策略参数"""

import random
import copy
import numpy as np
from config import N, ACTION_DIM
from env import Game2048Env


# ── 基因型: 启发式权重的编码 ──────────────────────────────────────────

WEIGHT_NAMES = ["empty", "monotonicity", "smoothness", "corner",
                "edge", "max_value", "merge_potential"]
WEIGHT_DEFAULT = [10.0, 5.0, 2.0, 4.0, 2.0, 1.0, 3.0]

def weights_to_genome(weights):
    return np.array([weights[n] for n in WEIGHT_NAMES], dtype=np.float64)

def genome_to_weights(genome):
    return {n: float(genome[i]) for i, n in enumerate(WEIGHT_NAMES)}


# ── 适应度评估 ────────────────────────────────────────────────────────

def evaluate_fitness(genome, n_games=10, max_steps=1024, search_depth=2):
    from heuristic_search import HeuristicSearchAgent   # 可移出到模块顶部
    weights = genome_to_weights(genome)
    agent = HeuristicSearchAgent(search_depth=search_depth, weights=weights, use_expectimax=True)
    env = Game2048Env()
    scores = []
    for _ in range(n_games):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action, _, _ = agent.select_action(state)
            if action is None:
                break
            state, _, done, _ = env.step(action)   # 关键：更新 state
            steps += 1
        scores.append(env.score)
    return float(np.mean(scores))


# ── 遗传算法 ──────────────────────────────────────────────────────────

def tournament_selection(population, fitnesses, tournament_size=3):
    """锦标赛选择"""
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = indices[0]
    for i in indices[1:]:
        if fitnesses[i] > fitnesses[best_idx]:
            best_idx = i
    return copy.deepcopy(population[best_idx])

def crossover(p1, p2):
    """均匀交叉"""
    child = np.zeros_like(p1)
    for i in range(len(p1)):
        child[i] = p1[i] if random.random() < 0.5 else p2[i]
    return child

def mutate(genome, rate=0.3, scale=0.4):
    """高斯变异 (确保权重非负)"""
    for i in range(len(genome)):
        if random.random() < rate:
            genome[i] += np.random.randn() * scale * genome[i]
            genome[i] = max(0.01, genome[i])
    return genome


class GeneticOptimizer:
    """遗传算法优化器"""

    def __init__(self, pop_size=30, n_generations=20, n_eval_games=10,
                 search_depth=2, mut_rate=0.3, mut_scale=0.4):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.n_eval_games = n_eval_games
        self.search_depth = search_depth
        self.mut_rate = mut_rate
        self.mut_scale = mut_scale

        # 初始化种群: 围绕默认值随机扰动
        self.default_genome = weights_to_genome(
            {n: v for n, v in zip(WEIGHT_NAMES, WEIGHT_DEFAULT)}
        )
        self.population = []
        for _ in range(pop_size):
            g = self.default_genome.copy()
            for i in range(len(g)):
                g[i] *= 1.0 + np.random.randn() * 0.5
                g[i] = max(0.01, g[i])
            self.population.append(g)

        self.best_genome = None
        self.best_fitness = float("-inf")
        self.history = []

    def run(self, verbose=True):
        """运行遗传算法"""
        for gen in range(1, self.n_generations + 1):
            # 评估适应度
            fitnesses = []
            for i, genome in enumerate(self.population):
                fit = evaluate_fitness(genome, n_games=self.n_eval_games,
                                       search_depth=self.search_depth)
                fitnesses.append(fit)
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_genome = copy.deepcopy(genome)

            gen_best = float(np.max(fitnesses))
            gen_avg = float(np.mean(fitnesses))
            self.history.append((gen_best, gen_avg))

            if verbose:
                print(f"Gen {gen}/{self.n_generations} | "
                      f"Best: {gen_best:.1f} | Avg: {gen_avg:.1f} | "
                      f"PopBestFit: {self.best_fitness:.1f}")

            # 选择、交叉、变异产生下一代
            next_pop = [copy.deepcopy(self.best_genome)]  # 精英保留
            while len(next_pop) < self.pop_size:
                p1 = tournament_selection(self.population, fitnesses)
                p2 = tournament_selection(self.population, fitnesses)
                child = crossover(p1, p2)
                child = mutate(child, rate=self.mut_rate, scale=self.mut_scale)
                next_pop.append(child)
            self.population = next_pop

        best_weights = genome_to_weights(self.best_genome)
        if verbose:
            print(f"\n优化完成!")
            print(f"最佳适应度: {self.best_fitness:.1f}")
            print(f"最佳权重: {best_weights}")
        return best_weights


def run_evo_optimization():
    """运行进化计算优化 (可直接运行)"""
    print("=" * 60)
    print("Part 3: 进化计算 — 遗传算法优化启发式参数")
    print("=" * 60)

    optimizer = GeneticOptimizer(
        pop_size=20, n_generations=10, n_eval_games=5,
        search_depth=2, mut_rate=0.3, mut_scale=0.4,
    )
    best_weights = optimizer.run(verbose=True)

    print("\n评估优化前后效果...")

    # 基线 (默认权重)
    from heuristic_search import HeuristicSearchAgent, evaluate_heuristic_agent
    default_agent = HeuristicSearchAgent(search_depth=2, use_expectimax=True)
    default_stats = evaluate_heuristic_agent(default_agent, n_episodes=30)
    print(f"\n默认权重  | MeanScore={default_stats['mean_score']:.1f} | "
          f"MaxTile={default_stats['mean_max_tile']:.0f} | "
          f"512Rate={default_stats['tile_512_rate']*100:.1f}%")

    # 优化后
    opt_agent = HeuristicSearchAgent(search_depth=2, weights=best_weights, use_expectimax=True)
    opt_stats = evaluate_heuristic_agent(opt_agent, n_episodes=30)
    print(f"GA优化后  | MeanScore={opt_stats['mean_score']:.1f} | "
          f"MaxTile={opt_stats['mean_max_tile']:.0f} | "
          f"512Rate={opt_stats['tile_512_rate']*100:.1f}%")

    return best_weights


if __name__ == "__main__":
    run_evo_optimization()
