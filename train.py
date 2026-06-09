from collections import deque
import os

import numpy as np

from agent import Agent
from config import (
    PPO_NUM_UPDATES,
    EVAL_WINDOW,
    EVAL_EVERY_UPDATES,
    EVAL_EPISODES,
    SAVE_EVERY_UPDATES,
    MIN_UPDATES_BEFORE_SAVE,
    EARLY_STOP_PATIENCE,
    EARLY_STOP_MIN_DELTA,
    BEST_METRIC_ALPHA,
    BEST_MODEL_PATH,
    MODEL_PATH,
)


def _save_plot(
    scores,
    best_scores,
    max_tiles,
    eval_updates,
    tile_256_rates,
    tile_512_rates,
    tile_1024_rates,
    path="training_score_plot.png",
):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plot skipped: {e}")
        return

    if not scores:
        return

    x = np.arange(1, len(scores) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    axes[0].plot(x, scores, alpha=0.7, label="Episode Score")
    axes[0].plot(x, best_scores, linewidth=2, label="Best Score")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, max_tiles, color="tab:orange", alpha=0.8, label="Max Tile")
    axes[1].set_ylabel("Max Tile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if eval_updates:
        eval_x = np.asarray(eval_updates)
        axes[2].plot(eval_x, np.asarray(tile_256_rates) * 100.0, marker="o", label="256 Rate")
        axes[2].plot(eval_x, np.asarray(tile_512_rates) * 100.0, marker="o", label="512 Rate")
        axes[2].plot(eval_x, np.asarray(tile_1024_rates) * 100.0, marker="o", label="1024 Rate")
    axes[2].set_xlabel("Update")
    axes[2].set_ylabel("Rate (%)")
    axes[2].set_ylim(0, 100)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Training plot saved to {path}")


def train():
    agent = Agent()
    if os.path.exists(MODEL_PATH):
        agent.load_model(MODEL_PATH)

    print(f"Standard PPO | {agent.n_envs} envs × {agent.n_steps} steps | "
          f"{PPO_NUM_UPDATES} updates (~{PPO_NUM_UPDATES * agent.n_envs * agent.n_steps:,} timesteps)")

    recent_scores = deque(maxlen=EVAL_WINDOW)
    recent_max_tiles = deque(maxlen=EVAL_WINDOW)
    episode_scores = []
    episode_max_tiles = []
    best_score_hist = []
    eval_update_hist = []
    tile_256_rate_hist = []
    tile_512_rate_hist = []
    tile_1024_rate_hist = []

    best_eval_metric = float("-inf")
    best_checkpoint_metric = float("-inf")
    no_improve = 0
    best_score_seen = 0
    best_max_tile_seen = 0

    for update in range(1, PPO_NUM_UPDATES + 1):
        stats, (pi_loss, v_loss, entropy) = agent.collect_and_update()

        for score in stats["ep_returns"]:
            recent_scores.append(score)
            episode_scores.append(score)
            best_score_seen = max(best_score_seen, score)
            best_score_hist.append(best_score_seen)

        for mt in stats["ep_max_tiles"]:
            recent_max_tiles.append(mt)
            episode_max_tiles.append(mt)
            best_max_tile_seen = max(best_max_tile_seen, mt)

        window_score = float(np.mean(recent_scores)) if recent_scores else 0.0
        window_max_tile = float(np.mean(recent_max_tiles)) if recent_max_tiles else 0.0
        n_eps = len(stats["ep_returns"])

        if update % EVAL_EVERY_UPDATES == 0 or update == 1:
            eval_stats = agent.evaluate(n_episodes=EVAL_EPISODES)
            eval_update_hist.append(update)
            tile_256_rate_hist.append(eval_stats["tile_256_rate"])
            tile_512_rate_hist.append(eval_stats["tile_512_rate"])
            tile_1024_rate_hist.append(eval_stats["tile_1024_rate"])
            metric = (
                5000.0 * eval_stats["tile_1024_rate"]
                + 1200.0 * eval_stats["tile_512_rate"]
                + 100.0 * eval_stats["tile_2048_rate"]
                + eval_stats["mean_score"]
                + 0.1 * eval_stats["mean_max_tile"]
            )

            print(
                f"Update {update}/{PPO_NUM_UPDATES} | "
                f"RolloutEps: {n_eps} | "
                f"WindowScore: {window_score:.1f} | "
                f"WindowMaxTile: {window_max_tile:.0f} | "
                f"EvalScore: {eval_stats['mean_score']:.1f} | "
                f"EvalMaxTile: {eval_stats['mean_max_tile']:.0f} | "
                f"BestTile: {eval_stats['best_max_tile']} | "
                f"WinRate: {eval_stats['win_rate']*100:.1f}% | "
                f"512Rate: {eval_stats['tile_512_rate']*100:.1f}% | "
                f"1024Rate: {eval_stats['tile_1024_rate']*100:.1f}% | "
                f"2048Rate: {eval_stats['tile_2048_rate']*100:.1f}% | "
                f"PiLoss: {pi_loss:.4f} | VLoss: {v_loss:.4f} | Ent: {entropy:.4f}"
            )

            if metric > best_eval_metric + EARLY_STOP_MIN_DELTA:
                best_eval_metric = metric
                if update >= MIN_UPDATES_BEFORE_SAVE:
                    agent.save_model(BEST_MODEL_PATH)
                    best_checkpoint_metric = metric
                    no_improve = 0
            else:
                if update >= MIN_UPDATES_BEFORE_SAVE and best_checkpoint_metric > float("-inf"):
                    no_improve += 1

            if no_improve >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stop at update {update}, best saved metric = {best_checkpoint_metric:.1f}, "
                    f"best eval metric = {best_eval_metric:.1f}"
                )
                break
        elif update % 5 == 0:
            print(
                f"Update {update}/{PPO_NUM_UPDATES} | "
                f"RolloutEps: {n_eps} | WindowScore: {window_score:.1f} | "
                f"BestScoreHist: {best_score_seen:.0f} | BestTileHist: {best_max_tile_seen} | "
                f"PiLoss: {pi_loss:.4f} | VLoss: {v_loss:.4f}"
            )

        if update % SAVE_EVERY_UPDATES == 0:
            agent.save_model(MODEL_PATH)

    agent.save_model(MODEL_PATH)
    if os.path.exists(BEST_MODEL_PATH):
        agent.load_model(BEST_MODEL_PATH)
        print(f"Loaded best model from {BEST_MODEL_PATH}")

    final = agent.evaluate(n_episodes=100)
    print(
        f"\nFinal eval (100 eps): score={final['mean_score']:.1f}, "
        f"max_tile={final['mean_max_tile']:.0f}, best_tile={final['best_max_tile']}, "
        f"win_rate={final['win_rate']*100:.1f}%"
    )

    _save_plot(
        episode_scores,
        best_score_hist,
        episode_max_tiles,
        eval_update_hist,
        tile_256_rate_hist,
        tile_512_rate_hist,
        tile_1024_rate_hist,
    )


if __name__ == "__main__":
    train()
