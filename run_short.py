"""快速验证训练脚本 (50 updates)。"""
import config

config.PPO_NUM_UPDATES = 50
config.EVAL_EVERY_UPDATES = 10
config.EVAL_EPISODES = 10
config.EARLY_STOP_PATIENCE = 9999
config.SAVE_EVERY_UPDATES = 9999

from train import train

if __name__ == "__main__":
    train()
