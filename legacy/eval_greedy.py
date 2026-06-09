from env import Game2048Env
from agent import Agent
import numpy as np
from collections import Counter

def evaluate(episodes=100):
    env = Game2048Env()
    agent = Agent()
    agent.load_model()
    scores = []
    max_tiles = []
    invalid_moves = 0
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, valid = env.step(action)
            if not valid:
                invalid_moves += 1
            state = next_state
        scores.append(env.score)
        max_tiles.append(int(env.board.max()))
    print(f"Evaluated {episodes} episodes (greedy):")
    print(f"Score mean: {np.mean(scores):.2f}, std: {np.std(scores):.2f}, max: {np.max(scores):.2f}")
    print(f"Max tile counts: {dict(Counter(max_tiles))}")
    print(f"Invalid move ratio: {invalid_moves/(episodes):.4f} (avg invalids per episode)")

if __name__ == '__main__':
    evaluate(episodes=100)
