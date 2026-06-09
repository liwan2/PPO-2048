import time
import numpy as np
from env import Game2048Env
from agent import Agent
from config import *

EPISODES = 5

env = Game2048Env()
agent = Agent()

print("Device:", agent.device)

# warm up
state = env.reset()
for _ in range(5):
    a = agent.select_action(state, evaluate=False)
    ns, r, d, v = env.step(a)
    agent.store_transition(state, a, r, ns, d)
    _ = agent.optimize_model()
    state = ns

# benchmark
start = time.time()
total_steps = 0
opt_calls = 0
opt_time = 0.0
step_time = 0.0

for ep in range(EPISODES):
    state = env.reset()
    done = False
    steps = 0
    while not done:
        t0 = time.time()
        a = agent.select_action(state, evaluate=False)
        ns, r, d, v = env.step(a)
        t1 = time.time()
        step_time += (t1 - t0)

        agent.store_transition(state, a, r, ns, d)
        t2 = time.time()
        _ = agent.optimize_model()
        t3 = time.time()
        opt_time += (t3 - t2)
        opt_calls += 1

        state = ns
        steps += 1
        total_steps += 1
        # safety break
        if steps >= 200:
            break

    print(f"Episode {ep+1} steps={steps}")

end = time.time()
wall = end - start
print(f"Episodes: {EPISODES}")
print(f"Total steps: {total_steps}")
print(f"Wall time: {wall:.2f}s")
print(f"Avg time per episode: {wall/EPISODES:.3f}s")
print(f"Avg steps per episode: {total_steps/EPISODES:.2f}")
print(f"Avg time per step (action+env): {step_time/total_steps:.6f}s")
print(f"Avg time per optimize call: {opt_time/max(1,opt_calls):.6f}s")
print(f"Optimizations per step ratio: {opt_calls/total_steps:.3f}")

# Estimate full training time based on MAX_EPISODES and avg steps
avg_steps = total_steps/EPISODES
est_time_total = (wall/EPISODES) * MAX_EPISODES
est_hours = est_time_total/3600.0
print(f"Estimated full training time for {MAX_EPISODES} episodes: {est_hours:.2f} hours")
