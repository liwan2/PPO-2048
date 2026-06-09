from env import Game2048Env
from agent import Agent
import numpy as np

agent = Agent()
# simulate some random gameplay to fill buffers
env = Game2048Env()
steps = 1000
agent.start_episode()
for i in range(steps):
    s = env.reset() if i % 50 == 0 else s
    action = np.random.randint(0,4)
    ns, r, done, valid = env.step(action)
    agent.store_transition(s, action, r, ns, done)
    s = ns
    if done:
        agent.finalize_episode(int(np.max(env.board)))
        agent.start_episode()

# print buffer stats
print('Main memory entries:', len(agent.memory))
for i, m in enumerate(agent.elite_memories, start=1):
    try:
        tot = m.tree.total()
    except Exception:
        tot = None
    print(f'Elite[{i}] entries:', len(m), ' total:', tot)

# Try to sample a mixed batch
try:
    state, action, reward, next_state, done, pool_ids, idxs, is_weight = agent._sample_stage_mixed_batch(32, beta=0.6)
    print('Sample shapes:', state.shape, action.shape)
    print('Pool id counts:', np.unique(pool_ids, return_counts=True))
    print('is_weight stats: min, mean, max', float(is_weight.min()), float(is_weight.mean()), float(is_weight.max()))
except Exception as e:
    print('Sampling failed:', e)

# Try several optimize steps
for _ in range(5):
    res = agent.optimize_model()
    print('optimize result:', res, 'frame:', agent.frame, 'steps_done:', agent.steps_done)

# Check target sync
print('Training stage', agent.training_stage)
print('Done diag')
