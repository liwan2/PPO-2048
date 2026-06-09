import numpy as np
from agent import Agent
from config import ELITE_BLOCK_SIZE

agent = Agent()
# set training stage so elite pools are sampled (weights>0 for stage 1)
agent.training_stage = 1
# create a fake episode of 20 transitions
fake_episode = []
for i in range(20):
    state = np.zeros((4,4), dtype=np.int32) + i
    action = i % 4
    reward = float(i)
    next_state = state + 1
    done = (i == 19)
    fake_episode.append((state, action, reward, next_state, done))

# push into episode elite pool stage 1 (index 0)
agent.episode_elite_pools[0].append(fake_episode)
print('Episode pool size:', len(agent.episode_elite_pools[0]))
# create a fake per-transition elite memory entry so weight>0
for t in fake_episode:
    agent.elite_memories[0].push(*t)

# test sampling
state, action, reward, next_state, done, pool_ids, idxs, is_weight = agent._sample_stage_mixed_batch(16, beta=0.6)
print('sample shapes:', state.shape, action.shape)
print('pool ids unique:', np.unique(pool_ids, return_counts=True))
print('is_weight stats:', is_weight.min(), is_weight.mean(), is_weight.max())

# show a few sampled states first entries
print('first states[0]:\n', state[0])
print('done counts:', sum(done))
