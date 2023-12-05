import llfbench
import gymnasium as gym
import metaworld

import numpy as np

print(metaworld.MT1.ENV_NAMES)

horizon = 1000
env = gym.make('llf-metaworld-hand-insert',
             instruction_type='b',
             feedback_type=['fp', 'r']) # can also be a single string, like 'hn'

# This tests the wrapper.
obs, _ = env.reset()
for k, v in obs.items():
        print(k, v)
        print()
for i in range(horizon):
    action = env.expert_action
    print(action)
    obs, reward, done, timeout, info = env.step(action)
    print(env.textualize_observation(obs))
    for k, v in obs.items():
        print(k, v)
        print()
    print('success', info['success'])
    if info['success']:
        print(f"Succeeded after {i+1} steps.")
        break
    if done:
        break

# This tests the scripted policy in metaworld.
mw_env = env.mw_env
mw_policy = env.mw_policy
obs, _  = mw_env.reset()
for i in range(horizon):
    action = mw_policy.get_action(obs)
    obs, reward, done, timeout, info = mw_env.step(action)
    print('success', info['success'])
    if info['success']:
        print(f"Succeeded after {i+1} steps.")
        break
    if done or timeout:
        break