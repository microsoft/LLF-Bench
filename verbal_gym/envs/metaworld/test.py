import verbal_gym
import gymnasium as gym
import metaworld

import numpy as np

print(metaworld.MT1.ENV_NAMES)


env = gym.make('verbal-hand-insert-b-hn-v2')
# env = gym.make('verbal-reach-b-hn-v2')

# This tests the wrapper.
obs = env.reset()
for i in range(1000):
    action = env.expert_action
    obs, reward, done, timeout, info = env.step(action)
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
for i in range(1000):
    action = mw_policy.get_action(obs)
    obs, reward, done, timeout, info = mw_env.step(action)
    print('success', info['success'])
    if info['success']:
        print(f"Succeeded after {i+1} steps.")
        break
    if done or timeout:
        break