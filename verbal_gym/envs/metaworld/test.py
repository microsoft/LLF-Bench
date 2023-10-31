import verbal_gym
import gym
import metaworld

import numpy as np

print(metaworld.ML1.ENV_NAMES)


env = gym.make('verbal-hand-insert-b-fp-v2')
obs = env.reset()
print(obs)
action = np.array([0,0,0,0]) # env.action_space.sample()


for _ in range(1000):
    action = env.mw_policy.get_action(env.current_observation)
    obs, reward, done, info = env.step(action)
    for k, v in obs.items():
        print(k, v)
        print()
    breakpoint()
    if done:
        break