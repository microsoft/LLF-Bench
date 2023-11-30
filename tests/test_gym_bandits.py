import llfbench
import gymnasium as gym

env = gym.make('llf-BanditTenArmedRandomRandom-b-fp-v0')

obs = env.reset()
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    for k,v in obs.items():
        print(k, v)
        print()