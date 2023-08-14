import gym, verbal_gym

env = gym.make('verbal-BanditTenArmedRandomRandom-v0')

print('Problem: {}'.format(env.docstring))
for _ in range(5):
    env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    print('Feedback: {}'.format(info['feedback']))