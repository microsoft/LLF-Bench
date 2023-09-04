import pdb
import gym
import verbal_gym


env = gym.make('verbal-PickBestSummary-v0')

print('Problem: {}'.format(env.docstring))
for _ in range(5):
    results = env.reset()
    pdb.set_trace()
    # action = env.action_space.sample()
    action = env.get_oracle_action()
    obs, reward, done, info = env.step(action)
    print('Reward: {}'.format(reward))
    print('Feedback: {}'.format(info['feedback']))
    pdb.set_trace()
