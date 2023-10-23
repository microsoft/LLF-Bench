import gym
from gym.envs.registration import register
import gym_bandits
from verbal_gym.envs.env_wrapper import VerbalGymWrapper, TerminalFreeWrapper, RandomActionOrderWrapper

environments = [
    'BanditTenArmedRandomFixed-v0',
    'BanditTenArmedRandomRandom-v0',
    'BanditTenArmedGaussian-v0',
    'BanditTenArmedUniformDistributedReward-v0',
    'BanditTwoArmedDeterministicFixed-v0',
    'BanditTwoArmedHighHighFixed-v0',
    'BanditTwoArmedHighLowFixed-v0',
    'BanditTwoArmedLowLowFixed-v0',
]

class BanditGymWrapper(gym.Wrapper):
    def reset(self):
        observation = self.env.reset()
        docstring =  self.env.env.env.env.__doc__  # This is hardcoded for gym_bandits
        n_actions = self.env.action_space.n
        instruction = docstring+f"\nYour action is an integer between 0 and {n_actions-1}."
        return dict(instruction=instruction, observation=None, feedback=None)

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        observation = dict(instruction=None, observation=None, feedback=f"You received a reward of {reward}.")
        return observation, reward, terminal, info

def make_verbal_env(env_name, **kwargs):
    """ Make the original env and wrap it with the VerbalGymWrapper. """
    env = gym.make(env_name)
    env = BanditGymWrapper(env)
    return VerbalGymWrapper(TerminalFreeWrapper(RandomActionOrderWrapper(env)))

for environment in environments:
    register(
        id='verbal-{}'.format(environment),
        entry_point='verbal_gym.envs.gym_bandits:make_verbal_env',
        kwargs={'env_name': environment},
    )
