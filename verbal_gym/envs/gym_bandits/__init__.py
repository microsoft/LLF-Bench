import gym
from gym.envs.registration import register
import gym_bandits
from verbal_gym.envs.env_wrapper import VerbalGymWrapper

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

def make_verbal_env(env_name, **kwargs):
    """ Make the original env and wrap it with the VerbalGymWrapper. """
    env = gym.make(env_name)
    docstring =  env.env.env.env.__doc__  # This is hardcoded for gym_bandits
    return VerbalGymWrapper(env, docstring)

for environment in environments:
    register(
        id='verbal-{}'.format(environment),
        entry_point='verbal_gym.envs.gym_bandits:make_verbal_env',
        kwargs={'env_name': environment},
    )
    print('Registered , verbal-{}'.format(environment))