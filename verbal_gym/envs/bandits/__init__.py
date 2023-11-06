import gym as old_gym
import gymnasium as gym
from gymnasium.envs.registration import register
from verbal_gym.utils import generate_combinations_dict
from verbal_gym.envs.bandits.wrapper import BanditGymWrapper
import gym_bandits  # this is needed so that gym_bandits is registered


ENVIRONMENTS = (
    'BanditTenArmedRandomFixed-v0',
    'BanditTenArmedRandomRandom-v0',
    'BanditTenArmedGaussian-v0',
    'BanditTenArmedUniformDistributedReward-v0',
    'BanditTwoArmedDeterministicFixed-v0',
    'BanditTwoArmedHighHighFixed-v0',
    'BanditTwoArmedHighLowFixed-v0',
    'BanditTwoArmedLowLowFixed-v0',
)


def make_env(env_name,
             instruction_type='b',
             feedback_type='r',
             ):
    """ Make the original env and wrap it with the VerbalGymWrapper. """
    env = old_gym.make(env_name)  # env_name is the original env name of gym_bandits
    # we don't pass arguments here, because _reset in BanditGymWrapper calls __init__ of the env without arguments.
    return BanditGymWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type)


for env_name in ENVIRONMENTS:
    # default version (backwards compatibility)
    register(
        id=f"verbal-bandits-{env_name}",
        entry_point='verbal_gym.envs.bandits:make_env',
        kwargs=dict(env_name=env_name, feedback_type='r', instruction_type='b')
    )