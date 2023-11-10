import gymnasium as gym
from gymnasium.envs.registration import register
from verbal_gym.utils import generate_combinations_dict
from verbal_gym.envs.highway.wrapper import HighwayWrapper

ENVIRONMENTS = (
    'parking-v0',
)


def make_env(env_name,
             instruction_type='b',
             feedback_type='a',
             ):
    """ Make the original env and wrap it with the VerbalGymWrapper. """
    env = gym.make(env_name)
    return HighwayWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type)

for env_name in ENVIRONMENTS:
    # default version (backwards compatibility)
    register(
        id=f"verbal-highway-{env_name}",
        entry_point='verbal_gym.envs.highway:make_env',
        kwargs=dict(env_name=env_name, feedback_type='a', instruction_type='b')
    )