import numpy as np
from gymnasium.envs.registration import register
from verbal_gym.utils import generate_combinations_dict
from verbal_gym.envs.poem.wrapper import PoemGymWrapper

ENVIRONMENTS = (
    'Haiku',
    'Tanka',
    'LineSyllableConstrainedPoem',
    'SyllableConstrainedPoem',
)

def make_env(env_name,
             instruction_type='b',
             feedback_type='r',
             **kwargs
                  ):
    """ Make the original env and wrap it with the VerbalGymWrapper. """
    import importlib
    PoemCls = getattr(importlib.import_module("verbal_gym.envs.poem.formal_poems"), env_name)
    env = PoemCls(**kwargs)  # `feedback` doesn't matter here, as we will override it.
    return PoemGymWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type)


for env_name in ENVIRONMENTS:
    # default version (backwards compatibility)
    register(
        id=f"verbal-poem-{env_name}-v0",
        entry_point='verbal_gym.envs.poem:make_env',
        kwargs=dict(env_name=env_name, feedback_type='a', instruction_type='b')
    )