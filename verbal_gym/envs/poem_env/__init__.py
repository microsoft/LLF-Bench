import numpy as np
import gym
from gym.envs.registration import register
from verbal_gym.utils.benchmark_utils import generate_combinations_dict
from verbal_gym.envs.poem_env.wrapper import PoemGymWrapper

ENVIRONMENTS = (
    'Haiku',
    'Tanka',
    'LineSyllableConstrainedPoem',
    'SyllableConstrainedPoem',
)

def make_poem_env(env_name,
                  instruction_type='b',
                  feedback_type='r',
                  silent=True,
                  use_extractor=False
                  ):
    """ Make the original env and wrap it with the VerbalGymWrapper. """
    import importlib
    PoemCls = getattr(importlib.import_module("verbal_gym.envs.poem_env.formal_poems"), env_name)
    env = PoemCls(feedback=0, silent=silent, use_extractor=use_extractor)  # `feedback` doesn't matter here, as we will override it.
    return PoemGymWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type)


configs = generate_combinations_dict(
                dict(env_name=ENVIRONMENTS,
                     feedback_type=PoemGymWrapper.FEEDBACK_TYPES,
                     instruction_type=PoemGymWrapper.INSTRUCTION_TYPES))

for config in configs:
    config['silent'] = True
    config['use_extractor'] = False
    register(
        id=f"verbal-{config['env_name']}-{config['instruction_type']}-{config['feedback_type']}-v0",
        entry_point='verbal_gym.envs.poem_env:make_poem_env',
        kwargs=config,
    )

for env_name in ENVIRONMENTS:
    # default version (backwards compatibility)
    register(
        id=f"verbal-{env_name}-v0",
        entry_point='verbal_gym.envs.poem_env:make_poem_env',
        kwargs=dict(env_name=env_name, feedback_type='r', instruction_type='b', silent=True, use_extractor=False)
    )