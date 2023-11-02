import numpy as np
from gymnasium.envs.registration import register
from verbal_gym.utils import generate_combinations_dict
from verbal_gym.envs.loss_landscape.wrapper import LossLandscapeGymWrapper

ENVIRONMENTS = (
    'Booth',
    'McCormick',
    'Rosenbrock',
    'SixHumpCamel',
)

def make_env(env_name,
             instruction_type='b',
             feedback_type='r',
             **kwargs):
    """ Make the original env and wrap it with the VerbalGymWrapper. """
    import importlib
    LossCls = getattr(importlib.import_module("verbal_gym.envs.loss_landscape.loss_descent"), env_name)
    env = LossCls(**kwargs)  # `feedback` doesn't matter here, as we will override it.
    return LossLandscapeGymWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type)


configs = generate_combinations_dict(
                dict(env_name=ENVIRONMENTS,
                     feedback_type=LossLandscapeGymWrapper.FEEDBACK_TYPES,
                     instruction_type=LossLandscapeGymWrapper.INSTRUCTION_TYPES))

for config in configs:
    register(
        id=f"verbal-{config['env_name']}-{config['instruction_type']}-{config['feedback_type']}-v0",
        entry_point='verbal_gym.envs.loss_landscape:make_env',
        kwargs=config,
    )

for env_name in ENVIRONMENTS:
    # default version (backwards compatibility)
    register(
        id=f"verbal-{env_name}-v0",
        entry_point='verbal_gym.envs.loss_landscape:make_env',
        kwargs=dict(env_name=env_name, feedback_type='r', instruction_type='b')
    )
