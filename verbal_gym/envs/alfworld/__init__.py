from gymnasium.envs.registration import register

from verbal_gym.envs.alfworld.alfworld import Alfworld
from verbal_gym.envs.alfworld.wrapper import AlfworldWrapper
from verbal_gym.utils import generate_combinations_dict


ENVIRONMENTS = (
    'Alfworld-v0',
)


def make_env(env_name,
             instruction_type='b',
             feedback_type='r',
             ):

    """ Make the original env and wrap it with the VerbalGymWrapper. """
    assert env_name.startswith("Alfworld"), f"Alfworld environment {env_name} must start with Alfworld"
    env = Alfworld(instruction_type=instruction_type, feedback_type=feedback_type)
    # we don't pass arguments here, because _reset in BanditGymWrapper calls __init__ of the env without arguments.
    return AlfworldWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type)


configs = generate_combinations_dict(
                dict(env_name=ENVIRONMENTS,
                     feedback_type=AlfworldWrapper.FEEDBACK_TYPES,
                     instruction_type=AlfworldWrapper.INSTRUCTION_TYPES))

for config in configs:
    env_name, version = config['env_name'].split('-v')
    register(
        id=f"verbal-{env_name}-{config['instruction_type']}-{config['feedback_type']}-v{version}",
        entry_point='verbal_gym.envs.alfworld:make_env',
        kwargs=config,
    )

for env_name in ENVIRONMENTS:
    # default version (backwards compatibility)
    register(
        id=f"verbal-{env_name}",
        entry_point='verbal_gym.envs.alfworld:make_env',
        kwargs=dict(env_name=env_name, feedback_type='r', instruction_type='b')
    )
