from gymnasium.envs.registration import register

from verbal_gym.envs.alfworld.alfworld import Alfworld
from verbal_gym.envs.alfworld.wrapper import AlfworldWrapper

# TODO download PPDL and other data necessary to run Alfworld

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


for env_name in ENVIRONMENTS:
    # default version (backwards compatibility)
    register(
        id=f"verbal-{env_name}",
        entry_point='verbal_gym.envs.alfworld:make_env',
        kwargs=dict(env_name=env_name, feedback_type='a', instruction_type='b')
    )
