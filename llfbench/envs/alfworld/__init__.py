import os
from gymnasium.envs.registration import register

from llfbench.envs.alfworld.alfworld import Alfworld
from llfbench.envs.alfworld.wrapper import AlfworldWrapper
from llfbench.envs.alfworld.alfworld_download import download_alfworld_data

# TODO download PPDL and other data necessary to run Alfworld
os.environ["ALFWORLD_DATA"] = "alfworld_data"

if not os.path.exists(os.environ["ALFWORLD_DATA"]) or len(os.environ["ALFWORLD_DATA"]) == 0:
    print(f"Downloading Alfworld data to {os.environ['ALFWORLD_DATA']}")
    download_alfworld_data()
else:
    print(f"Alfworld data already exists in {os.environ['ALFWORLD_DATA']} "
          f"If this is old or stale, then delete it and run the code again.")

ENVIRONMENTS = (
    'Alfworld-v0',
)


def make_env(env_name,
             instruction_type='b',
             feedback_type='r',
             ):

    """ Make the original env and wrap it with the LLFWrapper. """
    assert env_name.startswith("Alfworld"), f"Alfworld environment {env_name} must start with Alfworld"
    env = Alfworld(instruction_type=instruction_type, feedback_type=feedback_type)
    # we don't pass arguments here, because _reset in BanditGymWrapper calls __init__ of the env without arguments.
    return AlfworldWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type)


for env_name in ENVIRONMENTS:
    # default version (backwards compatibility)
    register(
        id=f"llf-{env_name}",
        entry_point='llfbench.envs.alfworld:make_env',
        kwargs=dict(env_name=env_name, feedback_type='a', instruction_type='b')
    )
