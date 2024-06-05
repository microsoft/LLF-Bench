from llfbench import envs
import gymnasium as gym

def make(env_name, *, instruction_type='b', feedback_type='a', visual=False):
    env = gym.make(env_name, instruction_type=instruction_type, feedback_type=feedback_type, visual=visual)
    return env

def supported_types(env_name):
    """ Return the supported INSTRUCTION_TYPES and FEEDBACK_TYPES for the given env_name. """
    env = gym.make(env_name)
    return env.INSTRUCTION_TYPES, env.FEEDBACK_TYPES
