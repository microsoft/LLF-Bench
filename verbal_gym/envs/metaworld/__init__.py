import gym
from gym.envs.registration import register
from verbal_gym.utils.benchmark_utils import generate_combinations_dict
from verbal_gym.envs.metaworld.wrapper import MetaworldWrapper
from collections import defaultdict
import importlib
import metaworld
import random
import time

st =time.time()
BENCHMARK = metaworld.ML1
ENVIRONMENTS = tuple(metaworld.ML1.ENV_NAMES)

def make_env(env_name,
             instruction_type='b',
             feedback_type='r',
             ):
    """ Make the original env and wrap it with the VerbalGymWrapper. """
    benchmark = BENCHMARK(env_name)
    env = benchmark.train_classes[env_name]()
    class Wrapper(gym.Wrapper):
         # a small wrapper to make sure the task is set
        def __init__(self, env):
            super().__init__(env)
            # XXX this is for the original gym api
            self.action_space = gym.spaces.Box(low=env.action_space.low, high=env.action_space.high)
            self.observation_space = gym.spaces.Box(low=env.observation_space.low, high=env.observation_space.high)
        @property
        def env_name(self):
            return env_name
        def step(self, action):  # XXX this is for the original gym api
            observation, reward, truncate, done, info = self.env.step(action)
            return observation, reward, done or truncate, info
        def reset(self):
            task = random.choice(benchmark.train_tasks)
            self.env.set_task(task)
            return self.env.reset()[0]
    env = Wrapper(env)
    return MetaworldWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type)


configs = generate_combinations_dict(
                dict(env_name=ENVIRONMENTS,
                     feedback_type=MetaworldWrapper.FEEDBACK_TYPES,
                     instruction_type=MetaworldWrapper.INSTRUCTION_TYPES))

for config in configs:
    env_name, version = config['env_name'].split('-v')
    register(
        id=f"verbal-{env_name}-{config['instruction_type']}-{config['feedback_type']}-v{version}",
        entry_point='verbal_gym.envs.metaworld:make_env',
        kwargs=config,
    )

for env_name in ENVIRONMENTS:
    # default version (backward compatibility)
    register(
        id=f"verbal-{env_name}",
        entry_point='verbal_gym.envs.metaworld:make_env',
        kwargs=dict(env_name=env_name, feedback_type='r', instruction_type='b')
    )