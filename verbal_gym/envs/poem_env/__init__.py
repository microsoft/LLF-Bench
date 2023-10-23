import numpy as np
import gym
from gym.envs.registration import register
from verbal_gym.utils.benchmark_utils import generate_combinations_dict
from verbal_gym.envs.env_wrapper import VerbalGymWrapper, TerminalFreeWrapper


ENVIRONMENTS = (
    'Haiku',
    'Tanka',
    'LineSyllableConstrainedPoem',
    'SyllableConstrainedPoem',
)
INSTRUCTION_TYPES = ('b') #, 'p', 'c')
FEEDBACK_TYPES = ('m', 'n', 'r', 'hn', 'fp')



class PoemGymWrapper(gym.Wrapper):

    def __init__(self, env, instruction_type, feedback_type):
        super().__init__(env)
        self.instruction_type = instruction_type
        self.feedback_type = feedback_type
        assert self.instruction_type in INSTRUCTION_TYPES
        assert self.feedback_type in FEEDBACK_TYPES
        self._feedback_types = list(FEEDBACK_TYPES)
        self._feedback_types.remove('m')

        self._feedback_type_table = {'r':0, 'hn':0.5, 'fp':1}
        self.feedback = self._feedback_type_table[self.feedback_type]

    # TODO paraphrase assignment
    def reset(self):
        instruction = self.env.reset()
        # TODO types of instructions
        return dict(instruction=instruction, observation=None, feedback=None)

    # TODO paraphrase feedback
    def step(self, action):
        feedback_type = np.random.choice(self._feedback_types) if self.feedback_type=='m' else self.feedback_type
        self.feedback = self._feedback_type_table[feedback_type]
        observation, reward, terminal, info = self.env.step(action)
        observation = dict(instruction=None, observation=None, feedback=info['feedback'])
        del info['feedback']
        return observation, reward, terminal, info


def make_poem_env(env_name,
                  instruction_type='b',
                  feedback_type='r',
                  silent=True,
                  use_extractor=False
                  ):
    """ Make the original env and wrap it with the VerbalGymWrapper. """
    import importlib
    PoemCls = getattr(importlib.import_module("verbal_gym.envs.poem_env.formal_poems"), env_name)
    env = PoemCls(feedback=0, silent=True, use_extractor=False)  # `feedback` doesn't matter here, as we will override it.
    env = PoemGymWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type)
    return VerbalGymWrapper(TerminalFreeWrapper(env))


configs = generate_combinations_dict(
                dict(env_name=ENVIRONMENTS,
                     feedback_type=FEEDBACK_TYPES,
                     instruction_type=INSTRUCTION_TYPES))

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