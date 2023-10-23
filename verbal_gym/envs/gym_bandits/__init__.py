import numpy as np
import gym
from gym.envs.registration import register
from verbal_gym.envs.env_wrapper import VerbalGymWrapper, TerminalFreeWrapper, RandomActionOrderWrapper
from verbal_gym.utils.benchmark_utils import generate_combinations_dict
from verbal_gym.envs.utils import format
import gym_bandits
from gym.utils import seeding
from .prompts import *


ENVIRONMENTS = (
    'BanditTenArmedRandomFixed-v0',
    'BanditTenArmedRandomRandom-v0',
    'BanditTenArmedGaussian-v0',
    'BanditTenArmedUniformDistributedReward-v0',
    'BanditTwoArmedDeterministicFixed-v0',
    'BanditTwoArmedHighHighFixed-v0',
    'BanditTwoArmedHighLowFixed-v0',
    'BanditTwoArmedLowLowFixed-v0',
)

INSTRUCTION_TYPES = ('b', 'p', 'c')
FEEDBACK_TYPES = ('m', 'n', 'r', 'hp', 'hn', 'fp', 'fn')

class BanditGymWrapper(gym.Wrapper):

    def __init__(self, env, instruction_type, feedback_type):
        env = RandomActionOrderWrapper(env)
        super().__init__(env)
        self.instruction_type = instruction_type
        self.feedback_type = feedback_type
        assert self.instruction_type in INSTRUCTION_TYPES
        assert self.feedback_type in FEEDBACK_TYPES
        self._feedback_types = list(FEEDBACK_TYPES)
        self._feedback_types.remove('m')

    def reset(self):
        self._bandit_env.__init__()  # gym_bandits implement the reset at the init for some reason.
        self.env.reset()  # bandit env has no observation
        docstring =  self._bandit_env.__doc__
        n_actions = self.env.action_space.n
        instruction = docstring +'\n' + format(b_instruction, low=0, high=n_actions-1)
        if self.instruction_type=='p':  # Give info of a bad action.
            bad_action = np.random.choice(np.delete(np.arange(self.env.action_space.n), self._best_arm))
            instruction += '\n'+format(p_instruction, bad_action=bad_action, reward=self._expected_reward(bad_action))
        if self.instruction_type=='c':
            instruction += '\n'+format(c_instruction, best_arm=self._best_arm)
        return dict(instruction=instruction, observation=None, feedback=None)

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)

        feedback = format(r_feedback, reward=reward)  # reward feedback
        feedback_type = np.random.choice(self._feedback_types) if self.feedback_type=='m' else self.feedback_type
        if feedback_type == 'hp':  # hindsight positive: explaination on why something is correct
            if action == self._best_arm:
                feedback += " "+format(hp_feedback)
        elif feedback_type == 'hn':  # hindsight negative: explaination on why something is incorrect
            if action != self._best_arm:
                feedback += " "+format(hn_feedback)
        elif feedback_type == 'fp':  # future positive: suggestion of things to do
            feedback += " "+format(hp_feedback, best_arm=self._best_arm, reward=self._expected_reward(self._best_arm))
        elif feedback_type == 'fn':  # future negative: suggestion of things to avoid
            bad_action = np.random.choice(np.delete(np.arange(self.env.action_space.n), self._best_arm))
            feedback += " "+format(fn_feedback, bad_action=bad_action, reward=self._expected_reward(bad_action))
        elif feedback_type == 'n':
            feedback = None
        observation = dict(instruction=None, observation=None, feedback=feedback)
        return observation, reward, terminal, info

    @property
    def _bandit_env(self): # This is hardcoded for gym_bandits
        return self.env.env.env.env.env  # this is the raw env

    def seed(self, seed=None):  # This to fix the seeding issue for gym_bandits
        self._bandit_env._seed(seed)

    @property
    def __reward_fun(self):
        # NOTE: This is based on the internal action space before
        # RandomActionOrderWrapper. Use it with caution.
        if not isinstance(self.r_dist[0], list):
            r_dist = self.env.r_dist
        else:
            r_dist = np.array([mean for mean, scale in self.env.r_dist])
        return self.env.p_dist * r_dist

    def _expected_reward(self, idx):  # external action space
        idx = self.internal_action(idx)
        return self.__reward_fun[idx]

    @property
    def _best_arm(self):  # external action space
        idx = self.__reward_fun.argmax()
        return self.external_action(idx)


def make_verbal_env(env_name,
                    instruction_type='b',
                    feedback_type='r',
                    ):
    """ Make the original env and wrap it with the VerbalGymWrapper. """
    env = gym.make(env_name)  # env_name is the original env name of gym_bandits
    env = BanditGymWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type)
    return VerbalGymWrapper(TerminalFreeWrapper(env))


configs = generate_combinations_dict(
                dict(env_name=ENVIRONMENTS,
                     feedback_type=FEEDBACK_TYPES,
                     instruction_type=INSTRUCTION_TYPES))

for config in configs:
    env_name, version = config['env_name'].split('-v')
    register(
        id=f"verbal-{env_name}-{config['instruction_type']}-{config['feedback_type']}-v{version}",
        entry_point='verbal_gym.envs.gym_bandits:make_verbal_env',
        kwargs=config,
    )

for env_name in ENVIRONMENTS:
    # default version (backwards compatibility)
    register(
        id=f"verbal-{env_name}",
        entry_point='verbal_gym.envs.gym_bandits:make_verbal_env',
        kwargs=dict(env_name=env_name, feedback_type='r', instruction_type='b')
    )