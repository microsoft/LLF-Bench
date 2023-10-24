import numpy as np
from verbal_gym.envs.env_wrappers import TerminalFreeWrapper, RandomActionOrderWrapper
from verbal_gym.envs.verbal_gym_env import VerbalGymWrapper
from verbal_gym.envs.gym_bandits.prompts import *


class BanditGymWrapper(VerbalGymWrapper):

    """ This is wrapper for gym_bandits. """

    INSTRUCTION_TYPES = ('b', 'p', 'c')
    FEEDBACK_TYPES = ('m', 'n', 'r', 'hp', 'hn', 'fp', 'fn')

    def __init__(self, env, instruction_type, feedback_type):
        env = TerminalFreeWrapper(RandomActionOrderWrapper(env))
        super().__init__(env, instruction_type, feedback_type)

    def _reset(self):
        self._bandit_env.__init__()  # gym_bandits implement the reset at the init for some reason.
        self.env.reset()  # bandit env has no observation
        docstring =  self._bandit_env.__doc__
        n_actions = self.env.action_space.n
        instruction = docstring +'\n' + self.format(b_instruction, low=0, high=n_actions-1)
        if self.instruction_type=='p':  # Give info of a bad action.
            bad_action = np.random.choice(np.delete(np.arange(self.env.action_space.n), self._best_arm))
            instruction += '\n'+self.format(p_instruction, bad_action=bad_action, reward=self._expected_reward(bad_action))
        if self.instruction_type=='c':
            instruction += '\n'+self.format(c_instruction, best_arm=self._best_arm)
        return dict(instruction=instruction, observation=None, feedback=None)

    def _step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        feedback = self.format(r_feedback, reward=reward)  # base reward feedback
        feedback_type = self._feedback_type
        if feedback_type == 'hp':  # hindsight positive: explaination on why something is correct
            if action == self._best_arm:
                feedback += " "+self.format(hp_feedback)
        elif feedback_type == 'hn':  # hindsight negative: explaination on why something is incorrect
            if action != self._best_arm:
                feedback += " "+self.format(hn_feedback)
        elif feedback_type == 'fp':  # future positive: suggestion of things to do
            feedback += " "+self.format(hp_feedback, best_arm=self._best_arm, reward=self._expected_reward(self._best_arm))
        elif feedback_type == 'fn':  # future negative: suggestion of things to avoid
            bad_action = np.random.choice(np.delete(np.arange(self.env.action_space.n), self._best_arm))
            feedback += " "+self.format(fn_feedback, bad_action=bad_action, reward=self._expected_reward(bad_action))
        elif feedback_type == 'n':
            feedback = None
        observation = dict(instruction=None, observation=None, feedback=feedback)
        return observation, reward, terminal, info

    @property
    def _bandit_env(self): # This is hardcoded for gym_bandits
        return self.env.env.env.env.env.env  # this is the raw env

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
