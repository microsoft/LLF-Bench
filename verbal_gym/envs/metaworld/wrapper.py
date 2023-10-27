from typing import Dict, Union
import numpy as np
from verbal_gym.envs.verbal_gym_env import VerbalGymWrapper
from verbal_gym.envs.metaworld.prompts import *
import metaworld
import importlib
import json
from textwrap import dedent, indent
from metaworld.policies.policy import move

DIGIT=3
class MetaworldWrapper(VerbalGymWrapper):

    """ This is wrapper for gym_bandits. """

    INSTRUCTION_TYPES = ('b') #('b', 'p', 'c')
    FEEDBACK_TYPES = ('r', 'fp', 'm') #, 'n', 'r', 'hp', 'hn', 'fp', 'fn')

    def __init__(self, env, instruction_type, feedback_type):
        super().__init__(env, instruction_type, feedback_type)
        # load the scripted policy
        module = importlib.import_module(f"metaworld.policies.sawyer_{self.env.env_name.replace('-','_')}_policy")
        self._policy = getattr(module, f"Sawyer{self.env.env_name.title().replace('-','')}Policy")()

    @property
    def mw_env(self):
        return self.env.env
    @property
    def mw_policy(self):
        return self._policy

    def _step(self, action):
        observation, reward, terminal, info = self.env.step(action)

        feedback_type = self.feedback_type

        if feedback_type=='r':
            feedback = self.format(r_feedback, reward=reward)
        elif feedback_type=='fp':
            expert_action = self.mw_policy.get_action(observation)
            feedback = self.format(fp_feedback, expert_action=self.parse_action(expert_action))
        elif feedback_type=='n':
            feedback=None
        else:
            raise NotImplementedError
        observation = self.parase_obs(observation)
        return dict(instruction=None, observation=observation, feedback=feedback), reward, terminal, info

    def parse_action(self, action):
        """ Parse action into text. """
        # return f"delta x: {action[0]:.2f}, delta y:{action[1]:.2f}, delta z:{action[2]:.2f}, gripper state:{action[3]:.1f}"
        return np.array2string(action, precision=2)

    def parase_obs(self, observation):
        """ Parse np.ndarray observation into text. """
        obs_dict = self._policy._parse_obs(observation)
        # remove unused parts
        unused_keys = [k for k in obs_dict.keys() if 'unused' in k]
        for k in unused_keys:
            del obs_dict[k]
        # convert np.ndarray to list
        for k,v in obs_dict.items():
            if isinstance(v, np.ndarray):
                obs_dict[k] = np.array2string(v, precision=3)
        observation_text = json.dumps(obs_dict)
        return observation_text

    def _reset(self):
        observation = self.parase_obs(self.env.reset())
        task = self.env.env_name.split('-')[0]
        instruction = self.format(mw_instruction, task=task)
        return dict(instruction=instruction, observation=observation, feedback=None)
