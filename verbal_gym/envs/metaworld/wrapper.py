from typing import Dict, Union
import numpy as np
from verbal_gym.envs.verbal_gym_env import VerbalGymWrapper
from verbal_gym.envs.metaworld.prompts import *
from verbal_gym.envs.metaworld.gains import P_GAINS
import metaworld
import importlib
import json
from textwrap import dedent, indent
from metaworld.policies.policy import move
from metaworld.policies.action import Action
from metaworld.policies import SawyerDrawerOpenV1Policy, SawyerDrawerOpenV2Policy # TODO

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
        self._time_out = 20 # for convergnece of P controller
        self._threshold = 1e-4 # for convergnece of P controller
        self._current_observation = None

    @property
    def mw_env(self):
        return self.env.env

    @property
    def mw_policy(self):
        return self._policy

    @property
    def current_observation(self):  # external interface
        """ This is a cache of the latest (raw) observation. """
        return self._current_observation

    @property
    def _current_pos(self):
        """ Curret position of the hand. """
        return self.mw_policy._parse_obs(self.current_observation)['hand_pos']

    def p_control(self, action):
        """ Compute the desired control based on a position target (action[:3])
        using P controller provided in Metaworld."""
        assert len(action)==4
        p_gain = P_GAINS[type(self.mw_policy)]
        if type(self.mw_policy) in [type(SawyerDrawerOpenV1Policy), type(SawyerDrawerOpenV2Policy)]:
            # This needs special cares. It's implemented differently.
            o_d = self.mw_policy._parse_obs(self.current_observation)
            pos_curr = o_d["hand_pos"]
            pos_drwr = o_d["drwr_pos"]
            # align end effector's Z axis with drawer handle's Z axis
            if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.06:
                p_gain = 4.0
            # drop down to touch drawer handle
            elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
                p_gain = 4.0
            # push toward a point just behind the drawer handle
            # also increase p value to apply more force
            else:
                p_gain= 50.0

        control = Action({"delta_pos": np.arange(3), "grab_effort": 3})
        control["delta_pos"] = move(self._current_pos, to_xyz=action[:3], p=p_gain)
        control["grab_effort"] = action[3]
        return control.array

    def _step(self, action):
        # Run P controller until convergence or timeout
        # action is viewed as the desired position + grab_effort
        previous_observation = self.current_observation
        for _ in range(self._time_out):
            control = self.p_control(action)
            observation, reward, terminal, info = self.env.step(control)
            self._current_observation = observation
            desired_pos = action[:3]
            if np.abs(desired_pos - self._current_pos).max() < self._threshold:
                break
            # print(np.linalg.norm(desired_pos - self._current_pos), reward)

        feedback_type = self.feedback_type
        if feedback_type=='r':
            feedback = self.format(r_feedback, reward=reward)
        elif feedback_type=='hp':
            raise NotImplementedError
        elif feedback_type=='hn':
            raise NotImplementedError
        elif feedback_type=='fp':
            expert_action = self.mw_policy.get_action(observation)
            feedback = self.format(fp_feedback, expert_action=self.textualize_expert_action(expert_action))
        elif feedback_type=='fn':
            raise NotImplementedError
        elif feedback_type=='n':
            feedback=None
        else:
            raise NotImplementedError
        observation = self.textualize_observation(observation)
        return dict(instruction=None, observation=observation, feedback=feedback), reward, terminal, info

    def _reset(self):
        self._current_observation = self.env.reset()
        observation = self.textualize_observation(self._current_observation)
        task = self.env.env_name.split('-')[0]
        instruction = self.format(mw_instruction, task=task)
        return dict(instruction=instruction, observation=observation, feedback=None)


    def textualize_expert_action(self, action):
        """ Parse action into text. """
        # return f"delta x: {action[0]:.2f}, delta y:{action[1]:.2f}, delta z:{action[2]:.2f}, gripper state:{action[3]:.1f}"
        # TODO should not be the raw action
        return np.array2string(action, precision=2)

    def textualize_observation(self, observation):
        """ Parse np.ndarray observation into text. """
        obs_dict = self.mw_policy._parse_obs(observation)
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