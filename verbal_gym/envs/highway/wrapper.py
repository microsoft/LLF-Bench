import numpy as np
import json
from verbal_gym.envs.verbal_gym_env import VerbalGymWrapper, Feedback
from verbal_gym.envs.highway.prompts import *


class HighwayWrapper(VerbalGymWrapper):

    """ This is a wrapper for highway-env. """

    INSTRUCTION_TYPES = ('b', 'p')
    FEEDBACK_TYPES = ('n', 'r', 'hp', 'hn', 'm')

    def __init__(self, env, instruction_type, feedback_type):
        super().__init__(env, instruction_type, feedback_type)

    def _reset(self, seed=None, options=None):
        options = options or {}
        observation, info = self.env.reset(seed=seed, options=options)
        text_observation = self.textualize_observation(observation)
        instruction = highway_instruction[0] +'\n' + b_instruction[0]
        return dict(instruction=instruction, observation=text_observation, feedback=None), info

    def _step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        feedback = Feedback()
        feedback_type = self._feedback_type
        if 'r' in feedback_type: # reward feedback
            feedback.r = self.format(r_feedback, reward=reward) # base reward feedback
        if 'hp' in feedback_type: # hindsight positive: explanation on why something is correct
            if reward >= -self.env.config["success_goal_reward"]:
                feedback.hp = self.format(hp_feedback)
        if 'hn' in feedback_type:  # hindsight negative: explanation on why something is incorrect
            crashed = any(vehicle.crashed for vehicle in self.env.controlled_vehicles)
            if crashed:
                feedback.hn = self.format(hn_feedback)
        text_observation = self.textualize_observation(observation)
        return_observation = dict(instruction=None, observation=text_observation, feedback=feedback)
        return return_observation, reward, terminated, truncated, info
    
    def textualize_observation(self, observation):
        text_observation = 'Desired goal:' + json.dumps(np.array2string(observation['desired_goal'], precision=3))
        text_observation += '\nAchieved goal:' + json.dumps(np.array2string(observation['achieved_goal'], precision=3))
        text_observation += '\nObservation:' + json.dumps(np.array2string(observation['observation'], precision=3))
        return text_observation