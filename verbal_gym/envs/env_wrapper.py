import gym
import copy
import numpy as np
from typing import Dict, Any, Tuple

class VerbalGymWrapper(gym.Wrapper):
    """
        This is the wrapper that turns a gym environment into a verbal gym
        environment.

        In verbal-gym, the environment's reward is not provided to the agent.
        Instead the agent learns from info of instructions, observations, and
        their feedback.

        We present this info to the agent via the an observation dict, which has
        keys: 'instruction', 'observation', 'feedback'. The 'instruction' is a
        string containing the task instruction and optionally examples or other
        prior information that might help explain the task. The 'observation' is
        a (partial) observation of the environment state. The 'feedback' is a
        string containing formative feedback for learning (which is a
        replacement for reward in RL). If any attribute is missing, it is
        represented as None. But at the beginning `instruction` must not be None
        and 'feedback' must be None.

        This wrapper is backward compatible with text-based gym environments,
        which returns a string as observation. In this case, the initial
        observation is treated as the instruction, and the reward is textualized
        and treated as the feedback.

    """

    def format_check(self, observation: Dict[str, Any]):
        assert isinstance(observation, dict), "The observation must be a dict."
        assert 'observation' in observation and 'feedback' in observation and 'instruction' in observation, "The observation must be a dict with keys: observation, feedback, instruction."

    def reset(self) -> Dict[str, str]:
        observation = self.env.reset()
        if type(observation)==str:  # backward compatibility
            observation = dict(instruction=observation, observation=None, feedback=None)
        self.format_check(observation)
        assert observation['feedback'] is None, "The feedback must be None in the initial observation"
        assert observation['instruction'] is not None, "The instruction must be provided in the initial observation"
        return observation

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        observation, reward, terminal, info = self.env.step(action)
        if type(observation)==str:  # backward compatibility
            observation = dict(instruction=None, observation=observation, feedback=f"You received a reward of {reward}.")
        self.format_check(observation)
        return observation, reward, terminal, info


class TerminalFreeWrapper(gym.Wrapper):
    #  Set terminal to False always
    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        return observation, reward, False, info

class RandomActionOrderWrapper(gym.Wrapper):

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self.__action_table = None

    def reset(self):
        self.__action_table = [i for i in range(self.env.action_space.n)]
        np.random.shuffle(self.__action_table)
        return self.env.reset()

    def step(self, action):
        action = self.internal_action(action)
        observation, reward, terminal, info = self.env.step(action)
        return observation, reward, terminal, info

    def internal_action(self, action):
        # map action from the external action space to the internal action space
        return self.__action_table[action]

    def external_action(self, action):
        # map action from the internal action space to the external action space
        return self.__action_table.index(action)


class FullInformationWrapper(gym.Wrapper):
    """
        This wrapper assumes env supports pickle serialization.
    """
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)

    def oracle_info(self):
        full_information = dict()
        for i in range(self.env.action_space.n):
            env = copy.deepcopy(self.env)
            observation, reward, terminal, info = env.step(i)
            full_information[i] = dict(
                observation=observation,
                reward=reward,
                terminal=terminal,
                feedback=info['feedback'],
            )
        return full_information

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        info['oracle_info'] = self.oracle_info()
        return observation, reward, terminal, info