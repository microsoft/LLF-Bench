import gym
import copy
import numpy as np

# This files contain some helper wrappers for the envs.

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