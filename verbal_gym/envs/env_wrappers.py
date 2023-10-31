import gym
import copy
import numpy as np
import random
import traceback

class TextWrapper(gym.Wrapper):
    # This is a wrapper that can be applied on top of VerbalGymWrapper to turn into a text-based env.
    RMIN = 0.0  # TODO maybe get this from the env

    def _parse_action(self, action):
        # parse action from string to internal action space
        # TODO
        if self.env.action_space == gym.spaces.Discrete:
            action = int(action)
        elif self.env.action_space == gym.spaces.Box:
            exec("action = np.array({})".format(action))
        elif self.env.action_space == gym.spaces.Text:
            pass
        else:
            raise NotImplementedError
        return action

    def _parse_observation(self, observation):
        # Maybe parse the observation dict to string?
        # TODO
        return observation

    def step(self, action):
        assert type(action) == str
        try:
            action = self._parse_action(action)
            observation, reward, done, info =  self.env.step(action)
        except Exception as e:
            if e == NotImplementedError:
                raise NotImplementedError
            feedback = f"Cannot parse action {action}.\n{traceback.format_exc()}"
            observation = dict(instruction=None, observation=None, feedback=feedback)
            reward = self.RMIN
            done = False
            info = {}
        return self._parse_observation(observation), reward, done, info


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