import gym
import copy

class TerminalFreeWrapper(gym.Wrapper):
    #  Set terminal to False always
    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        return observation, reward, False, info

class VerbalGymWrapper(gym.Wrapper):
    """
        This is basic example wrapper that turns a gym environment into a verbal
        gym environment. Each verbal gym environment has a `docstring` attribute
        that describes the problem. In addition, the `step` function should
        return a feedback string, in info['feedback'], which describes the
        verbal feedback.
    """

    def __init__(self, env, docstring):
        """ The user should provide a text description of the problem. """
        super().__init__(env)
        self.docstring = docstring

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        if 'feedback' not in info:
            info['feedback'] = 'You get a reward of {}.'.format(reward)
        return observation, reward, terminal, info


class FullInformationWrapper(gym.Wrapper):
    """
        This wrapper assumes env supports pickle serialization.
    """
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)

    def get_full_information(self):
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
        info['full_information'] = self.get_full_information()
        return observation, reward, terminal, info