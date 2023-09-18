import gym
import copy
import numpy as np
from verbal_gym.llm import DEFAULT_LLM

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
        action = self.__action_table[action]
        observation, reward, terminal, info = self.env.step(action)
        return observation, reward, terminal, info


class VerbalGymWrapper(gym.Wrapper):
    """
        This is basic example wrapper that turns a gym environment into a verbal
        gym environment. Each verbal gym environment has a `docstring` attribute
        that describes the problem. In addition, the `step` function should
        return a feedback string, in info['feedback'], which describes the
        verbal feedback.
    """

    def __init__(self, env, docstring,
                 paraphrase=False,
                 paraphrase_prompt="You're an expert in paraphrasing. Please paraphrase the following text: \n\n{}\n\n.",
                 temperature=0.0):
        """ The user should provide a text description of the problem. """
        super().__init__(env)
        if not hasattr(env, 'docstring'):
            self.docstring = docstring

        # these are verbal gym specific
        self._vg_docstring = docstring
        self._vg_paraphrase = paraphrase
        self._vg_paraphrase_prompt = paraphrase_prompt
        self._vg_temperature = temperature

    def reset(self):
        observation = self.env.reset()
        if self._vg_paraphrase:
            prompt = self._vg_paraphrase_prompt.format(self.docstring)
            self.docstring = DEFAULT_LLM.generate(prompt=prompt, temperature=self._vg_temperature)[0]
        return observation

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