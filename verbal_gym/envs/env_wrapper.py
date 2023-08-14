import gym

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
