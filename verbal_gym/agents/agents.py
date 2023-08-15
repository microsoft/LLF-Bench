import random


class Agent:

    def __init__(self, *args, **kwargs):
        # TODO: pass env_spec
        pass

    def act(self, *args, **kwargs):
        raise NotImplementedError

class RandomAgent(Agent):
    """ An random agent for a fixed number of actions. """

    def __init__(self, n_actions):
        assert type(n_actions)==int
        self.n_actions = n_actions

    def act(self, *args, **kwargs):
        return random.randint(self.n_actions)