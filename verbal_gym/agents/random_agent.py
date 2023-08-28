import random

from agents.abstract_agent import Agent


class RandomAgent(Agent):
    """ An random agent for a fixed number of actions. """

    NAME = "Random"

    def __init__(self, n_actions):
        super(Agent, self).__init__()
        assert type(n_actions)==int
        self.n_actions = n_actions

    def act(self, *args, **kwargs):
        return random.randint(0, self.n_actions-1)
