import random
from typing import Union, Dict, Any
from verbal_gym.agents.abstract_agent import Agent


class RandomAgent(Agent):
    """ An random agent for a fixed number of actions. """

    NAME = "Random"

    def __init__(self, n_actions: int):
        super(Agent, self).__init__()
        assert type(n_actions)==int
        self.n_actions = n_actions

    def act(self, observation: Union[str, Dict[str, Any]]) -> Any:
        return random.randint(0, self.n_actions-1)
