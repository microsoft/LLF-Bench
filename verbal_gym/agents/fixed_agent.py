from verbal_gym.agents.abstract_agent import Agent


class FixedAgent(Agent):
    """ A fix agent that always takes the same action """

    NAME = "Fixed"

    def __init__(self, fixed_action):
        super(Agent, self).__init__()
        self.fixed_action = fixed_action

    def act(self, *args, **kwargs):
        return self.fixed_action
