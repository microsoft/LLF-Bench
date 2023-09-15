from agents.basic_agent import BasicAgent
from agents.random_agent import RandomAgent
from agents.posterior_agents import PosteriorAgent
from agents.model_based_agents import ModelBasedAgent
from agents.full_information_agent import FullInformationAgent
from agents.tot_agent import ToTAgent
from agents.zero_shot_agent import ZeroshotLLM


class AgentSelector:

    def __init__(self):
        self.agent_classes = [RandomAgent, BasicAgent, ZeroshotLLM, PosteriorAgent, FullInformationAgent,
                              ModelBasedAgent, ToTAgent]
        self.name2class = dict()

        for agent_class in self.agent_classes:
            self.name2class[agent_class.NAME] = agent_class

    def get_agent(self, agent_name, *args, **kwargs):

        if agent_name in self.name2class:
            return self.name2class[agent_name](*args, **kwargs)
        else:
            raise AssertionError(
                f"Unknown agent {agent_name}. Agent names that exists are {list(self.name2class.keys())}.")
