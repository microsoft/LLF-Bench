import random
from textwrap import dedent, indent

from agents.abstract_agent import Agent
from verbal_gym.utils.misc_utils import print_color
from verbal_gym.agents.utils import extract_action


class FullInformationAgent(Agent):
    """ This is a helper function to use LLM to make decisions among K choices,
    given full information. """

    NAME = "FullInformationAgent"

    system_prompt = dedent("""
        You are an expert decision making agent. You will see "Problem
        Description" that tell you want to problem is about (such as the goal of
        the task, the action space you should choose from, the rules, the
        constraints, etc.) In addition, you will be presented with the feedback
        of all the acitons. You goal is to choose the right actions solve the
        task, according to "Problem Description".
    """)

    def __init__(self, llm, n_actions, verbose=False):
        super(Agent, self).__init__()
        self.llm = llm
        self.n_actions = n_actions
        self.verbose = verbose
        self.prompt_template = dedent("""\

        You're given with the problem below:

        Problem Description: {}

        You see the following actions and their feedbacks:

        {}

        Choose your action according to the problem description and explain why.

        The response should be in the following format:

            Reasoning for Decision: <your reasoning>
            Decision: #<your action>#

        Note that <your action> should be an integer from [0, {}). You must follow this format!!!
        """)

    def act(self, obs, feedback, full_information, **kwargs):
        world_info = '\n'.join([f'\t Action {k}: {v["feedback"]}' for k, v in full_information.items()])
        user_prompt = self.prompt_template.format(self.docstring, world_info, self.n_actions)

        response, _ = self.llm.generate(user_prompt)
        action = extract_action(response.split('\n')[-1], len(full_information))
        if self.verbose:
            print_color('User:\n{}'.format(user_prompt), "blue")
            print_color('Agent:\n{}'.format(response), "green")
            print_color(f'Action: {action}\n', 'red')
        return action

