from textwrap import dedent, indent

from agents.abstract_agent import Agent
from verbal_gym.utils.misc_utils import print_color
from verbal_gym.agents.utils import extract_action


class BasicAgent(Agent):

    NAME = "BasicAgent"

    system_prompt = dedent("""
    You are an agent tasked to solve an interactive problem with verbal
    feedback. You will see "Problem Description" that tell you want to problem
    is about (such as the goal of the task, the action space you should choose
    from, the rules, the constraints, etc.) After you choose an action, you will
    see the feedback from the environment. You goal is to choose the right
    actions solve the task as fast as possible, according to "Problem
    Description".
    """)

    def __init__(self, llm, n_actions, verbose=False, action_name='Action'):
        super(Agent, self).__init__()
        self.llm = llm
        self.n_actions = n_actions
        self.verbose = verbose
        self.action_name = action_name

        # history
        self.previous_action = None
        self.history = []

        self.prompt_template = dedent("""\
            You're given with the problem below:

            Problem Description: {}

            You have observed the following history of feedbacks:

            {}

            Choose your action according to the problem description, history of feedbacks, and explain why.

        """)

        if n_actions is not None:
            self.prompt_template += dedent(f"""\
            The response should be in the following format, where <your action> should be an integer from [0, {n_actions}). You must follow this format!!!

                Reasoning: <your reasoning>
                {action_name}: #<your action>#

            """)
        else:
            self.prompt_template += dedent(f"""\
            The response should be in the following format, where <your action> is the final answer. You must follow this format!!!

                Reasoning: <your reasoning>
                {action_name}: <your action>

            """)

    def reset(self, docstring):
        self.docstring = docstring
        self.history = []

    def update_history(self, feedback):
        world_info = 'None'
        if len(self.history) > 0:
            self.history[-1]['feedback'] = feedback
            # TODO how to format this nicely?
            # world_info = '\n'.join([f'\t {self.action_name} {item["action"]} --- {item["feedback"]}'
            # for item in self.history])
            world_info = '\n'.join(
                [indent(f'{self.action_name}: {item["action"]}\n\nFeedback: {item["feedback"]}\n\n\n','\t')
                 for item in self.history])

        return world_info

    def act(self, obs, feedback, **kwargs):
        world_info = self.update_history(feedback)
        user_prompt = self.prompt_template.format(self.docstring, world_info)
        response, _ = self.llm.generate(user_prompt)

        if self.verbose:
            print_color(f'User:\n\n{user_prompt}\n', "blue")
            print_color(f'Agent:\n\n{response}\n', "green")

        action = response.split(self.action_name+':')[-1]
        if self.n_actions is not None:
            action = extract_action(action, self.n_actions)

        if self.verbose:
            print_color(f'Action:\n\n{action}\n', 'red')
        self.history.append({'action': action, 'feedback': None})

        return action
