from textwrap import dedent, indent

from verbal_gym.agents.abstract_agent import Agent
from verbal_gym.utils.misc_utils import print_color
from verbal_gym.agents.utils import extract_action, ReplayBuffer

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

    def __init__(self,
                 llm,
                 n_actions,
                 verbose=False,
                 action_name='Action',
                 buffer_size=1000,
                 ignore_observation=True):
        """
            Args:
                llm: a large language model
                n_actions: number of actions
                verbose: whether to print out the prompt and response
                action_name: the name of the action
                buffer_size: the size of the replay buffer
                ignore_observation: whether to ignore the observation (for bandit setting)
        """


        super(Agent, self).__init__()
        self.llm = llm
        self.n_actions = n_actions
        self.verbose = verbose
        self.action_name = action_name
        self.buffer = ReplayBuffer(buffer_size)
        self.ignore_observation = ignore_observation

        self.prompt_template = dedent("""\
            You're given with the problem below:

            Problem Description: {}

            You have observed the following history of feedbacks:

            {}

            Choose your action according to the problem description, history of feedbacks, and explain why.

        """)

        if n_actions is not None:
            self.prompt_template += dedent(f"""\
            The response should be in the following format, where <your action> should be an integer from 0 and less than {n_actions}. You must follow this format!!!

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
        self.buffer.reset()

    @property
    def world_info(self):
        if len(self.buffer)==0:
            return ''
        if self.ignore_observation:
            world_info = '\n'.join(
                [indent(f'{self.action_name}: {item["action"]}\nFeedback: {item["feedback"]}\n\n','\t')
                for item in self.buffer])
        else:
            raise NotImplementedError

        return world_info

    def act(self, observation, feedback, **kwargs):

        # update with the latest feedback (ignored in the first call)
        self.buffer.update(feedback=feedback, next_observation=observation)
        world_info = self.world_info
        # create prompt
        user_prompt = self.prompt_template.format(self.docstring, world_info)
        response, _ = self.llm.generate(user_prompt)
        action = response.split(self.action_name+':')[-1]
        if self.n_actions is not None:
            action = extract_action(action, self.n_actions)

        if self.verbose:
            print_color(f'User:\n\n{user_prompt}\n', "blue")
            print_color(f'Agent:\n\n{response}\n', "green")
            print_color(f'Action:\n\n{action}\n', 'red')

        # update buffer and get world info
        self.buffer.append(observation=observation,
                           action=action,
                           feedback=None,
                           next_observation=None)
        return action
