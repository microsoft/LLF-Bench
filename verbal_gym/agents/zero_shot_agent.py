from textwrap import dedent
from agents.abstract_agent import Agent
from agents.utils import extract_action, ReplayBuffer
from utils.misc_utils import print_color


class ZeroshotLLM(Agent):
    """ An zero-shot LLM agent. """

    NAME = "Random"

    def __init__(self, llm,
                 n_actions,
                 verbose=False,
                 action_name='Action',
                 buffer_size=0,
                 ignore_observation=True):
        super(Agent, self).__init__()

        self.llm = llm
        self.n_actions = n_actions
        self.verbose = verbose
        self.action_name = action_name
        self.past_observation = []
        self.ignore_observation = ignore_observation

        if self.ignore_observation:
            self.prompt_template = dedent("""\
                                You're presented with the problem below:

                                Problem Description: {}

                                Choose your action according to the problem description, and explain why.

                            """)
        else:
            self.prompt_template = dedent("""\
                                        You're presented with the problem below:

                                        Problem Description: {}

                                        You have noticed the following observation based of your past action:

                                        {}
                                        
                                        Your current observation is the following:
                                        
                                        {}

                                        Choose your action for the current observation. 
                                        You should use the aid of the problem description, past observations, and 
                                        current observation, and explain why.

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

        assert type(n_actions) == int
        self.n_actions = n_actions

    def reset(self, docstring):
        self.docstring = docstring
        self.past_observation = []

    def act(self, observation, feedback, **kwargs):

        # update with the latest feedback (ignored in the first call)
        # create prompt

        if self.ignore_observation:
            user_prompt = self.prompt_template.format(self.docstring)
        else:
            past_observations = " \n ".join(self.past_observation)
            user_prompt = self.prompt_template.format(self.docstring, past_observations, observation)

        self.past_observation.append(observation)

        response, _ = self.llm.generate(user_prompt)

        action = response.split(self.action_name+':')[-1]
        if self.n_actions is not None:
            action = extract_action(action, self.n_actions)

        if self.verbose:
            print_color(f'User:\n\n{user_prompt}\n', "blue")
            print_color(f'Agent:\n\n{response}\n', "green")
            print_color(f'Action:\n\n{action}\n', 'red')

        return action
