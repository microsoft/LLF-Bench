import random
from verbal_gym.utils.misc_utils import print_color, extract_number

class Agent:
    """ An agent that interacts with an environment with verbal feedback. """
    def __init__(self, *args, **kwargs):
        # TODO: pass env_spec
        pass

    def reset(self, docstring):
        self.docstring = docstring

    def act(self, *args, **kwargs):
        raise NotImplementedError

class RandomAgent(Agent):
    """ An random agent for a fixed number of actions. """

    def __init__(self, action_min, action_max):
        assert type(action_min)==int and type(action_max)==int
        self.action_min = action_min
        self.action_max = action_max

    def act(self, *args, **kwargs):
        return random.randint(self.action_min, self.action_max)

class BasicAgent(Agent):

    def __init__(self, llm, action_range, verbose=False):
        self.llm = llm
        self.action_range = action_range
        self.previous_action = None
        self.history = []
        self.verbose = verbose

        self.problem_prompt_template = """
            You're given with the problem below:

            Problem Description: {}
        """
        self.command_prompt = """
            Choose your action according to the problem description and explain why.

            The response should be in the following format:

                Reasoning for Decision: <your reasoning>
                Decision: <your action>

            Note that <your action> should be an integer from [{}, {}). You must follow this format!!
        """.format(self.action_range[0], self.action_range[1])

    def reset(self, docstring):
        self.docstring = docstring
        self.problem_prompt = self.problem_prompt_template.format(docstring)
        self.history = []

    def act(self, obs, feedback, **kwargs):
        history_prompt=''
        if len(self.history)>0:
            self.history[-1]['feedback'] = feedback
            formatted_strings = [f"    Action {item['action']}: Feedback {item['feedback']}" for item in self.history]
            history_str = '\n            '.join(formatted_strings)
            history_prompt0 = """
            You have observed the following history of action and feedback:\n
            """
            history_prompt = history_prompt0+history_str+'\n'

        user_prompt = self.problem_prompt + history_prompt + self.command_prompt

        if self.verbose:
            print_color('User: {}'.format(user_prompt), "blue")

        response, _ = self.llm.query(user_prompt)

        if self.verbose:
            print_color('Agent: {}'.format(response), "green")

        numbers = extract_number(response.split('\n')[-1])
        action = int(numbers[0]) if numbers else random.randint(self.action_range[0], self.action_range[1])

        self.history.append({'action': action, 'feedback': None})
        return action