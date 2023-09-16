from textwrap import dedent, indent

from verbal_gym.agents.abstract_agent import Agent
from verbal_gym.utils.misc_utils import print_color
from verbal_gym.agents.utils import extract_action, ReplayBuffer


class BasicAgent(Agent):

    NAME = "BasicAgent"

    system_prompt = dedent("""
    You are an agent tasked to solve an interactive problem with verbal
    feedback. You will see "Problem Description" that tells you what the problem
    is about (such as the goal of the task, the action space you should choose
    from, the rules, the constraints, etc.). After you choose an action, you will
    see the feedback from the environment. Your goal is to choose the right
    actions to solve the task as fast as possible, according to "Problem
    Description".
    """)

    def __init__(self,
                 llm,
                 n_actions,
                 verbose=False,
                 action_name='Action',
                 buffer_size=1000,
                 ignore_observation=False,
                 prompt_template=None):
        """
            Args:
                llm: a large language model
                n_actions: number of actions
                verbose: whether to print out the prompt and response
                action_name: the name of the action
                buffer_size: the size of the replay buffer
                ignore_observation: whether to ignore the observation (for bandit setting)
                prompt_template: A prompt template with two parameters if ignore_observation is True and 3 otherwise
        """

        super(Agent, self).__init__()
        self.llm = llm
        self.n_actions = n_actions
        self.verbose = verbose
        self.action_name = action_name
        self.buffer = ReplayBuffer(buffer_size)
        self.ignore_observation = ignore_observation

        if prompt_template is not None:
            self.prompt_template = prompt_template
        else:

            if ignore_observation:
                self.prompt_template = dedent("""\
                    You're presented with the problem below:
        
                    Problem Description: {}
        
                    You have observed the following history of feedbacks:
        
                    {}
        
                    Choose your action according to the problem description, history of feedbacks, and explain why.
        
                """)
            else:
                self.prompt_template = dedent("""
                    
                    You're presented with the problem below: 
                    
                    Problem Description: {}

                    You have in the past taken the following path which consists of observations you saw, the actions 
                    you took, and the feedback you got for those actions:

                     {}
                     
                     You are currently observing the following {}.

                    Choose your action according to the problem description, your past history of actions, and your 
                    current observation and explain why.
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

        if self.ignore_observation:

            if len(self.buffer) == 0:
                world_info = 'None'
            else:
                world_info = '\n'.join(
                    [indent(f'{self.action_name}: {item["action"]}\n\nFeedback: {item["feedback"]}\n\n\n','\t')
                     for item in self.buffer])
        else:
            # We present the observation and feedback as
            # you took action <action>
            # this resulted in <observation>
            # and you got a feedback of <feedback>

            if len(self.buffer) == 0:
                world_info = 'None'
            else:
                world_info = "\n".join([
                    f"Observation: {dp['observation']}\n Action: {dp['action']}\n Feedback: {dp['feedback']}"
                    for dp in self.buffer])

        return world_info

    def act(self, observation, feedback, **kwargs):

        # update with the latest feedback (ignored in the first call)
        self.buffer.update(feedback=feedback, next_observation=observation)
        world_info = self.world_info

        # create prompt
        if self.ignore_observation:
            user_prompt = self.prompt_template.format(self.docstring, world_info)
        else:
            user_prompt = self.prompt_template.format(self.docstring, world_info, observation)

        response, _ = self.llm.generate(user_prompt, max_tokens=1000)

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
