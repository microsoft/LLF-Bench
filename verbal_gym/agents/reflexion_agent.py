"""
Adapted from https://github.com/noahshinn024/reflexion/blob/main/hotpotqa_runs/prompts.py
"""

from collections import deque

from verbal_gym.agents.parser_util import SimpleGuidanceParser
from verbal_gym.agents.basic_agent import BasicAgent

from verbal_gym.utils.misc_utils import print_color


class ReflectAgent:

    system_prompt = """You are an advanced reasoning agent that can improve based on self refection. """

    def __init__(self, llm, max_history=5, action_name='Action'):
        self.llm = llm
        self.parser = SimpleGuidanceParser("""
{{#system~}}
You are an advanced reasoning agent that can improve based on self refection. 
{{~/system}}

{{#user~}}
You have attempted to do the following task before and failed. 
The following reflection(s) give a plan to avoid failing to doing the task in the same way you did previously. 
Use them to improve your strategy of correctly doing the given task.
In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. 
Use complete sentences.  

{{#if exists_reflection_examples}}
Here are some examples:
{{~#each examples}}
Problem Description: {{this.observation}}
{{action_name}}: {{this.action}}
Feedback: {{this.feedback}}
Reflection: {{this.reflection}}

{{~/each}}
{{/if}}

Previous trial:
Problem Description: {{observation}}
{{action_name}}: {{action}}
Feedback: {{feedback}}
Reflection:
{{~/user}}

{{#assistant~}}
{{gen 'reflection' temperature=0.7}}
{{~/assistant}}
""")
        self.action_name = action_name
        self.reflection_examples = deque(maxlen=max_history)

    def __call__(self, observation, action, feedback):
        response, _ = self.llm.generate(self.parser(exists_reflection_examples=len(self.reflection_examples) > 0,
                                        examples=self.reflection_examples, action_name=self.action_name,
                                        observation=observation, action=action, feedback=feedback))
        self.reflection_examples.append({'observation': observation,
                                         'action': action,
                                         'feedback': feedback,
                                         'reflection': response})
        return response


class ReflexionAgent(BasicAgent):

    NAME = "ReflexionAgent"

    def __init__(self, llm, n_actions, verbose=False, action_name='Action',
                 reflection_agent=None, permute_history=True, buffer_size=5):
        super().__init__(llm, n_actions, verbose=verbose, action_name=action_name)
        self.prompt = SimpleGuidanceParser("""
{{#system~}}
You are an agent tasked to solve an interactive problem with verbal
feedback. You will see "Problem Description" that tell you want to problem
is about (such as the goal of the task, the action space you should choose
from, the rules, the constraints, etc.) After you choose an action, you will
see the feedback from the environment. You goal is to choose the right
actions solve the task as fast as possible, according to "Problem
Description".
{{~/system}}

{{#user~}}

Problem Description: {{observation}}

{{#if exists_reflection}}
You have attempted to solve the problem before and failed. 
The following reflection(s) give a plan to avoid failing to solve the problem in the same way you did previously. 
Use them to improve your strategy of correctly solving the given problem.

Past interacions:
{{~#each history}}
{{action_name}}: {{this.action}}
Feedback: {{this.feedback}}
Reflection: {{this.reflection}}

{{~/each}}
{{/if}}

Problem Description: {{observation}}
{{action_name}}:
{{~/user}}

{{#assistant~}}
{{gen 'poem' temperature=0.7}}
{{~/assistant}}
""")
        self.reflection_agent = reflection_agent
        self.reflection_agent.action_name = action_name # we do a quick sync here

        self.permute_history = permute_history
        self.history = deque(maxlen=buffer_size)

    def act(self, observation, feedback, **kwargs):
        # fill the history buffer
        # no reflection for the first time
        exists_reflection = True if len(self.history) > 0 else False
        messages = self.prompt(observation=observation,
                               action_name=self.action_name,
                               history=self.history,
                               exists_reflection=exists_reflection)
        action, _ = self.llm.generate(messages)

        if self.verbose:
            user_prompt = self.prompt.decode_typed_messages(messages)
            print_color(f'User:\n\n{user_prompt}\n', "blue")
            print_color(f'Agent:\n\n{action}\n', "green")

        # generate reflection
        reflection = self.reflection_agent(observation, action, feedback)
        self.history.append({'observation': observation,
                             'action': action,
                             'feedback': feedback,
                             'reflection': reflection})

        return action

    @property
    def docstring(self):
        return self._docstring

    @docstring.setter
    def docstring(self, value):
        self._docstring = value