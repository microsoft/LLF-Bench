"""
Adapted from https://github.com/noahshinn024/reflexion/blob/main/hotpotqa_runs/prompts.py
"""

from collections import deque

from verbal_gym.agents.parser_util import SimpleGuidanceParser
from verbal_gym.agents.basic_agent import BasicAgent

from verbal_gym.utils.misc_utils import print_color

class ReActAgent(BasicAgent):

    NAME = "ReActAgent"

    def __init__(self, llm, n_actions, verbose=False, action_name='Action',
                buffer_size=5):
        super().__init__(llm, n_actions, verbose=verbose, action_name=action_name)
        self.thought_prompt = SimpleGuidanceParser("""
{{#system~}}
You are an advanced reasoning agent that can think and analyze the situation.
{{~/system}}

{{#user~}}
Problem Description: {{observation}}

Before you actually solve the problem, you want to think about how to do this problem.
Thought should reason about the current problem and inform how to best solve the problem.

Thought:
{{~/user}}

{{#assistant~}}
{{gen 'thought' temperature=0.7}}
{{~/assistant}}
""")

        self.prompt = SimpleGuidanceParser("""
{{#system~}}
You are an agent tasked to solve an interactive problem with verbal
feedback. You will see "Problem Description" that tell you want to problem
is about (such as the goal of the task, the action space you should choose
from, the rules, the constraints, etc.) After you choose an action, you will
see the feedback from the environment. You goal is to choose the right
actions solve the task as fast as possible, according to "Problem Description".
{{~/system}}

{{#user~}}
Problem Description: {{observation}}

{{#if exists_history}}
Past interacions:
{{~#each history}}
{{action_name}}: {{this.action}}
Thought: {{this.thought}}
Feedback: {{this.feedback}}

{{~/each}}
{{/if}}

Problem Description: {{observation}}
Thought: {{thought}}
{{action_name}}:
{{~/user}}

{{#assistant~}}
{{gen 'poem' temperature=0.7}}
{{~/assistant}}
""")
        self.history = deque(maxlen=buffer_size)

    def act(self, observation, feedback, **kwargs):
        # fill the history buffer
        # no reflection for the first time
        exists_history = True if len(self.history) > 0 else False

        # generate thoughts
        messages = self.thought_prompt(observation=observation)
        thought, _ = self.llm.generate(messages)

        messages = self.prompt(observation=observation,
                               thought=thought,
                               action_name=self.action_name,
                               history=self.history,
                               exists_history=exists_history)
        action, _ = self.llm.generate(messages)

        if self.verbose:
            user_prompt = self.prompt.decode_typed_messages(messages)
            print_color(f'User:\n\n{user_prompt}\n', "blue")
            print_color(f'Agent:\n\n{action}\n', "green")


        self.history.append({'observation': observation,
                             'action': action,
                             'feedback': feedback,
                             'thought': thought})

        return action

    @property
    def docstring(self):
        return self._docstring

    @docstring.setter
    def docstring(self, value):
        self._docstring = value