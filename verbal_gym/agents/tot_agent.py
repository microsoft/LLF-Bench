import numpy as np
from collections import deque

from textwrap import dedent, indent
from verbal_gym.agents.utils import extract_action
from verbal_gym.utils.misc_utils import print_color
from verbal_gym.agents.basic_agent import BasicAgent

from verbal_gym.agents.parser_util import SimpleGuidanceParser


class VoterAgent:

    system_prompt = """You are an advanced reasoning agent that can identify the most promising avenue to explore from among a set of plans."""

    def __init__(self, llm, action_name='Action'):
        self.llm = llm
        self.parser = SimpleGuidanceParser("""
{{#system~}}
You are an advanced reasoning agent that can identify the most promising avenue to explore from among a set of plans.
{{~/system}}

{{#user~}}
You will be given a list of justifications produced by a decision-maker to reason about the optimal decision for their problem.
Pick the most promising justification from the list that will likely lead to the correct answer and optimal decision for the decision-maker.
Moreover, if you think that there is no further reasoning necessary, indicate so using the 'done' flag.
Your output should always be a string in the format id:done where id is the index of the most promising justification in the list of justifications.
For example, given the list <justification>justification1</justification><justification>justification2</justification><justification>justification3</justification>, 
if you think justification2 is the most promising and complete, your output should be 1:True.

Decision-Maker's Problem Description: {{observation}}

##########
Here are all the previous interactions and feedbacks that the decision-maker has collected about this problem:

{{world_info}}
##########                        

{{~#each responses}}
<justification>
{{action_name}}: {{this.action}}
Justification: {{this.response}}
</justification>
{{~/each}}

{{~/user}}

{{#assistant~}}
{{gen 'vote' temperature=0}}
{{~/assistant}}
""")
        self.action_name = action_name
        
    def __call__(self, observation, world_info, responses):
        response, _ = self.llm.generate(self.parser(responses=responses, action_name=self.action_name,
                                        observation=observation, world_info=world_info))
        return response


class ThinkerAgent:

    system_prompt = """You are an advanced reasoning agent that can identify the most optimal choice from among a set of options."""

    def __init__(self, llm, action_name='Action'):
        self.llm = llm
        self.parser = SimpleGuidanceParser("""
{{#system~}}
You are an advanced reasoning agent that can identify the most optimal choice from among a set of options.
{{~/system}}

{{#user~}}
You will be given a problem faced by a decision-maker, along with a set of actions available to them.
The decision-maker is considering a particular action as the optimal choice.
If you disagree that the particular action is not optimal, respond only with the single word "ERROR".
Otherwise, provide a justification why the action is indeed optimal. Use reasons like "<selected action> is better than <other action> because".
                                           
Decision-Maker's Problem Description: {{observation}}
                                           
##########
Here are all the previous interactions and feedbacks that the decision-maker has collected about this problem:

{{world_info}}
##########
                        
{{~#each actions}}
{{action_name}}: {{this.action}}
{{~/each}}

Considered optimal {{action_name}}: {{optimal_action}}
{{#if exists_justification}}
The following justification(s) were provided already. Augment them to convince the decision-maker about the optimality of the considered {{action_name}}. 
Justification:
{{optimal_justification}}
{{/if}}

{{~/user}}

{{#assistant~}}
{{gen 'thought' temperature=0}}
{{~/assistant}}
""")
        self.action_name = action_name
        
    def __call__(self, observation, world_info, actions, optimal_action, optimal_justification):
        response, _ = self.llm.generate(self.parser(actions=actions, action_name=self.action_name,
                                        observation=observation, optimal_action=str(optimal_action), world_info=world_info, 
                                        exists_justification = optimal_justification is not None, optimal_justification = optimal_justification))
        return response


class ToTAgent(BasicAgent):

    NAME = "ToTAgent"

    # Assuming a bandit verbal decision problem, at round T the ToTAgent does the following:
    # 1. For each discrete action (or a sampled set of actions), simulate a Response to justify why it is optimal relative to all the other actions
    # 2. Prompt a Voter to pick the most promising Response
    # 3. For the picked Response, simulate a Response that extends the justification
    # 4. Terminate after max_iter or if Voter decides to early terminate
    # 5. Goto Step 2

    def __init__(self, llm, n_actions, verbose=False, action_name='Action',
                 voter_agent=None, thinker_agent=None, permute_history=True, 
                 paraphrase_agent=None, logger=None, buffer_size=5, max_iter=10):
        super().__init__(llm, n_actions, verbose=verbose, action_name=action_name, buffer_size=buffer_size)
        self.permute_history = permute_history
        self.paraphrase_agent = paraphrase_agent

        self.voter_agent = voter_agent
        if voter_agent is not None:
            self.voter_agent.action_name = action_name # we do a quick sync here

        self.thinker_agent = thinker_agent
        if thinker_agent is not None:
            self.thinker_agent.action_name = action_name # we do a quick sync here

        self.logger = logger

        self.permute_history = permute_history
        self.max_iter = max_iter
        self.docstring = None

    def simulate_feedback_for_discrete_action(self, world_info, action_list):
        # Create the environment
        dataset = []
        
        for action_item in action_list:
            thought = self.thinker_agent(self.docstring, world_info, action_list, action_item['action'], None)
            
            if self.logger is not None:
                self.logger.log(f"Action {action_item['action']}\n")
                self.logger.log(f"Thought: {thought}\n\n")
            
            if not thought.startswith("ERROR"):
                dataset.append({'action': action_item['action'], 'response': thought})

        return dataset

    def act(self, obs, feedback, **kwargs):
        # Update history. History only contains things which actually happened
        self.buffer.update(feedback=feedback, next_observation=obs)
        world_info = self.world_info
        if self.n_actions is None:
            raise NotImplementedError
        
        action_list = []
        for action in range(self.n_actions):
            action_list.append({'action': str(action)})

        vote = None
        for i in range(self.max_iter):
            self.simulated_feedback = self.simulate_feedback_for_discrete_action(world_info, action_list)
            
            vote = self.voter_agent(self.docstring, world_info, self.simulated_feedback)
            done = vote.split(':')[1]
            if done.startswith('True'):
                break
        
        action = int(vote.split(':')[0])
        selected_action = action_list[action]['action']

        if self.n_actions is not None:
            action = extract_action(selected_action, self.n_actions)

        if self.verbose:
            if self.logger is not None:
                self.logger.log(f"Action:\n\n{action}\n")

            print_color(f'Action:\n\n{action}\n', 'red')

        self.history.append({'action': action, 'feedback': None})

        return action

    def paraphrase(self, sentence):
        return self.paraphrase_agent.paraphrase(sentence) if self.paraphrase_agent is not None else sentence
