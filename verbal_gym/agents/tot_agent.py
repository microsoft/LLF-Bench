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
Moreover, if you think that there is no further reasoning necessary, indicate so using a boolean 'done' flag.
Your output should always be a string in the format done:id where id is the index of the most promising justification in the list of justifications.
For example, given the list <justification>justification0</justification><justification>justification1</justification><justification>justification2</justification>, 
if you think justification1 is the most promising and complete, your output should be True:1
If however you think that justification0 is the most promising but not yet complete, your answer should be False:0
The text below contains all of the information that the decision-maker knows about their problem so far.
Put yourself in the shoes of the decision-maker and commit to taking an action now.
In that way they may gather additional information that can help you make a more informed guess in the future.

Decision-Maker's Problem Description: {{observation}}

##########
Here are all the previous interactions and feedbacks that the decision-maker has collected about this problem:

{{world_info}}
##########                        

{{#each responses}}
<justification>
{{this.response}}
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
Even if you need additional information to make the optimal decision, you need to commit to a decision that best hedges against uncertainty now.
As the decision-maker continues interacting, they may gather additional information that may help you make a more informed guess in the future.
                                           
Decision-Maker's Problem Description: {{observation}}
                                           
##########
Here are all the previous interactions and feedbacks that the decision-maker has collected about this problem:

{{world_info}}
##########

Considered optimal {{action_name}}: {{optimal_action}}
{{#if exists_justification}}
The following justification(s) were provided already. Augment them to convince the decision-maker about the optimality of the considered {{action_name}}. 
Justification:
{{optimal_justification}}
{{/if}}

Available Actions:
{{#each actions}}
{{action_name}}: {{this.action}}
{{~/each}}

{{~/user}}

{{#assistant~}}
{{gen 'thought' temperature=0}}
{{~/assistant}}
""")
        self.action_name = action_name
        
    def __call__(self, observation, world_info, actions, optimal_action, optimal_justification):
        text = self.parser(observation=observation, world_info=world_info, actions=actions, action_name=self.action_name,
                                         optimal_action=str(optimal_action), 
                                        exists_justification = optimal_justification is not None, optimal_justification = optimal_justification)
        response, _ = self.llm.generate(text)
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
                 paraphrase_agent=None, logger=None, buffer_size=5, max_iter=10, num_simulated_actions=4):
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
        self.num_simulated_actions = num_simulated_actions
        
    def simulate_feedback_for_discrete_action(self, world_info, action_list):
        # Create the environment
        dataset = []
        
        for action_item in action_list:
            justification = None
            if 'response' in action_item.keys():
                justification = action_item['response']
            thought = self.thinker_agent(self.docstring, world_info, action_list, action_item['action'], justification)
            
            if self.verbose:
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
        
        action_list = []
        if self.n_actions is not None:
            for action in range(self.n_actions):
                action_list.append({'action': str(action)})
        else:
            user_prompt = self.prompt_template.format(self.docstring, world_info)
            user_prompt += "\n Ensure that your answer differs from these existing answers (if any).\n"
            for i in range(self.num_simulated_actions):
                response, _ = self.llm.generate(user_prompt)
                action = response.split(self.action_name+':')[-1]
                action_list.append({'action': action})
                user_prompt += "\n "+self.action_name + ": "+action+"\n"
            
        vote = None
        selected_action = None
        for i in range(self.max_iter):           
            returned_data = self.simulate_feedback_for_discrete_action(world_info, action_list)
            if len(returned_data)>0:
                self.simulated_feedback = returned_data
            for action_item in self.simulated_feedback:
                if self.n_actions is not None:
                    action_id = int(action_item['action'])
                else:
                    action_id = 0
                    for j in range(len(action_list)):
                        if action_list[j]['action'] == action_item['action']:
                            action_id = j
                            break
                if 'response' in action_list[action_id].keys():
                    action_list[action_id]['response'] += ";" + action_item['response']
                else:
                    action_list[action_id]['response'] = action_item['response']
            
            if len(self.simulated_feedback)>1:
                vote = self.voter_agent(self.docstring, world_info, self.simulated_feedback)
            else:
                vote = "True:0"
            if self.verbose:
                if self.logger is not None:
                    self.logger.log(f"Vote: {vote}\n")
            tokens = vote.split(':')
            done = tokens[0]
            if done.startswith('True'):
                break
        
        index = 0
        if len(self.simulated_feedback)>1:
            index = extract_action(vote, len(self.simulated_feedback), separator=':')
        selected_action = self.simulated_feedback[index]['action']
                
        if self.verbose:
            if self.logger is not None:
                self.logger.log(f"Selected Action:\n\n{selected_action}\n")

            print_color(f'Action:\n\n{selected_action}\n', 'red')

        # update buffer and get world info
        self.buffer.append(observation=obs,
                           action=selected_action,
                           feedback=None,
                           next_observation=None)

        if self.n_actions is not None:
            selected_action = int(selected_action)        
        return selected_action

    def paraphrase(self, sentence):
        return self.paraphrase_agent.paraphrase(sentence) if self.paraphrase_agent is not None else sentence
