from verbal_gym.agents.agents import BasicAgent
from verbal_gym.agents.utils import extract_action
from verbal_gym.utils.misc_utils import print_color

import numpy as np
from textwrap import dedent


class ParaphraseAgent:
    system_prompt = dedent("""
    You are an expert in paraphrasing. You will see a sentence and you need to
    paraphrase it compactly.
    """)

    def __init__(self, llm):
        self.llm = llm
        self.promot_template = dedent("""\
            You see the following sentence:
            {}
            Paraphrase:
        """)

    def paraphrase(self, sentence):
        response, _ = self.llm.generate(self.promot_template.format(sentence))
        return response


class PosteriorAgent(BasicAgent):

    def __init__(self, llm, n_actions, verbose=False,
                permute_history=True, paraphrase_agent=None):
        super().__init__(llm, n_actions, verbose=verbose)
        self.permute_history = permute_history
        self.paraphrase_agent = paraphrase_agent

    def update_history(self, feedback):
        world_info=''
        if len(self.history)>0:
            self.history[-1]['feedback'] = feedback
            # XXX Add random permuation and paraphrasing
            history = np.random.permutation(self.history) if self.permute_history else self.history
            world_info = '\n'.join([f'\t Action {item["action"]}: {self.paraphrase(item["feedback"])}' for item in history])
        return world_info

    def paraphrase(self, sentence):
        return self.paraphrase_agent.paraphrase(sentence) if self.paraphrase_agent is not None else sentence

    def act(self, obs, feedback, **kwargs):
        # XXX Add random permuation and paraphrasing
        world_info = self.update_history(feedback)
        docstring =  self.paraphrase(self.docstring)
        # Below is the original code
        user_prompt = self.prompt_template.format(docstring, world_info, self.n_actions)
        response, _ = self.llm.generate(user_prompt)
        action = extract_action(response.split('\n')[-1], self.n_actions)

        if self.verbose:
            print_color(f'Original: {self.docstring}\nParaphrase: {docstring}', 'green')
            print_color(f'User: {user_prompt}', "blue")
            print_color(f'Agent: {response}', "green")
            print_color(f'Action: {action}', 'red')

        self.history.append({'action': action, 'feedback': None})

        return action