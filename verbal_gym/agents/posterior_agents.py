from verbal_gym.agents.agents import BasicAgent
from verbal_gym.agents.utils import extract_action
from verbal_gym.utils.misc_utils import print_color

import numpy as np
from textwrap import dedent, indent


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

    def __init__(self, llm, n_actions, verbose=False, action_name='Action',
                permute_history=True, paraphrase_agent=None):
        super().__init__(llm, n_actions, verbose=verbose, action_name=action_name)
        self.permute_history = permute_history
        self.paraphrase_agent = paraphrase_agent

    def reset(self, docstring):
        self._docstring = docstring

    def update_history(self, feedback):
        world_info='None'
        if len(self.history)>0:
            self.history[-1]['feedback'] = feedback
            # XXX Add random permuation and paraphrasing
            history = np.random.permutation(self.history) if self.permute_history else self.history
            world_info = '\n'.join([ indent(f'{self.action_name}: {item["action"]}\n\nFeedback: {self.paraphrase(item["feedback"])}\n\n\n','\t') for item in history])
        return world_info

    def paraphrase(self, sentence):
        return self.paraphrase_agent.paraphrase(sentence) if self.paraphrase_agent is not None else sentence

    @property
    def docstring(self):
        return self.paraphrase(self._docstring)