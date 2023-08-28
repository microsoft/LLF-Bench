import numpy as np
from textwrap import dedent, indent

from verbal_gym.agents.basic_agent import BasicAgent


class ParaphraseAgent:

    system_prompt = dedent("""
    You are an expert in paraphrasing. You will see a sentence and you need to
    paraphrase it compactly, while keeping the numerics.
    """)

    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = dedent("""\
            You see the following sentence:
            {}
            Paraphrase:
        """)

    def paraphrase(self, sentence):
        response, _ = self.llm.generate(self.prompt_template.format(sentence))
        return response


class PosteriorAgent(BasicAgent):

    NAME = "PosteriorAgent"

    def __init__(self, llm, n_actions, verbose=False, action_name='Action',
                permute_history=True, paraphrase_agent=None, paraphrase_at_given=True):
        super().__init__(llm, n_actions, verbose=verbose, action_name=action_name)
        self.permute_history = permute_history
        self.paraphrase_agent = paraphrase_agent
        self.paraphrase_at_given = paraphrase_at_given

    def reset(self, docstring):
        self._docstring = docstring

    def update_history(self, feedback):
        # XXX Add random permuation and paraphrasing

        if self.paraphrase_at_given:  # Here we just paraphrase the feedback when it is given.
            history_text = 'None'

            if len(self.history) > 0:
                self.history[-1]['feedback'] = self.paraphrase(feedback)
                history = np.random.permutation(self.history) if self.permute_history else self.history
                history_text = '\n'.join(
                    [indent(f'{self.action_name}: {item["action"]}\n\nFeedback: {item["feedback"]}\n\n\n','\t')
                     for item in history])

        else:  # Here we do paraphrasing when we need to return history observations.
            history_text = 'None'
            if len(self.history) > 0:
                self.history[-1]['feedback'] = feedback
                history = np.random.permutation(self.history) if self.permute_history else self.history
                history_text = '\n'.join(
                    [indent(f'{self.action_name}: {item["action"]}\n\nFeedback: {self.paraphrase(item["feedback"])}\n\n\n','\t')
                     for item in history])
        return history_text

    def paraphrase(self, sentence):
        return self.paraphrase_agent.paraphrase(sentence) if self.paraphrase_agent is not None else sentence

    @property
    def docstring(self):
        return self.paraphrase(self._docstring)