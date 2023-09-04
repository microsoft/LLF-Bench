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

        if paraphrase_at_given:
            # We overwrite buffer.update and buffer.append
            def paraphrase_wrapper(buffer_update):
                def wrapper(**kwargs):
                    for k in kwargs:
                        if k in ('feedback'):  # TODO: add more if needed
                            kwargs[k] = self.paraphrase(kwargs[k])
                    return buffer_update(**kwargs)
                return wrapper
            self.buffer.update = paraphrase_wrapper(self.buffer.update)
            self.buffer.append = paraphrase_wrapper(self.buffer.append)

    @property
    def world_info(self):
        if len(self.buffer)==0:
            return ''
        if self.ignore_observation:
            paraphrase = self.paraphrase if not self.paraphrase_at_given else lambda x: x
            world_info = [indent(f'{self.action_name}: {item["action"]}\nFeedback: {paraphrase(item["feedback"])}\n\n','\t') for item in self.buffer]
            world_info = np.random.permutation(world_info) if self.permute_history else world_info
            world_info = '\n'.join(world_info)
        else:
            raise NotImplementedError
        return world_info

    def paraphrase(self, sentence):
        if sentence is None or sentence == '':
            return ''
        return self.paraphrase_agent.paraphrase(sentence) if self.paraphrase_agent is not None else sentence

    @property
    def docstring(self):
        return self.paraphrase(self._docstring)

    @docstring.setter
    def docstring(self, value):
        self._docstring = value