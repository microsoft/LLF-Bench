from verbal_gym.llm.llm import LLM
from verbal_gym.llm.openai_utils import call_model

class GPT(LLM):
    """ This is based on ChatCompletion api. """

    def __init__(self,
                 system_prompt='', *,
                 model="gpt-35-turbo",
                 temperature=0.0,
                 timeout=100,
                 max_tokens=None,
                 max_attempts=50,
                 **kwargs):
        """ Initialize the LLM """
        super().__init__(system_prompt)
        self.spec = dict(
            model = model,
            temperature = temperature,
            timeout = timeout,
            max_tokens = max_tokens,
            max_attempts = max_attempts,
        )

    def reset(self):
        """ This resets the LLM and removes the chat history. """
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def chat(self, prompt, **kwargs):
        """ This is a history-dependent response. """
        self.messages.append({"role": "user", "content": prompt})
        spec = self.spec.copy()
        spec.update(kwargs)
        return call_model(self.messages, **spec)

    def generate(self, prompt, **kwargs):
        """ This is one-time query response. """
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}]
        spec = self.spec.copy()
        spec.update(kwargs)
        return call_model(messages, **spec)
