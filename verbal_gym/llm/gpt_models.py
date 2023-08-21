from verbal_gym.llm.llm import LLM
from verbal_gym.llm.openai_utils import call_model

class GPT(LLM):
    """ This is based on ChatCompletion api. """

    def __init__(self,
                 system_prompt='', *,
                 model="gpt-35-turbo",
                 temperature=0.0,
                 request_timeout=100,
                 max_tokens=None,
                 max_attempts=50):
        """ Initialize the LLM """
        super().__init__(system_prompt)
        self.temperature = temperature
        self.spec = dict(
            request_timeout = request_timeout,
            max_tokens = max_tokens,
            max_attempts = max_attempts,
            model = "gpt-35-turbo"
        )

    def reset(self):
        """ This resets the LLM and removes the chat history. """
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def chat(self, user_prompt, temperature=None):
        """ This is a history-dependent response. """
        self.messages.append({"role": "user", "content": user_prompt})
        temperature = temperature or self.temperature
        response, info = call_model(self.messages, temperature=temperature, **self.spec)
        return response, info

    def generate(self, user_prompt, temperature=None):
        """ This is one-time query response. """
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}]
        temperature = temperature or self.temperature
        response, info = call_model(messages, temperature=temperature, **self.spec)
        return response, info
