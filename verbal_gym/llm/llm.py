from abc import ABC
class LLM(ABC):
    """ This class represents a black box LLM. """

    def __init__(self, system_prompt='', *args, **kwargs):
        """ Initialize the LLM """
        self.system_prompt = system_prompt
        self.reset()

    def reset(self):
        """ This resets the LLM and removes the chat history. """
        raise NotImplementedError

    def chat(self, user_prompt, *args, **kwargs):
        """ This is a history-dependent chat response. """
        raise NotImplementedError
        return response, info

    def query(self, user_prompt, *args, **kwargs):
        """ This is one-time query response. """
        raise NotImplementedError
        return response, info
