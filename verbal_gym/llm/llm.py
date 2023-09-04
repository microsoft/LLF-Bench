from abc import ABC
class LLM(ABC):
    """ This class represents a black box LLM. """
    # NOTE when subclassing this class, please include **kwargs in each method.

    def __init__(self, system_prompt='', **kwargs):
        """ Initialize the LLM """
        self.system_prompt = system_prompt
        self.reset()

    def reset(self):
        """ This resets the LLM and removes the chat history. """
        raise NotImplementedError

    def chat(self,
             prompt, *,
             logprob=None, # None or int. If int, return top-k logprobs.
             timeout,  # Maximum time to wait if failure occurs before re-trying
             temperature,  # temperature of the generation
             max_tokens,  # maximum number of tokens to generate
             max_attempts):  # maximum number of attempts to call the model
        """ This is a history-dependent chat response. """
        raise NotImplementedError
        # info is dict with keys 'logprobs' and 'response', where 'response' is the original response object from the model.
        return generation, info

    def generate(self,
                 prompt, *,
                 logprob=None, # None or int. If int, return top-k logprobs.
                 timeout,  # Maximum time to wait if failure occurs before re-trying
                 temperature,  # temperature of the generation
                 max_tokens,  # maximum number of tokens to generate
                 max_attempts):  # maximum number of attempts to call the model

        """ This is one-time query response. """
        raise NotImplementedError
        # info is dict with keys 'logprobs' and 'response', where 'response' is the original response object from the model.        return response, info
        return generation, info


    def logprob(self,
                prompt, *,
                timeout,  # Maximum time to wait if failure occurs before re-trying
                temperature,  # temperature of the generation
                max_tokens,  # maximum number of tokens to generate
                max_attempts):  # maximum number of attempts to call the model
        """ This is one-time query response. """
        raise NotImplementedError
        return logprob
