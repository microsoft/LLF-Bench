from verbal_gym.llm.gpt.abstract_gpt import AbstractGPT


class GPT3(AbstractGPT):

    def __init__(self):
        super(GPT3, self).__init__(deployment_name="text-davinci-003")


class GPT35(AbstractGPT):

    def __init__(self):
        super(GPT35, self).__init__(deployment_name="gpt-35-turbo")

    def logprob(self, prompt, *args, **kwargs):
        raise NotImplementedError("Cannot get log-prob for model gpt-35-turbo")

    def get_logprobs(self, prompt, MAX_WAITTIME_SEC=300):
        raise NotImplementedError("Cannot get log-prob for model gpt-35-turbo")

    def generate(self, prompt, max_tokens=None, logprobs=None, temperature=0.0, echo=False, MAX_WAITTIME_SEC=300):

        if logprobs is not None:
            raise NotImplementedError("Cannot get log-prob for model gpt-35-turbo")

        generation, info = super().generate(prompt,
                                            max_tokens=max_tokens,
                                            logprobs=None,
                                            temperature=temperature,
                                            echo=echo,
                                            MAX_WAITTIME_SEC=MAX_WAITTIME_SEC)

        return generation, info
