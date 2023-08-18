from verbal_gym.llm.gpt.abstract_gpt import AbstractGPT


class GPT3(AbstractGPT):

    def __init__(self):
        super(GPT3, self).__init__(deployment_name="text-davinci-003")


class GPT35(AbstractGPT):

    def __init__(self):
        super(GPT35, self).__init__(deployment_name="gpt-35-turbo")
