from verbal_gym.gpt.abstract_gpt import AbstractGPT


class GPT3(AbstractGPT):

    def __init__(self):
        super(AbstractGPT, self).__init__(deployment_name="text-davinci-003")


class GPT35(AbstractGPT):

    def __init__(self):
        super(AbstractGPT, self).__init__(deployment_name="gpt-35-turbo")
