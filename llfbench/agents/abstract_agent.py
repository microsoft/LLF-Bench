from llfbench.agents.utils import ReplayBuffer

class Agent:
    """ An agent that interacts with an environment with language feedback. """

    NAME = "AbstractAgent"

    def __init__(self, *args, **kwargs):
        # TODO: pass env_spec
        self.docstring = None

    def reset(self, docstring):
        self.docstring = docstring

    def act(self, *args, **kwargs):
        raise NotImplementedError


class BasicAgent(Agent):

    NAME = "BasicAgent"

    def __init__(self,
                 verbose=False,
                 buffer_size=20,
                 prompt_template=None):
        """
            Args:
                verbose: whether to print out the prompt and response
                buffer_size: the size of the replay buffer
        """

        super().__init__()
        self.verbose = verbose
        self.buffer = ReplayBuffer(buffer_size)
        self.prompt_template = prompt_template

    def reset(self, docstring):
        self.docstring = docstring
        self.buffer.reset()