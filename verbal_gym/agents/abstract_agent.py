import numpy as np
from typing import Union, Dict, Any

class Agent:
    """ An agent that interacts with an environment with verbal feedback. """

    NAME = "AbstractAgent"

    def act(self, observation: Union[str, Dict[str, Any]]) -> Any:
        """ This is called at each step.

            Args:
                observation: the observation from the environment. It can be a
                string or a dict with keys "observation", "feedback", and
                "task".

            Returns:
                action: the action to take. It can be an int, a float, a string,
                and np.ndarray depending on the environment.
        """
        raise NotImplementedError
