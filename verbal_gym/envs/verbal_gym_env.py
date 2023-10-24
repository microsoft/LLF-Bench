import gym
import numpy as np
from typing import Dict, Any, Tuple, Union, List

"""

Naming convention for envs:

[basename]-[instruction_type]-[feedback_type]-[version]

instruction_type:
    - (b) Basic (Goal, Syntax, Action Space)
    - (p) Partial (Basic + Offline data (e.g., observation, action, feedback))
    - (c) Complete (Info sufficient to infer the optimal policy)

feedback_type;
    - (n) none
    - (m) mixture: mix of (r), (hp), (hn), (fp), (fn)
    - (r) reward: textualization of reward
    - (hp) hindsight positive: explaination on why something is correct
    - (hn) hindsight negative: explaination on why something is incorrect
    - (fp) future positive: suggestion of things to do
    - (fn) future negative: suggestion of things to avoid

An example env name is: gridworld-b-fn-v0

"""

class VerbalGymWrapper(gym.Wrapper):
    """
        This is the wrapper that turns a gym environment into a verbal gym
        environment.

        In verbal-gym, the environment's reward is not provided to the agent.
        Instead the agent learns from info of instructions, observations, and
        their feedback.

        We present this info to the agent via the an observation dict, which has
        keys: 'instruction', 'observation', 'feedback'. The 'instruction' is a
        string containing the task instruction and optionally examples or other
        prior information that might help explain the task. The 'observation' is
        a (partial) observation of the environment state. The 'feedback' is a
        string containing formative feedback for learning (which is a
        replacement for reward in RL). If any attribute is missing, it is
        represented as None. But at the beginning `instruction` must not be None
        and 'feedback' must be None.

        This wrapper is backward compatible with text-based gym environments,
        which returns a string as observation. In this case, the initial
        observation is treated as the instruction, and the reward is textualized
        and treated as the feedback.

        This wrapper mainly implments format checking and a helper method for
        sampling from a set of paraphrased prompts.

        Instruction for subclassing:

        Implment methods (_reset and _step) and update the supported
        INSTRUCTION_TYPES and FEEDBACK_TYPES. See the convension above for the
        explnation of these types.
    """

    # These are the instruction and feedback types that are supported by this environment.
    INSTRUCTION_TYPES = ('b', 'p', 'c')
    FEEDBACK_TYPES = ('m', 'n', 'r', 'hp', 'hn', 'fp', 'fn')

    def __init__(self, env : gym.Env, instruction_type : str, feedback_type: str, paraphrase_idx : Union[None, int] = None):
        """
            Initialize the wrapper.

            Args:
                env: The original gym environment.

                instruction_type: The type of instruction. b: basic, p: partial,
                c: complete. Should be one of the INSTRUCTION_TYPES.

                feedback_type: The type of feedback. m: mixed, n: none, r:
                reward, hp: hindsight positive, hn: hindsight negative, fp:
                future positive, fn: future negative. Should be one of the
                FEEDBACK_TYPES.

                paraphrase_idx: The index of the paraphrased prompt to use. If
                None, a randomly selected prompt is used.
        """
        super().__init__(env)
        self.instruction_type = instruction_type
        self.feedback_type = feedback_type # This is the external api.
        self._feedback_type = feedback_type # This is the feedback type that is used in the current step.
        assert self.instruction_type in self.INSTRUCTION_TYPES
        assert self.feedback_type in self.FEEDBACK_TYPES
        self._feedback_types = list(self.FEEDBACK_TYPES)
        self._feedback_types.remove('m')
        self.paraphrase_idx = paraphrase_idx

    def format(self, prompts : List[str], **kwargs):
        """ A helper method for selecting from a set of paraphrased prompts."""
        if self.paraphrase_idx is None:
            return np.random.choice(prompts).format(**kwargs)
        else:
            prompts[self.paraphrase_idx % len(prompts)].format(**kwargs)

    def obs_check(self, observation: Dict[str, Any]):
        """ This is a sanity check for the observation dict."""
        assert isinstance(observation, dict), "The observation must be a dict."
        assert 'observation' in observation and 'feedback' in observation and 'instruction' in observation, \
               "The observation must be a dict with keys: observation, feedback, instruction."

    def reset(self) -> Dict[str, str]:
        """ Reset the environment and return the initial observation."""
        observation = self._reset()
        if type(observation)==str:  # backward compatibility
            observation = dict(instruction=observation, observation=None, feedback=None)
        self.obs_check(observation)
        assert observation['feedback'] is None, "The feedback must be None in the initial observation."
        assert observation['instruction'] is not None, "The instruction must be provided in the initial observation."
        return observation

    def _reset(self) -> Union[str, Dict[str, str]]:
        """ Implement this in the subclass. """
        raise NotImplementedError

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """ Step the environment and return the observation, reward, terminal, and info."""
        self._feedback_type = np.random.choice(self._feedback_types) if self.feedback_type=='m' else self.feedback_type
        observation, reward, terminal, info = self._step(action)
        if type(observation)==str:  # backward compatibility
            observation = dict(instruction=None, observation=observation, feedback=f"You received a reward of {reward}.")
        self.obs_check(observation)
        return observation, reward, terminal, info

    def _step(self, action: Any) -> Tuple[Union[str, Dict[str, Any]], float, bool, Dict[str, Any]]:
        """ Implement this in the subclass.
            Use self._feedback_type to determine the feedback.
        """
        raise NotImplementedError
