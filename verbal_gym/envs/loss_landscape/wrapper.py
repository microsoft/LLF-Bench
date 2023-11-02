from verbal_gym.envs.env_wrappers import TerminalFreeWrapper, EnvCompatibility
from verbal_gym.envs.verbal_gym_env import VerbalGymWrapper
# from verbal_gym.envs.loss_landscape.loss_descent import
from verbal_gym.envs.loss_landscape.prompts import *

"""
The original env produces support for both
- Directional feedback: 0,0.5,1
- Didactic feedback: 'r', 'hp', 'hn', 'fp', 'fn'

This wrapper will only produce didactic feedback
"""


class LossLandscapeGymWrapper(VerbalGymWrapper):
    INSTRUCTION_TYPES = ('b')  # , 'p', 'c')
    FEEDBACK_TYPES = ('m', 'r', 'hp', 'hn', 'fp', 'fn')

    def __init__(self, env, instruction_type, feedback_type):
        super().__init__(TerminalFreeWrapper(EnvCompatibility(env)), instruction_type, feedback_type)

    def _reset(self, *, seed=None, options=None):  # TODO types of instructions
        instruction = self._loss_env.docstring
        obs = self.env.reset(seed=seed, options=options)

        instruction = self.reformat(instruction, loss_b_instruction)
        return dict(instruction=instruction, observation=obs, feedback=None)

    def _step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        didactic_feedback = info['didactic_feedback']
        del info['feedback']
        del info['didactic_feedback']

        feedback = "No feedback available."
        if self._feedback_type == 'r':
            feedback = self.reformat(didactic_feedback[self._feedback_type], r_feedback_pos, template=r_feedback_pos_template)
            feedback = self.reformat(feedback, r_feedback_neg, template=r_feedback_neg_template)
        elif self._feedback_type in didactic_feedback:
            temp_dim1 = eval("{}_feedback_dim1".format(self._feedback_type))
            feedback = self.reformat(didactic_feedback[self._feedback_type],
                                     eval("{}_feedback_dim1".format(self._feedback_type)),
                                     template=temp_dim1)
            temp_dim2 = eval("{}_feedback_dim2".format(self._feedback_type))
            feedback = self.reformat(feedback,
                                     eval("{}_feedback_dim2".format(self._feedback_type)),
                                     template=temp_dim2)

        observation = dict(instruction=None, observation=observation, feedback=feedback)
        return observation, reward, terminated, truncated, info

    @property
    def _loss_env(self):
        return self.env.env.env
