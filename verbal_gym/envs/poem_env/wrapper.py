from verbal_gym.envs.env_wrappers import TerminalFreeWrapper
from verbal_gym.envs.verbal_gym_env import VerbalGymWrapper
from verbal_gym.envs.poem_env.formal_poems import Haiku, Tanka, LineSyllableConstrainedPoem, SyllableConstrainedPoem
from verbal_gym.envs.poem_env.prompts import *

class PoemGymWrapper(VerbalGymWrapper):

    INSTRUCTION_TYPES = ('b') #, 'p', 'c')
    FEEDBACK_TYPES = ('m', 'r', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type, paraphrase_idx=None):
        super().__init__(TerminalFreeWrapper(env), instruction_type, feedback_type, paraphrase_idx=paraphrase_idx)
        self._feedback_type_table = {'r':0, 'hn':0.5, 'fp':1}

    def _reset(self):  # TODO types of instructions
        instruction = self.env.reset()
        if type(self._poem_env) == Haiku:
            instruction = self.reformat(instruction, haiku_b_instruction)
        elif type(self._poem_env) == Tanka:
            instruction = self.reformat(instruction, tanka_b_instruction)
        elif type(self._poem_env) == LineSyllableConstrainedPoem:
            instruction = self.reformat(instruction, line_syllable_constrained_poem_b_instruction)
        elif type(self._poem_env) == SyllableConstrainedPoem:
            instruction = self.reformat(instruction, syllable_constrained_poem_b_instruction)
        return dict(instruction=instruction, observation=None, feedback=None)

    def _step(self, action):
        self._poem_env.feedback = self._feedback_type_table[self._feedback_type]
        observation, reward, terminal, info = self.env.step(action)
        feedback = info['feedback']
        del info['feedback']
        # Use reformat to replace some patterns
        if self._poem_env.feedback in (0, 0.5, 1.0):
            feedback = self.reformat(feedback, r_feedback_pos)
            feedback = self.reformat(feedback, r_feedback_neg)
        if self._poem_env.feedback in (0.5, 1.0):
            feedback = self.reformat(feedback, hn_feedback)
        if self._poem_env.feedback in (1.0,):
            feedback = self.reformat(feedback, fp_feedback)  # line_number_incorrect
            feedback = self.reformat(feedback, line_fp_feedback_1)  # produce_line_feedback
            feedback = self.reformat(feedback, line_fp_feedback_2)  # produce_line_feedback
        observation = dict(instruction=None, observation=None, feedback=feedback)
        return observation, reward, terminal, info

    @property
    def _poem_env(self):
        return self.env.env