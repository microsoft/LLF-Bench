from verbal_gym.envs.env_wrappers import TerminalFreeWrapper, EnvCompatibility
from verbal_gym.envs.verbal_gym_env import VerbalGymWrapper, Feedback
from verbal_gym.envs.poem_env.formal_poems import Haiku, Tanka, LineSyllableConstrainedPoem, SyllableConstrainedPoem
from verbal_gym.envs.poem_env.prompts import *

# class PoemGymWrapper(VerbalGymWrapper):
#
#     INSTRUCTION_TYPES = ('b') #, 'p', 'c')
#     FEEDBACK_TYPES = ('m', 'r', 'hn', 'fp')
#
#     def __init__(self, env, instruction_type, feedback_type):
#         super().__init__(TerminalFreeWrapper(EnvCompatibility(env)), instruction_type, feedback_type)
#         self._feedback_type_table = {'r':0, 'hn':0.5, 'fp':1}
#
#     def _reset(self, *, seed=None, options=None):  # TODO types of instructions
#         instruction, info = self.env.reset(seed=seed, options=options)
#         if type(self._poem_env) == Haiku:
#             instruction = self.reformat(instruction, haiku_b_instruction)
#         elif type(self._poem_env) == Tanka:
#             instruction = self.reformat(instruction, tanka_b_instruction)
#         elif type(self._poem_env) == LineSyllableConstrainedPoem:
#             instruction = self.reformat(instruction, line_syllable_constrained_poem_b_instruction)
#         elif type(self._poem_env) == SyllableConstrainedPoem:
#             instruction = self.reformat(instruction, syllable_constrained_poem_b_instruction)
#         return dict(instruction=instruction, observation=None, feedback=None), info
#
#     def _step(self, action):
#         self._poem_env.feedback = self._feedback_type_table[self._feedback_type]
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         feedback = info['feedback']
#         del info['feedback']
#         # Use reformat to replace some patterns
#         if self._poem_env.feedback in (0, 0.5, 1.0):
#             feedback = self.reformat(feedback, r_feedback_pos)
#             feedback = self.reformat(feedback, r_feedback_neg)
#         if self._poem_env.feedback in (0.5, 1.0):
#             feedback = self.reformat(feedback, hn_feedback)
#         if self._poem_env.feedback in (1.0,):
#             feedback = self.reformat(feedback, fp_feedback)  # line_number_incorrect
#             feedback = self.reformat(feedback, line_fp_feedback_1)  # produce_line_feedback
#             feedback = self.reformat(feedback, line_fp_feedback_2)  # produce_line_feedback
#         observation = dict(instruction=None, observation=None, feedback=feedback)
#         return observation, reward, terminated, truncated, info
#
#     @property
#     def _poem_env(self):
#         return self.env.env.env

class PoemGymWrapper(VerbalGymWrapper):

    INSTRUCTION_TYPES = ('b') #, 'p', 'c')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp', 'fn')

    def __init__(self, env, instruction_type, feedback_type):
        super().__init__(TerminalFreeWrapper(EnvCompatibility(env)), instruction_type, feedback_type)
        # self._feedback_type_table = {'r':0, 'hn':0.5, 'fp':1}

    def _reset(self, *, seed=None, options=None):  # TODO types of instructions
        instruction, info = self.env.reset(seed=seed, options=options)
        if type(self._poem_env) == Haiku:
            instruction = self.reformat(instruction, haiku_b_instruction)
        elif type(self._poem_env) == Tanka:
            instruction = self.reformat(instruction, tanka_b_instruction)
        elif type(self._poem_env) == LineSyllableConstrainedPoem:
            instruction = self.reformat(instruction, line_syllable_constrained_poem_b_instruction)
        elif type(self._poem_env) == SyllableConstrainedPoem:
            instruction = self.reformat(instruction, syllable_constrained_poem_b_instruction)
        return dict(instruction=instruction, observation=None, feedback=None), info

    def _step(self, action):
        # self._poem_env.feedback = self._feedback_type_table[self._feedback_type]
        observation, reward, terminated, truncated, info = self.env.step(action)
        didactic_feedback = info['feedback']
        del info['feedback']
        del info['original_feedback']

        paraphrased_feedback = Feedback()

        for feedback_type in self._feedback_type:
            feedback = didactic_feedback[feedback_type]
            if feedback_type == 'r':
                feedback = self.reformat(feedback, r_feedback_pos)
                feedback = self.reformat(feedback, r_feedback_neg)
            elif feedback_type == 'hn':
                feedback = self.reformat(feedback, line_number_hn_feedback)
                feedback = self.reformat(feedback, syllable_hn_feedback)
            elif feedback_type == 'hp':
                feedback = self.reformat(feedback, syllable_hp_feedback)
            elif feedback_type == 'fp':
                feedback = self.reformat(feedback, line_number_fp_feedback)
                feedback = self.reformat(feedback, syllable_fp_feedback_1)
                feedback = self.reformat(feedback, syllable_fp_feedback_2)
            elif feedback_type == 'fn':
                feedback = self.reformat(feedback, line_number_fn_feedback)
            else:
                raise ValueError(f'Unknown feedback type: {feedback_type}')
            paraphrased_feedback[feedback_type] = feedback

        observation = dict(instruction=None, observation=None, feedback=paraphrased_feedback)

        # Use reformat to replace some patterns
        # if self._poem_env.feedback in (0.5, 1.0):
        #     feedback = self.reformat(feedback, hn_feedback)
        # if self._poem_env.feedback in (1.0,):
        #     feedback = self.reformat(feedback, fp_feedback)  # line_number_incorrect
        #     feedback = self.reformat(feedback, line_fp_feedback_1)  # produce_line_feedback
        #     feedback = self.reformat(feedback, line_fp_feedback_2)  # produce_line_feedback
        # observation = dict(instruction=None, observation=None, feedback=feedback)

        return observation, reward, terminated, truncated, info

    @property
    def _poem_env(self):
        return self.env.env.env