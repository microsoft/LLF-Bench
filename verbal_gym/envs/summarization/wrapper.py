from verbal_gym.envs.env_wrappers import TerminalFreeWrapper
from verbal_gym.envs.verbal_gym_env import VerbalGymWrapper

class SummaryEnvWrapper(VerbalGymWrapper):

    INSTRUCTION_TYPES = ('b', 'p', 'c')
    FEEDBACK_TYPES = ('m', 'r', 'hn', 'hp', 'fp', 'fn')

    def __init__(self, env, instruction_type, feedback_type):
        super().__init__(TerminalFreeWrapper(env), instruction_type, feedback_type)

    def _reset(self):  # TODO types of instructions
        instruction = self.env.reset()
        return dict(instruction=instruction, observation=None, feedback=None)

    def _step(self, action): # TODO types of feedback
        observation, reward, terminal, info = self.env.step(action)
        feedback = info['feedback']
        del info['feedback']
        observation = dict(instruction=None, observation=None, feedback=feedback)
        return observation, reward, terminal, info