from verbal_gym.envs.env_wrappers import TerminalFreeWrapper
from verbal_gym.envs.verbal_gym_env import VerbalGymWrapper

class PoemGymWrapper(VerbalGymWrapper):

    INSTRUCTION_TYPES = ('b') #, 'p', 'c')
    FEEDBACK_TYPES = ('m', 'r', 'hn', 'fp')

    def __init__(self, env, instruction_type, feedback_type, paraphrase_idx=None):
        super().__init__(TerminalFreeWrapper(env), instruction_type, feedback_type, paraphrase_idx=paraphrase_idx)
        self._feedback_type_table = {'r':0, 'hn':0.5, 'fp':1}

    def _reset(self):  # TODO types of instructions
        instruction = self.env.reset()
        return dict(instruction=instruction, observation=None, feedback=None)

    def _step(self, action):
        self._poem_env.feedback = self._feedback_type_table[self._feedback_type]
        observation, reward, terminal, info = self.env.step(action)
        observation = dict(instruction=None, observation=None, feedback=info['feedback'])
        del info['feedback']
        return observation, reward, terminal, info

    @property
    def _poem_env(self):
        return self.env.env