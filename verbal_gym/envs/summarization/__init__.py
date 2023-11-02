from gymnasium.envs.registration import register
from verbal_gym.envs.summarization.pick_best_summary import PickBestSummary
from verbal_gym.envs.summarization.wrapper import SummaryEnvWrapper

def make_env(
             instruction_type='b',
             feedback_type='r',
             **kwargs):
    """ Make the original env and wrap it with the VerbalGymWrapper. """
    env = PickBestSummary(**kwargs)
    return SummaryEnvWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type)


register(
    id='verbal-PickBestSummary-v0',
    entry_point='verbal_gym.envs.summarization:make_env',
)
