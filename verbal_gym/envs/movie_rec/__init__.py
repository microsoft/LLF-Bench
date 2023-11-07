from gymnasium.envs.registration import register
from verbal_gym.utils import generate_combinations_dict
from verbal_gym.envs.movie_rec.wrapper import MovieRecGymWrapper

environments = [
    'MovieRec'
]

def make_env(env_name,
             instruction_type='b',
             feedback_type='r',
             **kwargs):
    """ Make the original env and wrap it with the VerbalGymWrapper. """
    import importlib
    MovieCls = getattr(importlib.import_module("verbal_gym.envs.movie_rec.movie_rec"), env_name)
    env = MovieCls(**kwargs)  # `feedback` doesn't matter here, as we will override it.
    return MovieRecGymWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type)

for environment in environments:
    register(
        id=f'verbal-{environment}-v0',
        entry_point=f'verbal_gym.envs.movie_rec.movie_rec:{environment}',
    )

configs = generate_combinations_dict(
                dict(env_name=environments,
                     feedback_type=MovieRecGymWrapper.FEEDBACK_TYPES,
                     instruction_type=MovieRecGymWrapper.INSTRUCTION_TYPES))

for config in configs:
    register(
        id=f"verbal-optimization-{config['env_name']}-{config['instruction_type']}-{config['feedback_type']}-v0",
        entry_point='verbal_gym.envs.movie_rec:make_env',
        kwargs=config,
    )