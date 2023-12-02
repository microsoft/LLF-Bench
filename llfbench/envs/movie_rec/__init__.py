from gymnasium.envs.registration import register
from llfbench.utils import generate_combinations_dict
from llfbench.envs.movie_rec.wrapper import MovieRecGymWrapper

environments = [
    'MovieRec'
]

def make_env(env_name,
             instruction_type='b',
             feedback_type='r',
             **kwargs):
    """ Make the original env and wrap it with the LLFWrapper. """
    import importlib
    MovieCls = getattr(importlib.import_module("llfbench.envs.movie_rec.movie_rec"), env_name)
    env = MovieCls(**kwargs)  # `feedback` doesn't matter here, as we will override it.
    return MovieRecGymWrapper(env, instruction_type=instruction_type, feedback_type=feedback_type)

register(
    id=f"llf-rec-{environments[0]}-v0",
    entry_point='llfbench.envs.movie_rec:make_env',
    kwargs=dict(env_name=environments[0], feedback_type='a', instruction_type='b')
)