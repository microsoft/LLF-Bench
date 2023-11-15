import importlib
from verbal_gym.envs import gridworld
from verbal_gym.envs import bandits
from verbal_gym.envs import optimization
from verbal_gym.envs import movie_rec
from verbal_gym.envs import poem
from verbal_gym.envs import highway

if importlib.util.find_spec('metaworld'):
    from verbal_gym.envs import metaworld

if importlib.util.find_spec('alfworld'):
    from verbal_gym.envs import alfworld
