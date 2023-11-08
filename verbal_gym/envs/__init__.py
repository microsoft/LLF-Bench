from verbal_gym.envs import gridworld
from verbal_gym.envs import bandits
from verbal_gym.envs import loss_landscape
from verbal_gym.envs import movie_rec
from verbal_gym.envs import poem
from verbal_gym.envs import highway
#from verbal_gym.envs import alfworld   # Commenting until we add the install instructions for AlfWorld

import importlib
if importlib.util.find_spec('metaworld'):
    from verbal_gym.envs import metaworld