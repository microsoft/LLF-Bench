import numpy as np

def format(prompts, **kwargs):
    return np.random.choice(prompts).format(**kwargs)