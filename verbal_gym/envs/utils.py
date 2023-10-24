import numpy as np

def format(prompts, _idx=None, **kwargs):
    if _idx is None:
        return np.random.choice(prompts).format(**kwargs)
    else:
        return prompts[_idx % len(prompts)].format(**kwargs)