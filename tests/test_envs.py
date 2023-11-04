import gymnasium as gym
import verbal_gym
import numpy as np
import random


def step_env(env_name, seed):
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    obs, info = env.reset(seed=seed)

    if isinstance(env.action_space , gym.spaces.Text):
        action = 'test action'
        if 'verbal-optimization' in env_name:
            action = 'x = [-4.0, 5.0]'
    else:
        env.action_space.seed(seed)
        action = env.action_space.sample()

    next_obs, reward, terminated, truncated, next_info = env.step(action)
    return dict(obs=obs,
                action=action,
                info=info,
                next_obs=next_obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                next_info=next_info)


def check_equivalence(nested1, nested2, name=None):
    """ Check whether two nested structures are equivalent."""
    assert type(nested1) == type(nested2)
    if isinstance(nested1, dict):
        assert nested1.keys() == nested2.keys()
        for k in nested1.keys():
            check_equivalence(nested1[k], nested2[k], name=k)
    elif isinstance(nested1, list) or isinstance(nested1, tuple):
        assert len(nested1) == len(nested2)
        for i in range(len(nested1)):
            check_equivalence(nested1[i], nested2[i])
    else:
        if isinstance(nested1, np.ndarray):
            assert np.array_equal(nested1, nested2), (name, nested1, nested2)
        else:
            assert nested1 == nested2, (name, nested1, nested2)

def test_env(env_name, seed=0):

    # env_name = 'verbal-box-close-b-n-v2'
    # env_name = 'verbal-drawer-open-b-n-v2'

    # if 'Gridworld' in env_name or 'verbal-optimization' in env_name or 'verbal-metaworld' in env_name or 'MovieRec' in env_name or 'verbal-poem' in env_name:
    #     return

    print(env_name)
    env = gym.make(env_name)
    ouputs1 = step_env(env_name, seed)
    ouputs2 = step_env(env_name, seed)
    check_equivalence(ouputs1, ouputs2)


all_envs = []
for env_name in gym.envs.registry:
    if 'verbal-' in env_name:
        all_envs.append(env_name)

print('Number of verbal-gym environments: ', len(all_envs))

for env_name in all_envs:
    test_env(env_name)