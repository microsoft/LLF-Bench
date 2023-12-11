import gymnasium as gym
import llfbench
import numpy as np
import random
from tqdm import tqdm
from llfbench.utils.utils import generate_combinations_dict
from llfbench.envs.llf_env import LLFWrapper


def obs_space_contains_obs(obs, obs_space):
    for k in obs:
        if isinstance(obs[k], str):
            if len(obs)==0:
                return True
        return obs_space[k].contains(obs[k]) or (obs[k] is None)

def step_env(env_name, seed, config):
    random.seed(seed)
    np.random.seed(seed)

    env = llfbench.make(env_name, **config)
    assert len(env.reward_range)==2
    obs, info = env.reset(seed=seed)

    if isinstance(env.action_space , gym.spaces.Text):
        action = 'test action'
        if 'llf-optimization' in env_name:
            action = 'x = [1.0, 2.0]'
        elif 'llf-rec-MovieRec' in env_name:
            action = """[{"title": "John Wick"}]"""
    else:
        env.action_space.seed(seed)
        action = env.action_space.sample()

    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert env.action_space.contains(action)
    assert obs_space_contains_obs(obs, env.observation_space)
    assert obs_space_contains_obs(next_obs, env.observation_space)
    assert type(terminated) == bool
    assert type(truncated) == bool
    assert type(reward)==float or type(reward)==int
    assert type(info)==dict and type(next_info) == dict

    assert obs['instruction'] is not None
    assert obs['feedback'] is None
    assert next_obs['instruction'] is None
    assert not info['success']
    assert 'success' in next_info
    assert reward>= env.reward_range[0] and reward<= env.reward_range[1]

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

def test_wrapper(env):
    if isinstance(env,LLFWrapper):
        return True
    elif hasattr(env, 'env'):
        return test_wrapper(env.env)
    else:
        return False

def test_env(env_name, seed=0):
    print(env_name)
    instruction_types, feedback_types = llfbench.supported_types(env_name)
    feedback_types = list(feedback_types) + ['n', 'a', 'm']
    configs = generate_combinations_dict(dict(instruction_type=instruction_types, feedback_type=feedback_types))
    for config in configs:
        env = llfbench.make(env_name, **config)  # test llfbench.make
        assert test_wrapper(env) # test LLFWrapper is used
        ouputs1 = step_env(env_name, seed, config=config)
        ouputs2 = step_env(env_name, seed, config=config)
        check_equivalence(ouputs1, ouputs2)

def test_benchmark(benchmark_prefix):
    all_envs = []
    for env_name in gym.envs.registry:
        if benchmark_prefix in env_name:
            all_envs.append(env_name)
    print(f'Number of {benchmark_prefix} environments: ', len(all_envs))
    for env_name in tqdm(all_envs):
        test_env(env_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark_prefix', type=str, default='llf-bandits')
    test_benchmark(**vars(parser.parse_args()))