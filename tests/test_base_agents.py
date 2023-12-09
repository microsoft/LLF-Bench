import random
import argparse
import llfbench
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from llfbench.envs.llf_env import LLFWrapper
from llfbench.utils.utils import generate_combinations_dict


def get_random_action(env, env_name, info):

    if isinstance(env.action_space, gym.spaces.Text):
        if "alfworld" in env_name.lower():
            action = random.choice(info["admissible_commands"])
        else:
            action = None
    else:
        action = env.action_space.sample()

    return action


def get_expert_action(env, env_name, info):

    if "expert_action" in info:
        action = info["expert_action"]
        if action is None:
            # IF action is None, it means that all expert actions are taken so just take random actions
            action = get_random_action(env, env_name, info)
    else:
        action = None

    return action


def obs_space_contains_obs(obs, obs_space):
    for k in obs:
        if isinstance(obs[k], str):
            if len(obs) == 0:
                return True
        return obs_space[k].contains(obs[k]) or (obs[k] is None)


def get_return(env, env_name, agent):

    assert len(env.reward_range) == 2

    obs, info = env.reset()
    total_return = 0.0
    completed = False

    while not completed:

        if agent == "random":
            action = get_random_action(env, env_name, info)
        elif agent == "expert":
            action = get_expert_action(env, env_name, info)
        else:
            raise AssertionError(f"Unhandled agent type {agent}. Can only handle random and expert agent.")

        if action is None:
            print(f"Cannot evaluate {agent} for {env_name}")
            return float("nan")

        next_obs, reward, terminated, truncated, next_info = env.step(action)

        total_return += reward
        completed = terminated or truncated

        assert env.action_space.contains(action)
        assert obs_space_contains_obs(obs, env.observation_space)
        assert obs_space_contains_obs(next_obs, env.observation_space)
        assert type(terminated) == bool
        assert type(truncated) == bool
        assert type(reward) == float or type(reward) == int
        assert type(info) == dict and type(next_info) == dict
        assert env.reward_range[0] <= reward <= env.reward_range[1]
        info = next_info

    return total_return


def test_wrapper(env):
    if isinstance(env, LLFWrapper):
        return True
    elif hasattr(env, 'env'):
        return test_wrapper(env.env)
    else:
        return False


def test_env(env_name, agent, num_eps=1, seed=0):

    instruction_types, feedback_types = llfbench.supported_types(env_name)
    feedback_types = list(feedback_types) + ['n', 'a', 'm']
    configs = generate_combinations_dict(dict(instruction_type=instruction_types, feedback_type=feedback_types))

    for config in configs:

        random.seed(seed)
        np.random.seed(seed)

        env = llfbench.make(env_name, **config)  # test llfbench.make
        assert test_wrapper(env)                 # test LLFWrapper is used

        all_returns = [get_return(env=env, env_name=env_name, agent=agent) for _ in range(num_eps)]

        print(f"Environment: {env_name}, Config {config}, Number of episodes {num_eps}, "
              f"Mean return {np.mean(all_returns):.3f}, "
              f"Std return {np.std(all_returns):.3f}, "
              f"Max return {np.max(all_returns):.3f}, "
              f"Min return {np.min(all_returns):.3f}")


def test_benchmark(benchmark_prefix, num_eps, agent):

    all_envs = []
    for env_name in gym.envs.registry:
        if benchmark_prefix in env_name:
            all_envs.append(env_name)

    print(f'Number of {benchmark_prefix} environments: ', len(all_envs))

    for env_name in tqdm(all_envs):
        test_env(env_name=env_name,
                 agent=agent,
                 num_eps=num_eps)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark_prefix', type=str, default='llf-bandits')
    parser.add_argument('num_eps', type=int, default=10, help="number of episodes")
    parser.add_argument('agent', type=str, default="random", help="type of agent: random or expert",
                        choices=["random", "expert"])
    test_benchmark(**vars(parser.parse_args()))
