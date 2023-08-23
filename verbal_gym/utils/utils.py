import numpy as np
import random

def rollout(agent, env, *, horizon, return_full_information=False, log_data=False):
    """ A basic agent evaluation loop. """
    if return_full_information:
        assert hasattr(env,'get_full_information')

    observation = env.reset()
    default_docstring = 'This is an interactive decision making problem with verbal feedback.'
    docstring = getattr(env, 'docstring', observation if isinstance(observation,str) else default_docstring)  # in case the environment does not have docstring
    agent.reset(docstring)

    info = {}
    sum_of_rewards = 0.0
    data = dict(observations=[observation], actions=[], rewards=[], dones=[], infos=[])
    for i in range(horizon):
        feedback = info.get('feedback', None)
        if return_full_information:  # Oracle: the agent gets privileged information
            full_information = info.get('full_information', env.get_full_information())
            action = agent.act(observation, feedback, full_information=full_information)
        else:  # regular agent
            action = agent.act(observation, feedback)
        observation, reward, done, info = env.step(action)
        if log_data:
            for k in data.keys():
                data[k].append(locals(k))
        sum_of_rewards += reward
    return sum_of_rewards, data

def evaluate_agent(agent, env, *, horizon, n_episodes, return_full_information=False, n_workers=1):
    """ Evaluate an agent with n_episodes rollouts. """
    _rollout = lambda: rollout(agent, env, horizon=horizon, return_full_information=return_full_information)[0]
    if n_workers > 1:
        import ray
        ray_rollout = ray.remote(_rollout)
        scores = [ray_rollout.remote() for _ in range(n_episodes)]
        scores = ray.get(scores)
    else:
        scores = [_rollout() for _ in range(n_episodes)]
    scores = np.array(scores)
    return scores

def set_seed(seed, env=None):
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)