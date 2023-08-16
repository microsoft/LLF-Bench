import numpy as np
import random

def rollout(agent, env, *, horizon, log_data=False):
    """ A basic agent evaluation loop. """
    docstring = env.docstring
    agent.reset(docstring)

    observation = env.reset()
    info = {}
    sum_of_rewards = 0.0
    data = dict(observations=[observation], actions=[], rewards=[], dones=[], infos=[])
    for i in range(horizon):
        feedback = info.get('feedback', None)
        action = agent.act(observation, feedback)
        observation, reward, done, info = env.step(action)
        if log_data:
            for k in data.keys():
                data[k].append(locals(k))
        sum_of_rewards += reward
    return sum_of_rewards, data

def evaluate_agent(agent, env, *, horizon, n_episodes, n_workers=1):
    """ Evaluate an agent with n_episodes rollouts. """
    if n_workers > 1:
        import ray
        ray_rollout = ray.remote(lambda: rollout(agent, env, horizon=horizon)[0])
        scores = [ray_rollout.remote() for _ in range(n_episodes)]
        scores = ray.get(scores)
    else:
        scores = [rollout(agent, env, horizon=horizon)[0] for _ in range(n_episodes)]
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