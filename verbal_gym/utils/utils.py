import numpy as np
import random

def rollout(agent, env, horizon):
    """ A basic agent evaluation loop. """
    docstring = env.docstring
    agent.reset(docstring)

    obs = env.reset()
    info = {}
    sum_of_rewards = 0.0
    for i in range(horizon):
        feedback = info.get('feedback', None)
        action = agent.act(obs, feedback)
        observation, reward, done, info = env.step(action)
        sum_of_rewards += reward
    return sum_of_rewards

def evaluate_agent(agent, env, horizon, n_episodes):
    """ Evaluate an agent with n_episodes rollouts. """
    scores = []
    for i in range(n_episodes):
        scores.append(rollout(agent, env, horizon=horizon))
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