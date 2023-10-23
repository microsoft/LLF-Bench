import random
import numpy as np
from verbal_gym.utils.loggers import ListLogger


def rollout(agent, env, *, horizon, oracle=False, get_logger=None):
    """ A generic agent evaluation loop. """

    # If logger is not provided, then use a default logger
    get_logger = get_logger or ListLogger
    logger = get_logger()

    observation = env.reset()  # this is a dict with keys: observation, feedback, instruction
    info = {}
    sum_of_rewards = 0.0
    for i in range(horizon):

        if oracle:  # XXX Oracle: the agent gets privileged information
            observation['oracle_info'] = info.get('oracle_info')

        action = agent.act(observation)
        new_observation, reward, done, info = env.step(action)
        logger.log(observation=observation, action=action, reward=reward, done=done, info=info)

        sum_of_rewards += reward
        observation = new_observation
        if done:
            break

    return sum_of_rewards, logger.content


def evaluate_agent(agent, env, *, horizon, n_episodes, oracle=False, get_logger=None, n_workers=1):
    """ Evaluate an agent with n_episodes rollouts. """

    _rollout = lambda: rollout(agent, env,
                               horizon=horizon,
                               oracle=oracle,
                               get_logger=get_logger)

    if n_workers > 1:
        import ray
        ray_rollout = ray.remote(_rollout)
        results = [ray_rollout.remote() for _ in range(n_episodes)]
        results = ray.get(results)
    else:
        results = [_rollout() for _ in range(n_episodes)]

    # Extract the scores and data
    scores = [score for score, _ in results]
    scores = np.array(scores)
    data = [data for _, data in results]
    return (scores, data)


def set_seed(seed, env=None):
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)
