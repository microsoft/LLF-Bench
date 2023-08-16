import gym
import numpy as np
from verbal_gym.llm.gpt_models import GPT
from verbal_gym.agents.agents import Agent, RandomAgent, BasicAgent
from verbal_gym.utils.utils import evaluate_agent, set_seed
from verbal_gym.utils.misc_utils import print_color


def main(args):

    n_episodes = args.n_episodes
    horizon = args.horizon

    system_prompt = """\
    You are an agent taked to solve an interactive problem with verbal feedback.\
    You will see "Problem Description" that tell you want to problem is about (such\
    as the goal of the task, the action space you should choose from, the rules, the\
    constraints, etc.)\
    After you choose an action, you will see the feedback from the environment.\
    You goal is to choose the right actions solve the task as fast as possible, according to "Problem Description".\
    """
    print_color('System: {}'.format(system_prompt), "blue")

    # Create the environment
    env = gym.make(args.env_name)
    set_seed(args.seed, env)

    assert isinstance(env.action_space, gym.spaces.Discrete)
    action_range = [env.action_space.start, env.action_space.start+env.action_space.n]

   # TODO should save the stdout
    llm = GPT(system_prompt)
    gpt_agent = BasicAgent(llm, action_range, verbose=True)
    scores = evaluate_agent(gpt_agent, env, horizon=horizon, n_episodes=n_episodes, n_workers=args.n_workers)
    print('Basic LLM agent: mean score {:.2f}, std {:.2f}'.format(scores.mean(), scores.std()))

    random_agent = RandomAgent(action_range[0], action_range[1]-1)
    scores = evaluate_agent(random_agent, env, horizon=horizon, n_episodes=n_episodes, n_workers=args.n_workers)
    print('Random agent: mean score {:.2f}, std {:.2f}'.format(scores.mean(), scores.std()))


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--env_name',type=str, default='verbal-BanditTenArmedRandomRandom-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=1)
    return parser

if __name__=='__main__':
    parser = get_parser()
    main(parser.parse_args())