import os
import time
import pickle
import argparse
import logging

from user_agent import UserAgent
from utils import set_seed, evaluate_agent, print_color

import verbal_gym as gym
import gymnasium

def main(args):

    n_episodes = args.n_episodes
    horizon = args.horizon

    all_envs = []
    for env_name in gymnasium.envs.registry:
        if ((args.env_name == 'all') and env_name.startswith('verbal-')) or env_name.startswith(args.env_name):
            all_envs.append(env_name)
    print("Evaluating on the following environments:", all_envs)    
    for env_name in all_envs:
        # Create the environment
        env = gym.make(env_name)
        env = gym.envs.env_wrappers.TextWrapper(env)

        set_seed(args.seed, env)

        agent = None
        if args.agent == 'UserAgent':
            # User agent
            agent = UserAgent(verbose=args.verbose)

        scores = evaluate_agent(agent=agent,
                                env=env,
                                horizon=horizon,
                                n_episodes=n_episodes,
                                n_workers=1,
                                seed=args.seed)

        print_color('Agent: mean score {:.2f}, std {:.2f}'.format(scores.mean(), scores.std()),
                    color='red')

        # Save a small file with argparse values, so that we dont have to parse the text log
        results = {
            "setting": {k: v for k, v in vars(args).items()},
            "scores": scores,
            "mean_score": scores.mean(),
            "std_score": scores.std()
        }

        with open(f"{args.save_path}/results.pkl", "wb") as f:
            pickle.dump(results, f)


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--save_path', type=str, default="results")
    parser.add_argument('--logname', type=str, default="log.txt")
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--env_name', type=str, default='all')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--agent', type=str, default='UserAgent', choices=['UserAgent'])
    
    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    exp = f"experiment_env_{args.env_name}_" \
          f"episodes_{args.n_episodes}_horizon_{args.horizon}"

    exp_folder = f"{args.save_path}/{exp}"

    if os.path.exists(exp_folder):
        exp_folder = exp_folder + f"{int(time.time())}"

    args.save_path = exp_folder

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    main(args)
