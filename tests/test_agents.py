import os
import time
import pickle
import argparse
import logging

from llfbench.agents.basic_ai_agent import BasicAIAgent
from llfbench.agents.user_agent import UserAgent
from llfbench.agents.utils import set_seed, evaluate_agent, print_color
from llfbench.agents.llm import make_llm

import gymnasium
import llfbench as gym

def main(args):

    n_episodes = args.n_episodes
    horizon = args.horizon

    all_envs = []
    VISUAL = '-visual'

    for env_name in gymnasium.envs.registry:
        if ((args.env_name == 'all') and env_name.startswith('llf-')) or (env_name.startswith(args.env_name) and not env_name.endswith(VISUAL)):
            all_envs.append(tuple([env_name, False]))
        elif args.env_name.endswith(VISUAL) and env_name.startswith(args.env_name[:args.env_name.rfind(VISUAL)]):
            all_envs.append(tuple([env_name, True])) #env_name + VISUAL)
    print("Evaluating on the following environments:", all_envs)
    print("Number of environments:", len(all_envs))
    for env_name, visual in all_envs:
        # Create the environment
        env = gym.make(env_name,
                       instruction_type='b',
                       feedback_type='a' if not args.ignore_fp else ['r', 'hp', 'hn'], # ['r', 'hp', 'hn'], # ['fp', 'r'],
                       visual=visual)
        env = gym.envs.env_wrappers.TextWrapper(env)

        set_seed(args.seed, env)

        if args.agent == 'BasicAIAgent':
            # Basic agent
            system_prompt = BasicAIAgent.system_prompt
            llm = make_llm(args.model, system_prompt=system_prompt)
            gpt_agent = BasicAIAgent(llm, verbose=args.verbose)
        elif args.agent == 'UserAgent':
            # User agent
            gpt_agent = UserAgent(verbose=args.verbose)

        scores, data = evaluate_agent(agent=gpt_agent,
                                env=env,
                                horizon=horizon,
                                n_episodes=n_episodes,
                                n_workers=args.n_workers,
                                seed=args.seed,
                                log_data=True,
                                sparse_reward=args.sparse_reward)

        print_color('Agent: mean score {:.2f}, std {:.2f}'.format(scores.mean(), scores.std()),
                    color='red')

        # Save a small file with argparse values, so that we dont have to parse the text log
        results = {
            "setting": {k: v for k, v in vars(args).items()},
            "scores": scores,
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            'data': data,
        }

        with open(f"{args.save_path}/results.pkl", "wb") as f:
            pickle.dump(results, f)


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--save_path', type=str, default="results")
    parser.add_argument('--logname', type=str, default="log.txt")
    parser.add_argument('--horizon', type=int, default=1000)
    parser.add_argument('--env_name', type=str, default='all')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--agent', type=str, default='BasicAIAgent', choices=['BasicAIAgent', 'UserAgent'])
    parser.add_argument('--model', type=str, default='gpt-35-turbo')
    parser.add_argument("--sparse_reward", action="store_true")
    parser.add_argument('--ignore_fp', action='store_true')

    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    exp = f"experiment_env_{args.env_name}_" \
          f"episodes_{args.n_episodes}_horizon_{args.horizon}_model_{args.model.replace(':', '_')}"

    exp_folder = f"{args.save_path}/{exp}"

    if os.path.exists(exp_folder):
        exp_folder = exp_folder + f"{int(time.time())}"

    args.save_path = exp_folder

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    main(args)
