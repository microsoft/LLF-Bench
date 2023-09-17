import os
import gym
import time
import pickle
import argparse
import logging

from verbal_gym.llm import make_llm
from verbal_gym.agents.basic_agent import BasicAgent
from verbal_gym.utils.utils import set_seed
from verbal_gym.utils.utils import evaluate_agent
from verbal_gym.utils.misc_utils import print_color
from verbal_gym.utils.multiprocess_logger import MultiprocessingLoggerManager


def main(args):

    n_episodes = args.n_episodes
    horizon = args.horizon

    # Create the environment
    env = gym.make(args.env_name,
                   num_rooms=args.num_rooms,
                   horizon=args.horizon,
                   fixed=True,
                   feedback_level=args.feedback_type,
                   min_goal_dist=4)

    set_seed(args.seed, env)

    n_actions = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else None
    action_name = 'Action'
    if any([name in args.env_name for name in ('Haiku', 'Tanka', 'LineSyllableConstrainedPoem', 'SyllableConstrainedPoem')]):
        action_name = 'Poem'

    log_manager = MultiprocessingLoggerManager(file_path=f"{args.save_path}/{args.logname}",
                                               logging_level=logging.INFO)
    logger = log_manager.get_logger("Main")

    # Basic agent
    system_prompt = BasicAgent.system_prompt
    llm = make_llm(args.model, system_prompt=system_prompt)
    gpt_agent = BasicAgent(llm, n_actions, verbose=args.verbose, action_name=action_name)

    scores = evaluate_agent(agent=gpt_agent,
                            env=env,
                            horizon=horizon,
                            n_episodes=n_episodes,
                            n_workers=args.n_workers,
                            logger=logger)

    print_color('Basic LLM agent: mean score {:.2f}, std {:.2f}'.format(scores.mean(), scores.std()),
                color='red',
                logger=logger)

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
    parser.add_argument('--logname', type=str, default="basic_agent_log.txt")
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--env_name', type=str, default='verbal-BanditTenArmedRandomRandom-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--model', type=str, default='azure:gpt-35-turbo')

    # Gridworld experiment only
    parser.add_argument('--num_rooms', type=int, default=20)
    parser.add_argument('--feedback_type', type=str, default="gold", choices=["bandit", "gold", "oracle"])

    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    exp = f"experiment_env_{args.env_name}_num_rooms_{args.num_rooms}_feedback_{args.feedback_type}_" \
          f"episodes_{args.n_episodes}_horizon_{args.horizon}_model_{args.model}"

    exp_folder = f"{args.save_path}/{exp}"

    if os.path.exists(exp_folder):
        exp_folder = exp_folder + f"{int(time.time())}"

    args.save_path = exp_folder

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    main(args)
