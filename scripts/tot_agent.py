import gym
import logging

from verbal_gym.llm import make_llm
from verbal_gym.agents.tot_agent import ToTAgent, VoterAgent, ThinkerAgent
from verbal_gym.agents.posterior_agents import ParaphraseAgent
from verbal_gym.utils.utils import evaluate_agent, set_seed
from verbal_gym.utils.misc_utils import print_color
from verbal_gym.utils.multiprocess_logger import MultiprocessingLoggerManager


def main(args, logger):

    n_episodes = args.n_episodes
    horizon = args.horizon

    # Create the environment
    env = gym.make(args.env_name)
    set_seed(args.seed, env)

    n_actions = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else None
    action_name = 'Action'
    if any([name in args.env_name for name in ('Haiku', 'Tanka', 'LineSyllableConstrainedPoem', 'SyllableConstrainedPoem')]):
        action_name = 'Poem'

    # TODO should save the stdout

    # Posterior agent
    paraphrase_agent = None if args.not_paraphrase else ParaphraseAgent(make_llm(args.model, system_prompt=ParaphraseAgent.system_prompt, temperature=args.temperature))
    voter_agent = VoterAgent(make_llm(args.model, system_prompt=VoterAgent.system_prompt, temperature=args.temperature),
                               action_name=action_name)

    thinker_agent = ThinkerAgent(make_llm(args.model, system_prompt=ThinkerAgent.system_prompt, temperature=args.temperature),
                               action_name=action_name)

    tot_agent = ToTAgent(make_llm(args.model, system_prompt=VoterAgent.system_prompt, temperature=args.temperature),
                               n_actions,
                               action_name=action_name,
                               voter_agent=voter_agent,
                               thinker_agent=thinker_agent,
                               verbose=args.verbose,
                               permute_history=not args.not_permute_history,
                               paraphrase_agent=paraphrase_agent,
                               logger=logger,
                               buffer_size=args.buffer_size,
                               max_iter=args.max_iter
                               )
    scores = evaluate_agent(tot_agent, env, horizon=horizon, n_episodes=n_episodes, n_workers=args.n_workers)
    print_color('ToT agent: mean score {:.2f}, std {:.2f}'.format(scores.mean(), scores.std()), 'red')
    return scores



def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--env_name',type=str, default='verbal-BanditTenArmedRandomRandom-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--not_paraphrase', action='store_true')
    parser.add_argument('--not_permute_history', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--model', type=str, default='azure:gpt-35-turbo')
    parser.add_argument("--logfile", type=str, default="./results.txt")
    parser.add_argument('--buffer_size', type=int, default=5)
    parser.add_argument('--max_iter', type=int, default=4)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # Create a logger
    log_manager = MultiprocessingLoggerManager(file_path=args.logfile,
                                               logging_level=logging.INFO)
    logger = log_manager.get_logger("Main")

    main(args=args, logger=logger)
