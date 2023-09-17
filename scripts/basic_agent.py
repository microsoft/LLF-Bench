import gym
import argparse

from verbal_gym.llm import make_llm
from verbal_gym.agents.basic_agent import BasicAgent
from verbal_gym.utils.utils import set_seed
from verbal_gym.utils.utils import evaluate_agent
from verbal_gym.utils.misc_utils import print_color


def main(args):

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

    # Basic agent
    system_prompt = BasicAgent.system_prompt
    llm = make_llm(args.model, system_prompt=system_prompt)
    gpt_agent = BasicAgent(llm, n_actions, verbose=args.verbose, action_name=action_name)

    scores = evaluate_agent(gpt_agent, env, horizon=horizon, n_episodes=n_episodes, n_workers=args.n_workers)
    print_color('Basic LLM agent: mean score {:.2f}, std {:.2f}'.format(scores.mean(), scores.std()), 'red')


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--env_name',type=str, default='verbal-BanditTenArmedRandomRandom-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--model', type=str, default='azure:gpt-35-turbo')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
