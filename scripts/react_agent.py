import gym
from verbal_gym.llm import make_llm
from verbal_gym.agents.react_agent import ReActAgent
from verbal_gym.utils.utils import evaluate_agent, set_seed
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

    # React agent
    gpt_agent = ReActAgent(
        make_llm(args.model, system_prompt=ReActAgent.system_prompt, temperature=args.temperature),
        n_actions,
        action_name=action_name,
        verbose=args.verbose)

    scores = evaluate_agent(gpt_agent, env, horizon=horizon, n_episodes=n_episodes, n_workers=args.n_workers)
    scores = scores[0]
    print_color('Reflexion agent: mean score {:.2f}, std {:.2f}'.format(scores.mean(), scores.std()), 'red')
    return scores


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--env_name',type=str, default='verbal-SyllableConstrainedPoem-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--model', type=str, default='azure:gpt-35-turbo')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
