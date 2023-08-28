import gym
import verbal_gym
from verbal_gym.llm.gpt_models import GPT
from verbal_gym.envs.env_wrapper import FullInformationWrapper
from verbal_gym.utils.utils import evaluate_agent, set_seed
from verbal_gym.utils.misc_utils import print_color
import yaml


def create_agent(agent_name, env, *, agent_config=None, verbose=False):

    # Define action spec
    n_actions = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else None
    action_name = 'Action'
    if any([name in env.unwrapped.spec.id for name in ('Haiku', 'Tanka', 'LineSyllableConstrainedPoem', 'SyllableConstrainedPoem')]):
        action_name = 'Poem'

    # Create agent
    if agent_name=='posterior_agent':
        from verbal_gym.agents.posterior_agents import PosteriorAgent, ParaphraseAgent
        paraphrase_agent = None if not agent_config['paraphrase'] else \
                           ParaphraseAgent(GPT(ParaphraseAgent.system_prompt, temperature=agent_config['paraphrase_temperature'], model=agent_config['paraphrase_model']))
        agent = PosteriorAgent(GPT(PosteriorAgent.system_prompt, temperature=agent_config['temperature'], model=agent_config['model']),
                                   n_actions,
                                   action_name=action_name,
                                   verbose=verbose,
                                   permute_history=agent_config['permute_history'],
                                   paraphrase_at_given=agent_config['paraphrase_at_given'],
                                   paraphrase_agent=paraphrase_agent)
    elif agent_name=='basic_agent':
        from verbal_gym.agents.basic_agent import BasicAgent
        agent = BasicAgent(GPT(BasicAgent.system_prompt, model=agent_config['model']),
                               n_actions, verbose=verbose, action_name=action_name)
    elif agent_name=='random_agent':
        from verbal_gym.agents.basic_agent import RandomAgent
        assert n_actions is not None
        agent = RandomAgent(n_actions)
    elif agent_name=='full_info_agent':
        from verbal_gym.envs.env_wrapper import FullInformationWrapper
        from verbal_gym.agents.basic_agent import FullInformationAgent
        # Full information agent
        agent = FullInformationAgent(GPT(FullInformationAgent.system_prompt, model=agent_config['model']),
                                        n_actions=n_actions,
                                        verbose=verbose)
    else:
        raise NotImplementedError
    return agent

def main(args):

    # Create the environment
    env = gym.make(args.env_name)
    set_seed(args.seed, env)
    if 'full_info' in args.agent_name:
        env = FullInformationWrapper(env)

    # Create the agent
    if isinstance(args.agent_config, str):   # Load agent config from yaml
        args.agent_config = yaml.safe_load(open(args.agent_config, 'r'))
        assert args.agent_name == args.agent_config['agent_name']
    agent = create_agent(args.agent_name, env, agent_config=args.agent_config, verbose=args.verbose)

    # Evaluate the agent!
    scores = evaluate_agent(agent, env,
                            horizon=args.horizon,
                            n_episodes=args.n_episodes,
                            n_workers=args.n_workers,
                            return_full_information='full_info' in args.agent_name)

    print_color(f'{args.agent_name}: mean score {scores.mean():.2f}, std {scores.std():.2f}', 'red')

    return scores


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--env_name',type=str, default='verbal-BanditTenArmedRandomRandom-v0')
    parser.add_argument('--agent_name',type=str, default='random_agent')
    parser.add_argument('--agent_config',type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
