import gym
import verbal_gym
from verbal_gym.llm.gpt_models import GPT
from verbal_gym.envs.env_wrapper import FullInformationWrapper
from verbal_gym.utils.utils import evaluate_agent, set_seed
from verbal_gym.utils.misc_utils import print_color
import yaml, copy, time, pickle
from itertools import product


def create_agent(agent_config, env, verbose=False):

    agent_name = agent_config['agent_name']

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
        from verbal_gym.agents.random_agent import RandomAgent
        assert n_actions is not None
        agent = RandomAgent(n_actions)
    elif agent_name=='full_info_agent':
        from verbal_gym.envs.env_wrapper import FullInformationWrapper
        from verbal_gym.agents.full_information_agent import FullInformationAgent
        assert n_actions is not None
        # Full information agent
        agent = FullInformationAgent(GPT(FullInformationAgent.system_prompt, model=agent_config['model']),
                                        n_actions=n_actions,
                                        verbose=verbose)
    else:
        raise NotImplementedError
    return agent

def run_experiment(agent_config, env_name, *, horizon, n_episodes, seed=0, verbose=False, n_workers=1, **kwargs):
    """ Run experiemnts for a single agent. """

    # agnet_config is a dict, which should contain the following keys:
    #   agent_name: name of the agent. If not provided, `file` must be provided.
    #   file: path to a yaml file that contains the default config
    #   other keys used to create the agent in `create_agent`.

    if 'file' in agent_config and 'name' not in agent_config:
        default_config = yaml.safe_load(open(agent_config['file'], 'r'))
        del agent_config['file']
        default_config.update(agent_config)
        agent_config.update(default_config)

    agent_name = agent_config['agent_name']
    # Create the environment
    env = gym.make(env_name)
    set_seed(seed, env)
    if 'full_info' in agent_name:
        env = FullInformationWrapper(env)
    # Create the agent
    agent = create_agent(agent_config, env, verbose=verbose)
    # Evaluate the agent!
    scores, data = evaluate_agent(
                            agent, env,
                            horizon=horizon,
                            n_episodes=n_episodes,
                            n_workers=n_workers,
                            return_full_information='full_info' in agent_name,
                            log_data=True)
    print_color(f'{agent_name}: mean score {scores.mean():.2f}, std {scores.std():.2f}', 'red')
    return scores, data

import os, yaml
from tqdm import tqdm

def main(args):

    config = yaml.safe_load(open(args.config, 'r'))
    config.update(dict(  # Read values from args
            horizon=[args.horizon],
            n_episodes=[args.n_episodes],
            seed=[args.seed],
            verbose=[args.verbose],
            n_workers=[args.n_workers]
        ))

    from verbal_gym.utils.benchmark_utils import batch_exp

    def logger(inputs, outputs):
        # Set log path
        log_path = os.path.join(args.log_dir, inputs['env_name'], inputs['agent_config']['agent_name']+'_'+time.strftime("%m%d_%H%M%S"))
        # Log
        os.makedirs(log_path, exist_ok=True)
        yaml.dump(inputs, open(os.path.join(log_path, 'exp_config.yaml'), 'w'))
        scores, data = outputs
        stats = dict(mean=float(scores.mean()), std=float(scores.std()), scores=scores.tolist())
        pickle.dump(data,  open(os.path.join(log_path, 'data.pkl'), 'wb'))
        yaml.dump(stats, open(os.path.join(log_path, 'stats.yaml'), 'w'))

    batch_run_exps = batch_exp(run_experiment, logger)
    batch_run_exps(**config)  # run experiments


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='results')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--n_episodes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())