import gym
import logging

from agents.agent_selector import AgentSelector
from llm.gpt.gpt import GPT3
from utils.multiprocess_logger import MultiprocessingLoggerManager
from verbal_gym.llm.gpt_models import GPT
from verbal_gym.llm.openai_utils import init_openai_api
from verbal_gym.utils.utils import evaluate_agent, set_seed
from verbal_gym.utils.misc_utils import print_color


def main(args, logger):

    n_episodes = args.n_episodes
    horizon = args.horizon

    # Create the environment
    env = gym.make(args.env_name)
    set_seed(args.seed, env)

    critic_llm = GPT3()

    assert isinstance(env.action_space, gym.spaces.Discrete), "Currently only handles discrete actions"

    dataset = []

    for _ in range(args.model_eps):

        obs = env.reset()

        for action in range(env.action_space.n):

            simulated_feedback = critic_llm.generate(
                prompt=f"I am trying to solve a problem described as {env.docstring}."
                       f"I was presented with a specific problem {obs}. "
                       f"I chose action {action}. How did I do?",
                max_tokens=100,
                temperature=0.0)

            dataset.append((obs, action, simulated_feedback))

    # Create a prompt using the above dataset
    prompt = f"I am presented with the following problem described as follows {env.docstring}." \
             f""

    # Create prompt to take action based on the above feedback
    scores = evaluate_agent(gpt_agent, env, horizon=horizon, n_episodes=n_episodes, n_workers=args.n_workers)
    print_color('Basic Posterior agent: mean score {:.2f}, std {:.2f}'.format(scores.mean(), scores.std()), 'red')
    return scores


def get_parser():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--logfile", type=str, default="./results.txt")
    parser.add_argument("--env_name",type=str, default="verbal-BanditTenArmedRandomRandom-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--not_paraphrase", action="store_true")
    parser.add_argument("--not_permute_history", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model", type=str, default="azure:gpt-35-turbo")
    parser.add_argument("--model_eps", type=str, default="azure:gpt-35-turbo")

    return parser


class LogLevel(object):
    pass


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    agent_selector = AgentSelector()
    agent = agent_selector.get_agent("basic")

    # Create a logger
    log_manager = MultiprocessingLoggerManager(file_path=args.logfile,
                                               logging_level=logging.INFO)
    logger = log_manager.get_logger("Main")

    main(args=args,
         logger=logger)
