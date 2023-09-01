import re
import pdb
import gym
import logging
import numpy as np

from verbal_gym.llm import make_llm
from verbal_gym.agents.fixed_agent import FixedAgent
from verbal_gym.utils.multiprocess_logger import MultiprocessingLoggerManager
from verbal_gym.utils.utils import evaluate_agent, set_seed
from verbal_gym.utils.misc_utils import print_color


def main(args, logger):

    n_episodes = args.n_episodes
    horizon = args.horizon

    # Create the environment
    env = gym.make(args.env_name)
    set_seed(args.seed, env)

    critic_llm = make_llm(args.model)

    assert isinstance(env.action_space, gym.spaces.Discrete), "Currently only handles discrete actions"

    dataset = []

    for i in range(args.model_eps):

        obs = env.reset()

        for action in range(env.action_space.n):

            prompt = f"I am trying to solve a problem described as follows.\n {env.docstring}. " \
                     f"I was presented with a specific problem {obs}. " \
                     f"Is action {action} a good action or a bad action? why or why not?"

            simulated_feedback, _ = critic_llm.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.0)

            logger.log(f"Episode {i}, Action {action}\n")
            logger.log(f"Agent Prompt: {prompt}\n")
            logger.log(f"Simulated Feedback: {simulated_feedback}\n\n")
            dataset.append((obs, action, simulated_feedback))

    logger.log("=" * 20)
    logger.log("\n\n")

    # Assume bandit setting, compute aggregate feedback for each action and then take the best action
    aggregate_feedbacks = dict()
    for action in range(env.action_space.n):

        feedbacks = [f"- {dp_simulated_feedback} "
                     for dp_obs, dp_action, dp_simulated_feedback in dataset if dp_action == action]

        feedback_string = "\n".join(feedbacks)

        aggregation_prompt = \
            f"I am trying to solve a problem described as follows.\n {env.docstring}. " \
            f"I took action {action} many times and received the following feedback. \n {feedback_string}. \n" \
            f"Can you please summarize all the essential comments about this action, whether it was good or bad, " \
            f"below. Also include a recommendation whether it is a good action to take. \n"

        # Compute aggregate feedback
        aggregate_feedback, _ = critic_llm.generate(prompt=aggregation_prompt,
                                                    max_tokens=100,
                                                    temperature=0.0)

        logger.log(f"Computing Aggregate Feedback for Action {action}\n")
        logger.log(f"Aggregation Prompt {aggregation_prompt}\n")
        logger.log(f"Aggregate Feedback {aggregate_feedback}\n\n")

        aggregate_feedbacks[action] = aggregate_feedback

    evaluation_prompt = f"I am trying to solve a problem described as {env.docstring}. I can take {env.action_space.n}" \
                        f"possible actions. For each action, my overall feedback is given below:\n"

    for action in range(env.action_space.n):
        evaluation_prompt += f"Action {action}: {aggregate_feedbacks[action]}\n"

    evaluation_prompt += f"Based upon the above feedback. The best action is "

    answer_feedback, _ = critic_llm.generate(prompt=evaluation_prompt,
                                             max_tokens=5,
                                             temperature=0.0)

    answers = re.findall(r'\d+', answer_feedback)

    logger.log(f"Evaluating Prompt {evaluation_prompt}\n")
    logger.log(f"Answer Feedback {answer_feedback}\n")
    logger.log(f"Parsed Answers {answers}\n\n")

    if len(answers) == 0:
        print("Failed")
        return [0] * n_episodes

    fixed_action = int(answers[0])
    agent = FixedAgent(fixed_action=fixed_action)
    logger.log(f"Created a fixed agent with action {fixed_action}")

    # Create prompt to take action based on the above feedback
    scores, _ = evaluate_agent(agent, env, horizon=horizon, n_episodes=n_episodes, n_workers=args.n_workers)
    print_color('Basic Posterior agent: mean score {:.2f}, std {:.2f}'.format(scores.mean(), scores.std()), 'red')
    logger.log(f"Mean total reward on {n_episodes} episodes is {np.mean(scores)} with std of {np.std(scores)}")

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
    parser.add_argument("--model", type=str,  default="gcr:gpt-35")
    parser.add_argument("--model_eps", type=int, default=2)

    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    # Create a logger
    log_manager = MultiprocessingLoggerManager(file_path=args.logfile,
                                               logging_level=logging.INFO)
    logger = log_manager.get_logger("Main")

    main(args=args,
         logger=logger)
