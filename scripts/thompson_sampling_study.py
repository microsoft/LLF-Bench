import pdb
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta
from verbal_gym.llm.gpt.gpt import GPT3


class BernoulliBandit:

    def __init__(self):
        self.num_actions = 2
        self.params = {
            1: 0.8,
            2: 0.5
        }

    def step(self, action):

        r = random.random()
        if r < self.params[action]:
            return 1.0
        else:
            return 0.0


def generate_prompt(history, action):

    prompt = "I am trying to solve a problem where I have two possible actions: action 1 and action 2. " \
             "For each action, I get either a good reward or a bad reward. An action can give me both good or bad " \
             "reward with different probabilities. I want to eventually take an action that gives the good reward " \
             "with higher probability. In the past, I have taken the following actions and received the feedback as " \
             "stated below:\n"

    history_copy = list(history)
    random.shuffle(history_copy)
    prompt += "\n".join([f"- Took action {action} and got a good reward"
                         if reward == 1 else f"- Took action {action} and got a bad reward"
                         for action, reward in history_copy])

    prompt += f"\n Based on the above feedback, I should choose action {action}."
    return prompt


def get_bayes_prob(action_1_prior, action_2_prior,num_est=1000):

    action_1_samples = beta.rvs(a=action_1_prior["a"], b=action_1_prior["b"], size=num_est)
    action_2_samples = beta.rvs(a=action_2_prior["a"], b=action_2_prior["b"], size=num_est)

    action_prior = np.vstack([action_1_samples, action_2_samples])  # 2 x num_est
    actions_chosen = np.argmax(action_prior, axis=0) + 1

    action_counts = {1: 0, 2: 0}
    for action_chosen in actions_chosen:
        action_counts[action_chosen] += 1

    action_prob = {1: action_counts[1] / float(num_est), 2: action_counts[2] / float(num_est)}

    return action_prob


def get_mean_returns_and_count(history):

    sum_action = dict()
    counts = dict()

    for action, reward in history:

        if action not in sum_action:
            sum_action[action] = 0.0

        if action not in counts:
            counts[action] = 0.0

        sum_action[action] += reward
        counts[action] += 1

    mean_action = {action: sum_action[action] / float(max(1, counts[action])) for action in sum_action}
    return mean_action, counts


env = BernoulliBandit()
llm = GPT3()


def llm_sampling(num_eps=25):

    action_1_prior = {"a": 0.5, "b": 0.5}
    action_2_prior = {"a": 0.5, "b": 0.5}

    history = []

    warm_start_eps = 5
    for action in [1, 2]:
        for k in range(warm_start_eps):
            reward = env.step(action)

            if action == 1:
                action_1_prior = {"a": action_1_prior["a"] + reward,
                                  "b": action_1_prior["b"] + 1 - reward}
            elif action == 2:
                action_2_prior = {"a": action_2_prior["a"] + reward,
                                  "b": action_2_prior["b"] + 1 - reward}
            else:
                raise AssertionError(f"Action must be in {{1, 2}}. Found {action}")

            history.append((action, reward))

    random.shuffle(history)
    print(f"You are solving a bandit problem with two actions. Action 1 gives reward 1 with prob. {env.params[1]} "
          f"and 0 otherwise. Action 2 gives reward 1 with prob {env.params[1]} and 0 otherwise.")
    print(f"Starting with warm-starting where each of the two actions were taken {warm_start_eps} times.")

    bayes_action_1_prob = []
    action_1_prob = []

    action_1_mean = []
    action_1_count = []

    action_1_eps_indicator = []
    action_1_mean_indicator = []

    action_2_mean = []
    action_2_count = []

    for i in range(num_eps):

        # Parse the history and decide the next action
        print(f"Starting Round {i+1}")
        mean_action, counts = get_mean_returns_and_count(history)

        print(f"History: Action 1 (Count: {counts.get(1, 0.0)}, Mean Return: {mean_action.get(1, 0.0)} "
              f"and Action 2 (Count: {counts.get(2, 0.0)}, Mean Return: {mean_action.get(2, 0.0)}")

        action_1_mean.append(mean_action.get(1, 0.0))
        action_1_count.append(counts.get(1, 0.0))

        action_2_mean.append(mean_action.get(2, 0.0))
        action_2_count.append(counts.get(2, 0.0))

        prompt_action_1 = generate_prompt(history=history, action=1)
        prompt_action_2 = generate_prompt(history=history, action=2)

        logprob_action_1 = llm.logprob(prompt_action_1)
        logprob_action_2 = llm.logprob(prompt_action_2)

        logprobs = torch.FloatTensor([logprob_action_1, logprob_action_2])
        probs = torch.softmax(logprobs, dim=0)

        prob_action_1 = probs[0].item()
        prob_action_2 = probs[1].item()

        probs = [prob_action_1, prob_action_2]
        print(f"Episode {i + 1}: Probability of action is: Action 1: {prob_action_1:3f}"
              f" and Action 2: {prob_action_2:3f}.")

        bayes_prob = get_bayes_prob(action_1_prior, action_2_prior, num_est=1000)
        print(f"Episode {i + 1}: Bayes Probability of action is: Action 1: {bayes_prob[1]:3f} "
              f"and Action 2: {bayes_prob[2]:3f}.")

        bayes_action_1_prob.append(bayes_prob[1])

        r = random.random()
        if r < probs[0]:
            action = 1
        else:
            action = 2

        if action == 1:
            action_1_eps_indicator.append(i + 1)
            action_1_mean_indicator.append(prob_action_1)

        reward = env.step(action)
        history.append((action, reward))

        if action == 1:
            action_1_prior = {"a": action_1_prior["a"] + reward,
                              "b": action_1_prior["b"] + 1 - reward}
        elif action == 2:
            action_2_prior = {"a": action_2_prior["a"] + reward,
                              "b": action_2_prior["b"] + 1 - reward}
        else:
            raise AssertionError(f"Action must be in {{1, 2}}. Found {action}")

        print(f"Episode {i + 1}: Took action {action} and got reward {reward}.")

        action_1_prob.append(prob_action_1)

        print("\n\n")
        # pdb.set_trace()

    plt.clf()
    plt.title(f"Experiments on Bernoulli Bandit. Warm start with {warm_start_eps} rounds per action.")
    plt.xlabel("Episodes")
    episodes = list(range(1, num_eps + 1))

    plt.plot(episodes, action_1_prob, label="Prob. of action 1", color="red")
    plt.plot(episodes, bayes_action_1_prob, label="Bayes Prob. of action 1", color="orange", linestyle="--")

    plt.plot(action_1_eps_indicator, action_1_mean_indicator, ls="", marker="o", color="red", label='_nolegend_')

    plt.plot(episodes, [env.params[1]] * num_eps, label="Mean return of action 1", color="blue", linestyle="--")
    plt.plot(episodes, action_1_mean, label="Emp. mean return of action 1", color="blue")
    # plt.plot(roots,[poly[i] for i in mark], ls="", marker="o", label="points")

    plt.plot(episodes, [env.params[2]] * num_eps, label="Mean return of action 2", color="green", linestyle="--")
    plt.plot(episodes, action_2_mean, label="Emp. mean return of action 2", color="green")
    # plt.plot(episodes, action_2_count, label="Empirical count of action 2", color="green")

    plt.legend()
    plt.savefig("./bernoulli_bandit.png")
    pdb.set_trace()


def bayes_sampling(num_eps=25):

    action_1_prior = {"a": 0.5, "b": 0.5}
    action_2_prior = {"a": 0.5, "b": 0.5}
    action_1_prob = []

    for i in range(num_eps):

        num_est = 1000
        action_1_samples = beta.rvs(a=action_1_prior["a"], b=action_1_prior["b"], size=num_est)
        action_2_samples = beta.rvs(a=action_2_prior["a"], b=action_2_prior["b"], size=num_est)

        action_prior = np.vstack([action_1_samples, action_2_samples])      # 2 x num_est
        actions_chosen = np.argmax(action_prior, axis=0) + 1

        action_counts = {1: 0, 2: 0}
        for action_chosen in actions_chosen:
            action_counts[action_chosen] += 1

        action_prob = {1: action_counts[1] / float(num_est), 2: action_counts[2] / float(num_est)}

        action_1_prob.append(action_prob[1])
        print(f"Action 1 prob is {action_prob[1]} and action 2 prob is {action_prob[2]}")

        r = random.random()
        if r < action_prob[1]:
            action = 1
        else:
            action = 2

        reward = env.step(action)

        if action == 1:
            action_1_prior = {"a": action_1_prior["a"] + reward,
                              "b": action_1_prior["b"] + 1 - reward}
        elif action == 2:
            action_2_prior = {"a": action_2_prior["a"] + reward,
                              "b": action_2_prior["b"] + 1 - reward}
        else:
            raise AssertionError(f"Action must be in {{1, 2}}. Found {action}")

    plt.clf()
    plt.title(f"Experiments on Bernoulli Bandit.")
    plt.xlabel("Episodes")
    episodes = list(range(1, num_eps + 1))

    plt.plot(episodes, action_1_prob, label="Bayes Prob. of action 1", color="red")

    plt.legend()
    plt.savefig("./bernoulli_bandit_bayes_prob.png")
    pdb.set_trace()


if __name__ == '__main__':
    # bayes_sampling()
    llm_sampling(num_eps=100)
