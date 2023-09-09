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

        assert action in [0, 1]

        r = random.random()
        if r < self.params[action]:
            return 1.0
        else:
            return 0.0


class ThompsonSampling:

    def __init__(self, n_actions, a=0.5, b=0.5, num_est=1000):
        self.n_actions = n_actions
        self.num_est = num_est

        self.params = []
        for _ in range(n_actions):
            self.params.append({"a": a, "b": b})

    def update_from_history(self, history):
        for action, reward in history:
            self.update(action, reward)

    def update(self, action, reward):
        self.params[action]["a"] = self.params[action]["a"] + reward
        self.params[action]["b"] = self.params[action]["b"] + 1 - reward

    def get_bayes_prob(self):

        all_action_samples = []
        for i in range(self.n_actions):
            action_samples = beta.rvs(a=self.params[i]["a"], b=self.params[i]["b"], size=self.num_est)
            all_action_samples.append(action_samples)

        action_prior = np.vstack(all_action_samples)        # K x num_est
        actions_chosen = np.argmax(action_prior, axis=0)    # num_est
        assert actions_chosen.shape[0] == self.num_est

        action_counts = np.zeros(self.n_actions)
        for action_chosen in actions_chosen:
            action_counts[action_chosen] += 1.0

        action_prob = action_counts / float(self.num_est)   # num_est

        return action_prob


class LLMAgent:

    def __init__(self):
        self.llm = GPT3()

    def generate_prompt(self, history, action, permute=True):

        # TODO: generate all prob for each action
        # TODO: generate

        prompt = "I am trying to solve a problem where I have two possible actions: action 1 and action 2. " \
                 "For each action, I get either a good reward or a bad reward. An action can give me both good or bad " \
                 "reward with different probabilities. I want to eventually take an action that gives the good reward " \
                 "with higher probability. In the past, I have taken the following actions and received the feedback as " \
                 "stated below:\n"

        if permute:
            history_copy = list(history)
            random.shuffle(history_copy)
        else:
            history_copy = history
        prompt += "\n".join([f"- Took action {action} and got a good reward"
                             if reward == 1 else f"- Took action {action} and got a bad reward"
                             for action, reward in history_copy])

        prompt += f"\n Based on the above feedback, I should choose action {action}."
        return prompt

    @staticmethod
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


class CalibrationStudy:



    def llm_sampling(num_eps=25, thompson_action=False):

        if thompson_action:
            prob_type = "Thompson"
        else:
            prob_type = "LLM"

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
            llm_probs = torch.softmax(logprobs, dim=0)

            prob_action_1 = llm_probs[0].item()
            prob_action_2 = llm_probs[1].item()

            llm_probs = [prob_action_1, prob_action_2]
            print(f"Episode {i + 1}: Probability of action is: Action 1: {prob_action_1:3f}"
                  f" and Action 2: {prob_action_2:3f}.")

            bayes_prob = get_bayes_prob(action_1_prior, action_2_prior, num_est=1000)
            print(f"Episode {i + 1}: Bayes Probability of action is: Action 1: {bayes_prob[1]:3f} "
                  f"and Action 2: {bayes_prob[2]:3f}.")

            bayes_action_1_prob.append(bayes_prob[1])

            if thompson_action:
                probs_to_use = [bayes_prob[1], bayes_prob[2]]
            else:
                probs_to_use = llm_probs

            r = random.random()
            if r < probs_to_use[0]:
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
        plt.title(f"Experiments on Bernoulli Bandit. Warm start: {warm_start_eps}, Prob type {prob_type}.")
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
        plt.savefig(f"./bernoulli_bandit_{prob_type}.png")
        pdb.set_trace()



if __name__ == '__main__':
    llm_sampling(num_eps=200, thompson_action=True)
