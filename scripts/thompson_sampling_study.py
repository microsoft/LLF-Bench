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
            0: 0.8,
            1: 0.5
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

    def get_prob(self):

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

    def __init__(self, n_actions, permute=True):

        self.n_actions = n_actions
        self.permute = permute
        self.agent_history = None

        self.llm = GPT3()

    def update(self, action, reward):
        self.agent_history.append((action, reward))

    def get_prob(self):

        # TODO: generate all prob for each action
        # TODO: generate

        base_prompt = "I am trying to solve a problem where I have two possible actions: action 1 and action 2. " \
                      "For each action, I get either a good reward or a bad reward. An action can give me both " \
                      "good or bad reward with different probabilities. I want to eventually take an action that gives " \
                      "the good reward with higher probability. In the past, I have taken the following actions and " \
                      "received the feedback as stated below:\n"

        if self.permute:
            history_copy = list(self.agent_history)
            random.shuffle(history_copy)
        else:
            history_copy = self.agent_history

        logprob_actions = []
        for action in range(self.n_actions):

            prompt = base_prompt
            prompt += "\n".join([f"- Took action {action} and got a good reward"
                                if reward == 1 else f"- Took action {action} and got a bad reward"
                                for action, reward in history_copy])

            prompt += f"\n Based on the above feedback, I should choose action {action}."

            logprob_action = self.llm.logprob(prompt)
            logprob_actions.append(logprob_action)

        logprobs = torch.FloatTensor(logprob_actions)
        llm_probs = torch.softmax(logprobs, dim=0)

        llm_probs = [llm_probs[i].item() for i in range(self.n_actions)]

        return llm_probs


class CalibrationStudy:

    def __init__(self, agent_type, num_eps=25, warm_start_eps=5):

        self.num_eps = num_eps
        self.agent_type = agent_type

        self.warm_start_eps = warm_start_eps

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

    @staticmethod
    def take_action(prob):

        r = random.random()
        cumm = 0.0
        for i in range(len(prob)):
            if cumm <= r < cumm + prob[i]:
                return i
            cumm += prob[i]

        return len(prob) - 1

    def llm_sampling(self, env):

        llm_agent = LLMAgent()
        thompson_agent = ThompsonSampling()

        history = []

        for action in range(env.num_actions):
            for k in range(self.warm_start_eps):
                reward = env.step(action)
                history.append((action, reward))

        for i in range(self.num_eps):

            # Parse the history and decide the next action
            print(f"Starting Round {i+1}")
            mean_action, counts = self.get_mean_returns_and_count(history)

            print(f"History: Action 1 (Count: {counts.get(1, 0.0)}, Mean Return: {mean_action.get(1, 0.0)} "
                  f"and Action 2 (Count: {counts.get(2, 0.0)}, Mean Return: {mean_action.get(2, 0.0)}")

            llm_agent_prob = llm_agent.get_prob()
            thompson_agent_prob = thompson_agent.get_prob()

            print(f"Episode {i + 1}: Probability of action is: Action 1: {prob_action_1:3f}"
                  f" and Action 2: {prob_action_2:3f}.")

            print(f"Episode {i + 1}: Bayes Probability of action is: Action 1: {bayes_prob[1]:3f} "
                  f"and Action 2: {bayes_prob[2]:3f}.")

            prob = llm_agent_prob if self.agent_type == "llm" else thompson_agent_prob
            action = self.take_action(prob)
            reward = env.step(action)
            history.append((action, reward))

            # Update the agents.
            thompson_agent.update(action, reward)
            llm_agent.update(action, reward)

        print("Experiment Over. Generating plot and saving the data.")
        self.plot()
        self.save()

    def save(self):
        pass

    def plot(self):

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
    env = BernoulliBandit()
    llm_sampling(num_eps=200, thompson_action=True)
