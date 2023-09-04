import pdb
import math
import random

from verbal_gym.llm.gpt.gpt import GPT3


class BernoulliBandit:

    def __init__(self):
        self.num_actions = 2
        self.params = {
            1: 0.8,
            2: 0.5
        }

    def act(self, action):

        r = random.random()
        if r < self.params[action]:
            return 1.0
        else:
            return 0.0


def generate_prompt(history, action):

    prompt = "I am trying to solve a problem where I have two possible actions: action 1 and action 2. " \
             "For each action, I get either a good reward or bad reward. An action can give me both good or bad " \
             "reward with different probabilities. I want to eventually take an action that gives the good reward " \
             "with higher probability. In the past, I have taken the following actions and received the feedback as " \
             "stated below:\n"

    prompt += "\n".join([f"- Took action {action} and got a good reward"
                         if reward == 1 else f"- Took action {action} and got a bad reward"
                         for action, reward in history])

    prompt += f"\n Based on the above feedback, the best action to take is action {action}."
    return prompt


env = BernoulliBandit()
llm = GPT3()
history = []

for k in range(10):
    action = random.randint(0, 1) + 1
    reward = env.act(action)
    history.append((action, reward))

action = random.randint(0, 1) + 1

for i in range(10):

    reward = env.act(action)
    history.append((action, reward))

    print(f"Episode {i}: Took action {action} and got reward {reward}.")

    # Parse the history and decide the next action
    prompt_action_1 = generate_prompt(history=history, action=1)
    prompt_action_2 = generate_prompt(history=history, action=2)

    logprob_action_1 = llm.logprob(prompt_action_1)
    logprob_action_2 = llm.logprob(prompt_action_2)

    unnorm_prob_action_1 = math.exp(logprob_action_1)
    unnorm_prob_action_2 = math.exp(logprob_action_2)

    prob_action_1 = unnorm_prob_action_1 / float(unnorm_prob_action_1 + unnorm_prob_action_2)
    prob_action_2 = unnorm_prob_action_2 / float(unnorm_prob_action_1 + unnorm_prob_action_2)

    print(f"Episode {i}: Probability of action is: Action 1: {prob_action_1:3f} and Action 2: {prob_action_2:3f}.")

    # Sample according to the above prob
    r = random.random()
    if r < prob_action_1:
        action = 1
    else:
        action = 2

    pdb.set_trace()