import random
from verbal_gym.llm.gpt.gpt import GPT3
import gym, sys
class PickBestSummary(gym.Env):
    """
        Select the best summary
    """

    Binary, LOGPROB = range(2)

    def __init__(self, num_actions=4, summary_max_tokens=50, fixed=True, seed=1234,
                 feedback_max_tokens=40, reward_type="binary"):

        from datasets import load_dataset

        self.dataset = load_dataset('cnn_dailymail', '3.0.0')

        self.summary_generator_llm = GPT3()
        self.critic_llm = GPT3()

        assert num_actions >= 2, "There must be at least 2 actions."
        self.num_actions = num_actions

        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Text(sys.maxsize)

        self.summary_max_tokens = summary_max_tokens
        self.feedback_max_tokens = feedback_max_tokens
        self.fixed = fixed

        self.seed = seed
        self.rng = random.Random(seed)

        if reward_type == "binary":
            self.reward_type = PickBestSummary.Binary
        elif reward_type == "logprob":
            self.reward_type = PickBestSummary.LOGPROB
        else:
            raise AssertionError(f"Unhandled reward_type {reward_type}. Can only handle: binary, logprob")

        # Prompts for summary generation
        self.good_summary_prompt = "You are given the article below. \n Article: %s. \n Write a good, " \
                                   "high-quality summary capturing the essential points of the article. Summary:"

        self.bad_summary_prompt = "You are given the article below. \n Article: %s. \n Write a bad, " \
                                  "low-quality, garbage summary of the article which has many factual errors. Summary:"

        self.feedback_prompt = "Article: %s\n Summary: %s \n. You are provided an article and its summary above. " \
                               "Provide a feedback for the quality of this summary describing any positive features " \
                               "and specifically mentioning all factual errors or important information missed by the " \
                               "summary. Please be detailed."

        self.ctr = 0
        self.current_article = None
        self.current_summaries_with_reward = None

    def _generate_summaries(self):

        if self.current_article is None:
            raise AssertionError("You need to do reset first.")

        num_bad_summaries = int(self.num_actions / 2.0)

        # Suppose we have K actions, i.e., summaries to generate
        # Generate 1 good summary using argmax and temperature 0
        # Generate floor(k / 2) as fake examples with poor summary
        # Generate remaining as good summaries with temperature 1

        summaries_with_score = []

        # Good summaries
        good_prompt = self.good_summary_prompt % self.current_article
        for i in range(0, self.num_actions - num_bad_summaries):

            temperature = 0.0 if i == 0 else 1.0
            generation, info = self.summary_generator_llm.generate(prompt=good_prompt,
                                                                   max_tokens=self.summary_max_tokens,
                                                                   echo=True,
                                                                   logprobs=
                                                                   1 if self.reward_type == self.LOGPROB else None,
                                                                   temperature=temperature)

            if good_prompt != generation[:len(good_prompt)]:
                raise AssertionError(f"{generation} should start with {good_prompt}")

            summary = generation[len(good_prompt):]

            if self.reward_type == self.LOGPROB:
                reward = info["logprob"]
            elif self.reward_type == self.Binary:
                reward = 1.0 if i == 0 else 0.0
            else:
                raise AssertionError(f"Unhandled reward type {self.reward_type}")

            summaries_with_score.append((summary, reward))

        # Bad summaries
        bad_prompt = self.bad_summary_prompt % self.current_article
        for _ in range(0, num_bad_summaries):

            summary, info = self.summary_generator_llm.generate(prompt=bad_prompt,
                                                                max_tokens=self.summary_max_tokens,
                                                                logprobs=None,
                                                                temperature=1.0)

            if self.reward_type == self.LOGPROB:
                # Compute logprob of the summary under the new summary
                good_prompt_with_bad_summary = good_prompt + summary
                reward = self.summary_generator_llm.get_logprobs(prompt=good_prompt_with_bad_summary)
            elif self.reward_type == self.Binary:
                reward = 0
            else:
                raise AssertionError(f"Unhandled reward type {self.reward_type}")

            summaries_with_score.append((summary, reward))

        random.shuffle(summaries_with_score)

        return summaries_with_score

    def reset(self):

        if self.current_article is None:
            # First run so set the starting counter position
            self.ctr = self.rng.randint(0, len(self.dataset["train"]) - 1)

        if not self.fixed or self.current_article is None:
            self.current_article = self.dataset["train"][self.ctr]["article"]
            self.current_summaries_with_reward = self._generate_summaries()
            self.ctr = (self.ctr + 1) % len(self.dataset["train"])

        observation = f"You are given an article below. \n\n {self.current_article}. \n\n" \
                      f"You are also given {self.num_actions} many summaries some of which are good " \
                      f"and others can be incorrect. The list of summaries and their indices are:\n\n"

        for idx, summary_with_reward in enumerate(self.current_summaries_with_reward):
            summary, _ = summary_with_reward
            observation = observation + f"\n\n {idx}: {summary}"

        observation = observation + "\n\n Please select the best summary for the above article by entering its index."

        return observation

    def step(self, action):

        feedback, _ = self.critic_llm.generate(
            prompt=self.feedback_prompt % (self.current_article, self.current_summaries_with_reward[action][0]),
            max_tokens=self.feedback_max_tokens
        )

        # The next observation doesn't make sense but instead of returning None, I return the chosen action description
        next_observation = self.current_summaries_with_reward[action][0]
        reward = self.current_summaries_with_reward[action][1]
        done = True
        info = {
            "feedback": feedback
        }

        return next_observation, reward, done, info
