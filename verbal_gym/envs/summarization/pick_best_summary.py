import pdb

from verbal_gym.gpt.gpt import GPT3


class PickBestSummary:
    """
        Select the best summary
    """

    def __init__(self, num_actions=4, summary_max_tokens=50, feedback_max_tokens=40):

        from datasets import load_dataset

        self.dataset = load_dataset('cnn_dailymail', '3.0.0')

        self.summary_generator_llm = GPT3()
        self.critic_llm = GPT3()

        assert num_actions >= 2, "There must be at least 2 actions."
        self.num_actions = num_actions

        self.summary_max_tokens = summary_max_tokens
        self.feedback_max_tokens = feedback_max_tokens

        # Prompts for summary generation
        self.good_summary_prompt = "You are given the article below. \n Article: %s. \n Write a good, " \
                                   "high-quality summary capturing the essential points of the article."

        self.bad_summary_prompt = "You are given the article below. \n Article: %s. \n Write a bad, " \
                                  "low-quality, garbage summary of the article which has many factual errors.."

        self.feedback_prompt = "Article: %s\n Summary: %s \n. You are provided an article and its summary above. " \
                               "Provide a feedback for the quality of this summary describing any positive features " \
                               "and specifically mentioning all factual errors or important information missed by the " \
                               "summary. Please be detailed."

        self.ctr = 0
        self.current_article = None
        self.current_candidate_summaries = None

    def _generate_summaries(self):

        if self.current_article is None:
            raise AssertionError("You need to do reset first.")

        # Suppose we have K actions, i.e., summaries to generate
        # Generate 1 good summary using argmax and temperature 0
        # Generate floor(k / 2) as fake examples with poor summary
        # Generate remaining as good summaries with temperature 1

        summaries_with_score = []

        # Good summary
        good_prompt = self.good_summary_prompt % self.current_article
        summary, info = self.summary_generator_llm.generate(prompt=good_prompt,
                                                            max_tokens=self.summary_max_tokens,
                                                            logprobs=1,
                                                            temperature=0.0)

        summaries_with_score.append((summary, info["logprob"]))

        # int(K/2) bad summaries
        bad_prompt = self.bad_summary_prompt % self.current_article

        bad_summaries = int(self.num_actions / 2.0)
        for _ in range(0, bad_summaries):

            summary, info = self.summary_generator_llm.generate(prompt=bad_prompt,
                                                                max_tokens=self.summary_max_tokens,
                                                                logprobs=1,
                                                                temperature=1.0)
            summaries_with_score.append((summary, info["logprob"]))

        for _ in range(0, self.num_actions - 1 - bad_summaries):

            summary, info = self.summary_generator_llm.generate(prompt=bad_prompt,
                                                                max_tokens=self.summary_max_tokens,
                                                                logprobs=1,
                                                                temperature=1.0)
            summaries_with_score.append((summary, info["logprob"]))

    def reset(self):

        article = self.dataset["train"][self.ctr]["article"]
        self.ctr = (self.ctr + 1) % len(self.dataset["train"])

        candidate_summaries = self._generate_summaries()
        observation = {
            "article": article,
            "actions": candidate_summaries
        }
        info = {
            "help": "You are given an article in text and a list of summaries. Pick a summary indexed as "
                    "0 to num_summaries - 1 that is the most approprriate summary."
        }

        return observation, info

    def step(self, action):

        feedback = self.critic_llm.generate(
            prompt=self.feedback_prompt % (self.current_article, self.current_candidate_summaries[action]),
            max_tokens=self.feedback_max_tokens
        )
        return feedback
