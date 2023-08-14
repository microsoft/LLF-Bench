import pdb

from verbal_gym.gpt.gpt3 import GPT3


class PickBestSummary:
    """
        Select the best summary
    """

    def __init__(self, num_actions=4):

        from datasets import load_dataset

        self.dataset = load_dataset('cnn_dailymail', '3.0.0')

        self.summary_generator_llm = GPT3()
        self.critic_llm = GPT3()
        self.num_actions = num_actions

        self.ctr = 0
        self.current_article = None
        self.current_candidate_summaries = None

    def _generate_summaries(self):

        # Suppose we have K actions, i.e., summaries to generate
        # Generate 1 good summary
        # Generate k
        pass

    def reset(self):

        article = self.dataset["train"][self.ctr]["article"]
        self.ctr = (self.ctr + 1) % len(self.dataset["train"])

        candidate_summaries = []
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
            prompt=f"Article: {self.current_article}\n Summary: {self.current_candidate_summaries[action]} "
                   f"\n. You are provided an article and its summary above. Provide a feedback for the quality of "
                   f"this summary describing any positive features and specifically mentioning all factual errors"
                   f" or important information missed by the summary. Please be detailed.",
            max_tokens=40
        )
        return feedback


pdb.set_trace()

