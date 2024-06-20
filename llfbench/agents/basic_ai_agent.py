from textwrap import dedent, indent

from llfbench.agents.abstract_agent import BasicAgent
from llfbench.agents.utils import print_color


class BasicAIAgent(BasicAgent):

    NAME = "BasicAIAgent"

    system_prompt = dedent("""
    You are an agent tasked to solve an interactive problem with verbal feedback. You will see an Instruction. After you choose an action, you will see the feedback from the environment. Your goal is to choose the right actions to solve the task as fast as possible, according to the Instruction.

    Answer in the following format: First, begin with "Thought:" and write down your reflection on the feedback. Then in the next line write your response beginning with "Response:" and provide your chosen action. ONLY provide the chosen action after "Response:", without any additional comments or thoughts. Anything extra will cause errors, as your responses will be parsed by a computer program, not a human.

    Here is an example for an Instruction which asks you to choose a number between 1 and 10:

    Thought: I should choose a number that is not too high or too low, so I will choose 5.
    Response: 5

    An invalid response would be:

    Thought: I should choose a number that is not too high or too low, so I will choose 5.
    Response: I choose number 5
    """)

    def __init__(self,
                 llm,
                 verbose=False,
                 buffer_size=20,
                 prompt_template=None):
        """
            Args:
                llm: a large language model
                verbose: whether to print out the prompt and response
                buffer_size: the size of the replay buffer
                prompt_template: A prompt template with two parameters if ignore_observation is True and 3 otherwise
        """

        super().__init__(verbose, buffer_size, prompt_template)
        self.llm = llm

        if self.prompt_template is None:
            self.prompt_template = dedent("""
                History of feedbacks: {history}

                Current observation: {observation}

                Instruction: {instruction}
            """)

    @property
    def history(self):
        if len(self.buffer) == 0:
            history = 'None'
        else:
            history = "\n".join([
                f"Observation: {dp['observation']}\nAction: {dp['action']}\nFeedback: {dp['feedback']}\n"
                for dp in self.buffer])

        return history

    def act(self, observation, feedback, **kwargs):

        # update with the latest feedback (ignored in the first call)
        self.buffer.update(feedback=feedback, next_observation=observation)
        history = self.history

        # create prompt
        user_prompt = self.prompt_template.format(history=history, observation=observation, instruction=self.docstring)

        response, _ = self.llm.generate(user_prompt, max_tokens=1000)

        action = response.split('Response:')[-1].strip()

        if self.verbose:
            print_color(f'User:\n\n{user_prompt}\n', "blue")
            print_color(f'Agent:\n\n{response}\n', "green")
            print_color(f'Action:\n\n{action}\n', 'red')

        # update buffer and get world info
        self.buffer.append(observation=observation,
                           action=action,
                           feedback=None,
                           next_observation=None)
        return action
