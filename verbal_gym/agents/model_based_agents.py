import numpy as np

from textwrap import dedent, indent
from verbal_gym.agents.basic_agent import BasicAgent
from verbal_gym.agents.utils import extract_action
from verbal_gym.utils.misc_utils import print_color


class _BasicAgent(Agent):  # this is the old deprecated version

    NAME = "BasicAgent"

    system_prompt = dedent("""
    You are an agent tasked to solve an interactive problem with verbal
    feedback. You will see "Problem Description" that tell you want to problem
    is about (such as the goal of the task, the action space you should choose
    from, the rules, the constraints, etc.) After you choose an action, you will
    see the feedback from the environment. You goal is to choose the right
    actions solve the task as fast as possible, according to "Problem
    Description".
    """)

    def __init__(self, llm, n_actions, verbose=False, action_name='Action'):
        super(Agent, self).__init__()
        self.llm = llm
        self.n_actions = n_actions
        self.verbose = verbose
        self.action_name = action_name

        # history
        self.previous_action = None
        self.history = []

        self.prompt_template = dedent("""\
            You're given with the problem below:

            Problem Description: {}

            You have observed the following history of feedbacks:

            {}

            Choose your action according to the problem description, history of feedbacks, and explain why.

        """)

        if n_actions is not None:
            self.prompt_template += dedent(f"""\
            The response should be in the following format, where <your action> should be an integer from 0 and less than {n_actions}. You must follow this format!!!

                Reasoning: <your reasoning>
                {action_name}: #<your action>#

            """)
        else:
            self.prompt_template += dedent(f"""\
            The response should be in the following format, where <your action> is the final answer. You must follow this format!!!

                Reasoning: <your reasoning>
                {action_name}: <your action>

            """)

    def reset(self, docstring):
        self.docstring = docstring
        self.history = []

    def update_history(self, feedback):
        world_info = 'None'
        if len(self.history) > 0:
            self.history[-1]['feedback'] = feedback
            # TODO how to format this nicely?
            # world_info = '\n'.join([f'\t {self.action_name} {item["action"]} --- {item["feedback"]}'
            # for item in self.history])
            world_info = '\n'.join(
                [indent(f'{self.action_name}: {item["action"]}\n\nFeedback: {item["feedback"]}\n\n\n','\t')
                 for item in self.history])

        return world_info

    def act(self, obs, feedback, **kwargs):
        world_info = self.update_history(feedback)
        user_prompt = self.prompt_template.format(self.docstring, world_info)
        response, _ = self.llm.generate(user_prompt)

        if self.verbose:
            print_color(f'User:\n\n{user_prompt}\n', "blue")
            print_color(f'Agent:\n\n{response}\n', "green")

        action = response.split(self.action_name+':')[-1]
        if self.n_actions is not None:
            action = extract_action(action, self.n_actions)

        if self.verbose:
            print_color(f'Action:\n\n{action}\n', 'red')
        self.history.append({'action': action, 'feedback': None})

        return action


class ModelBasedAgent(_BasicAgent):

    NAME = "ModelBasedAgent"

    def __init__(self, llm, n_actions, verbose=False, action_name='Action',
                 simulate_once=False, num_simulations=1, permute_history=True, paraphrase_agent=None, logger=None):
        super().__init__(llm, n_actions, verbose=verbose, action_name=action_name)
        self.permute_history = permute_history
        self.paraphrase_agent = paraphrase_agent

        self.simulate_once = simulate_once
        self.num_simulations = num_simulations
        self._docstring = None
        self.logger = logger

        # Counters to be reset
        self.simulated_feedback = None

    def reset(self, docstring):
        self._docstring = docstring

    def update_history(self, feedback):

        world_info = 'None'

        if len(self.history) > 0:

            # This is done because feedback is []
            self.history[-1]['feedback'] = feedback

            # XXX Add random permuation and paraphrasing
            history = np.random.permutation(self.history) if self.permute_history else self.history
            world_info = '\n'.join(
                [indent(f'{self.action_name}: {item["action"]}\n\nFeedback: '
                        f'{self.paraphrase(item["feedback"])}\n\n\n','\t') for item in history])

        return world_info

    def simulate_feedback_for_discrete_action(self, obs):

        # Create the environment
        dataset = []

        for i in range(self.num_simulations):

            for action in range(self.n_actions):
                prompt = f"We are trying to solve a problem described as follows.\n {self._docstring}. " \
                         f"We are presented with a specific problem {obs}. " \
                         f"If we decide to take the following action: {action}.\n " \
                         f"Please tell us whether this is a good or bad choice. Why or why not? " \
                         f"Please be as detailed as possible."

                simulated_feedback, _ = self.llm.generate(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.0)

                if self.logger is not None:
                    self.logger.log(f"Episode {i}, Action {action}\n")
                    self.logger.log(f"Agent Prompt: {prompt}\n")
                    self.logger.log(f"Simulated Feedback: {simulated_feedback}\n\n")
                    dataset.append((obs, action, simulated_feedback))

        if self.logger is not None:
            self.logger.log("=" * 20)
            self.logger.log("\n\n")

        # Assume bandit setting, compute aggregate feedback for each action and then take the best action
        aggregate_feedbacks = dict()
        for action in range(self.n_actions):
            feedbacks = [f"- {dp_simulated_feedback} "
                         for dp_obs, dp_action, dp_simulated_feedback in dataset if dp_action == action]

            feedback_string = "\n".join(feedbacks)

            aggregation_prompt = \
                f"I am trying to solve a problem described as follows.\n {self.docstring}. " \
                f"I took action {action} many times and received the following feedback. \n {feedback_string}. \n" \
                f"Can you please summarize all the essential comments about this action, whether it was good or bad, " \
                f"below. Also include a recommendation whether it is a good action to take. \n"

            # Compute aggregate feedback
            aggregate_feedback, _ = self.llm.generate(prompt=aggregation_prompt,
                                                      max_tokens=100,
                                                      temperature=0.0)

            if self.logger is not None:
                self.logger.log(f"Computing Aggregate Feedback for Action {action}\n")
                self.logger.log(f"Aggregation Prompt {aggregation_prompt}\n")
                self.logger.log(f"Aggregate Feedback {aggregate_feedback}\n\n")

            aggregate_feedbacks[action] = aggregate_feedback

        return aggregate_feedbacks

    def simulate_feedback_for_text(self, obs, world_info):

        # Create the environment
        dataset = dict()

        for i in range(self.num_simulations):

            user_prompt = self.prompt_template.format(self.docstring, world_info)
            response, _ = self.llm.generate(user_prompt,
                                            max_tokens=100,
                                            temperature=1.0)
            action = response.split(self.action_name + ':')[-1]

            # Generate an action
            prompt = f"We are trying to solve a problem described as follows.\n {self._docstring}. " \
                     f"We are presented with a specific problem {obs}. " \
                     f"If we decide to take the following action: {action}.\n " \
                     f"Please tell us whether this is a good or bad choice. Please be as detailed as possible."

            simulated_feedback, _ = self.llm.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.0)

            if self.logger is not None:
                self.logger.log(f"Episode {i}, Action {action}\n")
                self.logger.log(f"Agent Prompt: {prompt}\n")
                self.logger.log(f"Simulated Feedback: {simulated_feedback}\n\n")

            if action in dataset:
                dataset[action] += "\n" + simulated_feedback
            else:
                dataset[action] = simulated_feedback

            # TODO we should include obs in decision making and storing feedback
            # dataset.append((obs, action, simulated_feedback))

        return dataset

    def simulations_to_text(self):
        # TODO Add an option to select simulations
        simulated_info = '\n'.join(
            [indent(f'{self.action_name}: {action}\n\nFeedback: {self.paraphrase(feedback)}\n\n\n', '\t')
             for action, feedback in self.simulated_feedback.items()])

        return simulated_info

    def act(self, obs, feedback, **kwargs):

        # Update history. History only contains things which actually happened
        world_info = self.update_history(feedback)

        # Given observation, simulate feedback. This happens based on simulate_once and type of actions
        # If simulate_once is True
        #   -  action is discrete, then we will simulate the effect of each action once and use the most promising one.
        #   -  action is Text, then we will generate a few text pieces and simulate feedback and use them as feedback
        #
        # If simulate_once is False
        #   - action is discrete, then we will simulate the effect of each action each time
        #   - action is Text, then sample a few text pieces and simulate feedback and use them as feedback .This can be
        #      slow in practice.

        if not self.simulate_once or self.simulated_feedback is None:
            # TODO: If simulate_once is False, add the option to aggregate feedback
            if self.n_actions is not None:
                self.simulated_feedback = self.simulate_feedback_for_discrete_action(obs)
            else:
                self.simulated_feedback = self.simulate_feedback_for_text(obs, world_info)

        # Combined world_info with simulated feedback
        simulated_info = self.simulations_to_text()
        world_and_simulations_info = world_info + " \n " + simulated_info

        user_prompt = self.prompt_template.format(self.docstring, world_and_simulations_info)
        response, _ = self.llm.generate(user_prompt)

        if self.verbose:

            if self.logger is not None:
                self.logger.log(f"User Prompt:\n\n{user_prompt}\n")
                self.logger.log(f"LLM Response:\n\n{response}\n")

            print_color(f'User Prompt:\n\n{user_prompt}\n', "blue")
            print_color(f'LLM Response:\n\n{response}\n', "green")

        action = response.split(self.action_name + ':')[-1]

        if self.n_actions is not None:
            action = extract_action(action, self.n_actions)

        if self.verbose:
            if self.logger is not None:
                self.logger.log(f"Action:\n\n{action}\n")

            print_color(f'Action:\n\n{action}\n', 'red')

        self.history.append({'action': action, 'feedback': None})

        return action

    def paraphrase(self, sentence):
        return self.paraphrase_agent.paraphrase(sentence) if self.paraphrase_agent is not None else sentence

    @property
    def docstring(self):
        return self.paraphrase(self._docstring)
