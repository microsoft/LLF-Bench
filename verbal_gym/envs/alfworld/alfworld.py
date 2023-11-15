import random
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

from verbal_gym.envs.verbal_gym_env import Feedback


class Alfworld:

    # Basic (b), partial (p), and complete (c)
    INSTRUCTION_TYPES = ('b', 'p', 'c')

    # Feedback type:
    # r: reward
    # hn: hindsight negative
    # hp: hindsight positive
    # fn: future negative
    # fp: future positive
    FEEDBACK_TYPES = ('r', 'hn', 'hp', 'fn', 'fp')

    def __init__(self, instruction_type, feedback_type):

        # load config
        self.config = generic.load_config()
        self.env_type = self.config['env']['type']  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

        # setup environment
        self.env = getattr(environment, self.env_type)(self.config, train_eval='train')
        self.env = self.env.init_env(batch_size=1)

        self.instruction_type = instruction_type
        self.feedback_type = feedback_type

        # Markers
        self.param_docstring = "You are in a house with a variety of objects. {task} You have to take a sequence of " \
                               "actions to full fill it. You will be told at each step, what actions are allowed " \
                               "and you must pick only one of those actions."
        self.docstring = None
        self.last_infos = None

    def _generate_docstring(self, reset_obs):

        # Separate task and use it as docstring
        task = reset_obs.split("\n\n")[-1]
        if not task.endswith("."):
            task = task + "."

        docstring = self.param_docstring.replace("{task}", task)

        return docstring

    def _generate_observation(self, obs, admissible_commands, won):

        actions = ", ".join(admissible_commands)
        obs_command = f"{obs}. You are allowed to take the following actions: {actions}."

        if won:
            obs_command += " Congratulations on solving the task!"
        return obs_command

    def reset(self):
        # Obs is text and info is a dict with the following keys:
        #   'won', 'extra.gamefile', 'expert_type', 'admissible_commands', 'expert_plan'
        obs, infos = self.env.reset()

        # Extract single item from batch
        obs = obs[0]
        won = bool(infos["won"][0])

        self.last_infos = infos
        self.docstring = self._generate_docstring(reset_obs=obs)

        # Create observation by combining current obs with admissible commands
        admissible_commands = infos["admissible_commands"][0]
        obs_command = self._generate_observation(obs=obs,
                                                 admissible_commands=admissible_commands,
                                                 won=won)

        return dict(instruction=self.docstring,
                    observation=obs_command,
                    feedback=None)

    def _generate_feedback(self, action, reward, info, past_info, feedback_type=None):

        if feedback_type is None:
            feedback_type = self.feedback_type

        feedback = Feedback()

        if "r" in feedback_type:
            feedback.r = f"You received a reward of {reward}."

        if "hn" in feedback_type:

            past_admissible_actions = past_info["admissible_commands"][0]
            past_opt_action = past_info["expert_plan"][0][0]

            bad_actions = list(past_admissible_actions)
            bad_actions.remove(past_opt_action)

            avoid_action = random.choice(bad_actions)

            if action == avoid_action:
                feedback.hn = f"You made a mistake by taking the bad action {avoid_action}."
            else:
                feedback.hn = f"You were correct in not taking the bad action {avoid_action}."

        if "hp" in feedback_type:

            past_opt_action = past_info["expert_plan"][0][0].lower().strip()

            if past_opt_action == action.lower().strip():
                feedback.hp = f"You did the right thing by taking the {past_opt_action} action."
            else:
                feedback.hp = f"You should have taken the {past_opt_action} action."

        if "fn" in feedback_type:

            admissible_actions = info["admissible_commands"][0]
            opt_action = info["expert_plan"][0][0]

            bad_actions = list(admissible_actions)
            bad_actions.remove(opt_action)

            avoid_action = random.choice(bad_actions)

            feedback.fn = f"You should not take the action {avoid_action} in this current step."

        if "fp" in feedback_type:

            opt_action = info["expert_plan"][0][0].lower().strip()
            feedback.fp = f"You should now take the {opt_action} action."

        return feedback

    def step(self, action):

        if self.last_infos is None:
            raise AssertionError("Found self.last_infos as None. You must reset before step.")

        if type(action) != str:
            raise TypeError(f"Expected action of type string but found {type(action)}.")

        # TODO: Parse the action later for strict parsing
        # admissible_commands = list(self.last_infos['admissible_commands'][0])

        # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
        # note: BUTLER generates commands word-by-word without using admissible_commands

        # step
        obs, rewards, dones, infos = self.env.step([action])

        # Extract single item from batch
        obs = obs[0]
        reward = rewards[0]
        done = dones[0]
        won = bool(infos["won"][0])

        # Create observation by combining current obs with admissible commands
        admissible_commands = infos["admissible_commands"][0]
        obs_command = self._generate_observation(obs=obs,
                                                 admissible_commands=admissible_commands,
                                                 won=won)

        # Feedback
        feedback = self._generate_feedback(action=action,
                                           reward=reward,
                                           info=self.last_infos,
                                           past_info=infos)

        info = {
            "feedback": feedback
        }

        self.last_infos = infos

        next_packed_obs = dict(instruction=None,
                               observation=obs_command,
                               feedback=feedback)

        return next_packed_obs, reward, done, info
