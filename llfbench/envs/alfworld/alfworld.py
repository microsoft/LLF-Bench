import sys
import random
import string
import gymnasium as gym

from llfbench.envs.alfworld.prompts import *
from llfbench.envs.llf_env import Feedback


class Alfworld(gym.Env):

    # Basic (b), partial (p), and complete (c)
    INSTRUCTION_TYPES = ('b')        # ('b', 'p', 'c')

    # Feedback type:
    # r: reward
    # hn: hindsight negative
    # hp: hindsight positive
    # fn: future negative
    # fp: future positive
    FEEDBACK_TYPES = ('r', 'hn', 'hp', 'fn', 'fp')

    def __init__(self, instruction_type, feedback_type):

        config_file = "llfbench/envs/alfworld/base_config.yaml"

        old_sys_argv = list(sys.argv)
        print(f"Reading file {config_file}")
        sys.argv = [old_sys_argv[0], config_file]

        import alfworld.agents.environment as environment
        import alfworld.agents.modules.generic as generic

        # load config
        self.config = generic.load_config()
        self.env_type = self.config['env']['type']  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

        # setup environment
        self.env = getattr(environment, self.env_type)(self.config, train_eval='train')
        self.env = self.env.init_env(batch_size=1)

        self.format = None
        self.instruction_type = instruction_type
        self.feedback_type = feedback_type

        self.action_space = gym.spaces.Text(sys.maxsize, charset=string.printable)
        self.observation_space = gym.spaces.Text(sys.maxsize, charset=string.printable)

        # TODO find the best way to set the horizon for alfworld. Currently, setting it to a guess value.
        self.horizon = 100
        self.timestep = 0

        # Markers
        self.docstring = None
        self.last_infos = None

        sys.argv = old_sys_argv

    def seed(self, seed):
        self.env.seed(seed)

    def _generate_docstring(self, reset_obs):

        # Separate task and use it as docstring
        task = reset_obs.split("\n\n")[-1]
        if not task.endswith("."):
            task = task + "."

        docstring = self.format(docstrings, task=task)

        return docstring

    def _generate_observation(self, obs, admissible_commands, won):

        actions = ", ".join(admissible_commands)
        obs_command = f"{obs}. You are allowed to take the following actions: {actions}."

        if won:
            obs_command += " Congratulations on solving the task!"
        return obs_command

    def reset(self, *, seed=None, options=None):

        if seed is not None:
            self.seed(seed)

        # Obs is text and info is a dict with the following keys:
        #   'won', 'extra.gamefile', 'expert_type', 'admissible_commands', 'expert_plan'
        obs, infos = self.env.reset()

        # Extract single item from batch
        obs = obs[0]
        won = bool(infos["won"][0])
        self.timestep = 0

        self.last_infos = infos
        self.docstring = self._generate_docstring(reset_obs=obs)

        # Create observation by combining current obs with admissible commands
        admissible_commands = infos["admissible_commands"][0]
        obs_command = self._generate_observation(obs=obs,
                                                 admissible_commands=admissible_commands,
                                                 won=won)

        info = {
            "success": False,
            "expert_action": infos["expert_plan"][0][0]
        }

        return dict(instruction=self.docstring,
                    observation=obs_command,
                    feedback=None), info

    def _generate_feedback(self, action, reward, info, past_info, feedback_type=None):

        if feedback_type is None:
            feedback_type = self.feedback_type

        feedback = Feedback()

        if "r" in feedback_type:
            feedback.r = self.format(reward_descp, reward=reward)

        if "hn" in feedback_type:

            past_admissible_actions = past_info["admissible_commands"][0]
            past_opt_action = past_info["expert_plan"][0][0]

            bad_actions = list(past_admissible_actions)
            bad_actions.remove(past_opt_action)

            avoid_action = random.choice(bad_actions)

            if action == avoid_action:
                feedback.hn = self.format(mistake_bad_action_descp, avoid_action=avoid_action)
            else:
                feedback.hn = self.format(correct_bad_action_descp, avoid_action=avoid_action)

        if "hp" in feedback_type:

            past_opt_action = past_info["expert_plan"][0][0].lower().strip()

            if past_opt_action == action.lower().strip():
                feedback.hp = self.format(correct_good_action_descp, past_opt_action=past_opt_action)
            else:
                feedback.hp = self.format(mistake_good_action_descp, past_opt_action=past_opt_action)

        if "fn" in feedback_type:

            admissible_actions = info["admissible_commands"][0]
            opt_action = info["expert_plan"][0][0]

            bad_actions = list(admissible_actions)
            bad_actions.remove(opt_action)

            avoid_action = random.choice(bad_actions)

            feedback.fn = self.format(avoid_bad_action_descp, avoid_action=avoid_action)

        if "fp" in feedback_type:

            opt_action = info["expert_plan"][0][0].lower().strip()
            feedback.fp = self.format(follow_opt_action_descp, opt_action=opt_action)

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

        self.timestep += 1

        # Extract single item from batch
        obs = obs[0]
        reward = rewards[0]
        done = dones[0]
        won = bool(infos["won"][0])

        terminated = False
        truncated = self.timestep == self.horizon

        # Create observation by combining current obs with admissible commands
        admissible_commands = infos["admissible_commands"][0]
        obs_command = self._generate_observation(obs=obs,
                                                 admissible_commands=admissible_commands,
                                                 won=won)

        # Feedback
        feedback = self._generate_feedback(action=action,
                                           reward=reward,
                                           info=infos,
                                           past_info=self.last_infos)

        info = {
            "feedback": feedback,
            "success": won,
            "expert_action": infos["expert_plan"][0][0]
        }

        self.last_infos = infos

        next_packed_obs = dict(instruction=None,
                               observation=obs_command,
                               feedback=feedback)

        return next_packed_obs, reward, terminated, truncated, info
