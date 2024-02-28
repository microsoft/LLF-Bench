import copy
import pdb
import random
import re
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import autogen
import gymnasium
from autogen import Agent, AssistantAgent, OpenAIWrapper, UserProxyAgent
from termcolor import colored

import llfbench as gym

try:
    config_list_gpt3 = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-3.5-turbo", "gpt-35-turbo"],
        },
    )
except:
    config_list_gpt3 = []

def extract_action(msg: str, action_space: list) -> str:
    for regex in [r"action:\s*(\d+)", r"action\s*(\d+)", r"\d+"]:
        action_substr = re.findall(regex, msg.lower(), re.DOTALL)
        if len(action_substr):
            action = action_substr[0].strip().rstrip()
            break
        else:
            action = msg

    if not isinstance(action_space, gymnasium.spaces.Discrete):
        raise NotImplementedError("Only Discrete action space is supported.")
        # TODO: support other types of action space
        return msg
    
    action_space = list(range(action_space.n))
    if action in action_space:
        return action

    # If we cannot extract the action,
    # Find the one in the action space
    first_action, first_action_idx = None, 1e10
    for valid_action in action_space:
        _idx = action.find(str(valid_action))
        if _idx >= 0 and _idx < first_action_idx:
            first_action = valid_action
            first_action_idx = _idx

    if first_action is not None:
        return first_action

    return random.choice(action_space)    # we are unable to extract action


class EnvAgent(UserProxyAgent):

    def __init__(
        self,
        env: object,
        name: str,
        with_history_in_state: bool = False,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        system_message: Optional[Union[str, List]] = "",
        description: Optional[str] = None,
    ):

        self._env = env    # store the environment
        self._with_history_in_state = with_history_in_state
        super().__init__(name=name,
                         system_message=system_message,
                         is_termination_msg=is_termination_msg,
                         code_execution_config=None,
                         default_auto_reply=default_auto_reply,
                         description=description,
                         max_consecutive_auto_reply=None)
        self._reply_func_list = []
        self.register_reply([Agent, None], EnvAgent._generate_env_reply)
        self.reset()

    def reset(self):
        observation, info = env.reset()
        self.clear_history()
        self._observation = observation
        self._total_reward = 0
        self._steps = 0
        return observation, info

    @property
    def state(self) -> str:
        return f"""
Observation: {self._observation['observation']}
Instruction: {self._observation['instruction']}
Feedback: {self._observation['feedback']}
---

Now, give me the action: """

    def _generate_env_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Generate a reply using autogen.oai."""

        if messages is None:
            messages = self._oai_messages[sender]

        if not self._with_history_in_state:
            # markov, clear previous history
            sender.clear_history()
        message = messages[-1]    
        # for environment, only the last action matters.
        
        if message["content"].strip().rstrip() == "":
            return True, self._env.prompt # API error. let's redo the query

        action = extract_action(message["content"],
                                action_space=self._env.action_space)
        
        print("Action:", colored(f"{action}", "blue"))

        observation, reward, done, truncated, info = self._env.step(action)
        self._observation = observation
        
        self._total_reward += reward
        self._steps += 1
        if done:
            reply = "TERMINATE"
        elif truncated:
            reply = "TERMINATE"
        else:
            reply = self.state

        return True, reply

    def final_info(self):
        result = f"Total Reward: {self._total_reward}\n"
        result += f"Num Steps: {self._steps}"
        return result


class Player(AssistantAgent):

    def __init__(
        self,
        name: str = "player",
        system_message: Optional[str] = "Play the game.",
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        code_execution_config: Optional[Union[Dict, Literal[False]]] = False,
        description: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            description=description,
            max_consecutive_auto_reply=None,
            **kwargs,
        )


if __name__ == "__main__":
    env = gym.make('llf-gridworld-v0')
    env_agent = EnvAgent(env=env, name="env_agent", with_history_in_state=True)
    player = Player(llm_config={"config_list": config_list_gpt3, "cache_seed": 32}, 
                    name="player")
    observation, info = env_agent.reset()
    env_agent.initiate_chat(player, message=env_agent.state)
    
    print("Total reward:", env_agent._total_reward)