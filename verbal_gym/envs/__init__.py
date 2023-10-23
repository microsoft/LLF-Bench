from verbal_gym.envs import gym_bandits
from verbal_gym.envs import summarization
from verbal_gym.envs import poem_env
from verbal_gym.envs import gridworld
from verbal_gym.envs.env_wrapper import VerbalGymWrapper

"""

Naming convention for envs:

[basename]-[instruction_type]-[feedback_type]-[version]

instruction_type:
    - (b) Basic (Goal, Syntax, Action Space)
    - (p) Partial (Basic + Offline data (e.g., observation, action, feedback))
    - (c) Complete (Info sufficient to infer the optimal policy)

feedback_type;
    - (n) none
    - (m) mixture: mix of (r), (hp), (hn), (fp), (fn)
    - (r) reward: textualization of reward
    - (hp) hindsight positive: explaination on why something is correct
    - (hn) hindsight negative: explaination on why something is incorrect
    - (fp) future positive: suggestion of things to do
    - (fn) future negative: suggestion of things to avoid

An example env name is: gridworld-b-fn-v0

"""