# LLF-Bench: Benchmark for Interactive Learning from Language Feedback

LLF Bench is a benchmark that provides a diverse collection of interactive learning problems where the agent gets language feedback instead of rewards (as in RL) or action feedback (as in imitation learning). The associated website and paper are:

**Website:** https://microsoft.github.io/LLF-Bench/

**Paper:** https://microsoft.github.io/LLF-Bench/llf_bench_paper.pdf

<img src="https://microsoft.github.io/LLF-Bench/images/llf-bench.png" width="750">

Each benchmark environment here follows the gym api.

    observation_dict, info = env.reset()
    observation_dict, reward, terminated, truncated, info = env.step(action)

`observation_dict` contains three fields:

- 'observation': a (partial) observation of the environment's state
- 'instruction': a natural language description of the task, including the objective and information about the action space, etc.
- 'feedback':  a natural language feedback to help the agent to better learn to solve the task.
When a field is missing, its value is represented as None. For example, 'instruction' is typically only given by `reset` whereas 'feedback' is only given by `step`.

`reward` is intended for evaluating an agent's performance. It should **not** be passed to the learning agent.

`terminated` indicates whether a task has been solved (i.e., the goal has been reached) or not.
`truncated` indicates whether the maximal episode length has been reached.
`info` returns an additional info dict of the environment.


## Principle

We design LLF-Bench as a benchmark to test the **learning** ability of interactive agents. We design each environment in LLF-Bench such that, from 'observation' and 'instruction' in `observation_dict`, it is sufficient (for a human) to tell when the task is indeed solved. Therefore, a policy that operates based purely on 'observation' and 'instruction' can solve these problems. However, we also design these environments such that 'observation' and 'instruction' are not sufficient for designing or efficiently learning the optimal policies. Each environment here is designed to have some ambiguities and latent characteristics in the dynamics, reward, or termination, so that the agent cannot infer the optimal policy just based on 'instruction'. 

These features are designed to test an agent's *learning* ability, especially, the ability to learn from language feedback. Language feedback can be viewed as a generalization of reward feedback in reinforcement learning. It can not only provide information about reward/success, but it can also convey expressive feedback such as explanations and suggestions. The language feedback is implemented as the field 'feedback' in `observation_dict`, which is to help the agent to learn better.


## Installation

Create conda env.

    conda create -n LLF-Bench python=3.8 -y
    conda activate LLF-Bench

Install the repo.

    pip install -e .
or
    pip install -e .[#option1,#option2,etc.]

Some valid options:

    metaworld: for using metaworld envs
    alfworld: for using alfworld envs

For example, to use metaworld, install the repo by `pip install -e.[metaworld]`.

Note that the `alfworld` option requires building/compiling from source files. Please ensure that your development environment has the appropriate utilities for C/C++ development (cmake, C compiler, etc.). On Linux or WSL, this can be accomplished by

    sudo apt-get update
    sudo apt-get install cmake build-essential

In addition, `alfworld` currently requires an older gym version. After installing llfbench, please downgrade the gym version to 0.15.4 by running

    pip install gym==0.15.4


For `metaworld` option, it requires libGL, which can be installed by

    sudo apt-get install ffmpeg libsm6 libxext6

For `optimization` option, it requires Jax, which can be installed by

    pip install jax jaxlib

For `reco` option, please follow the instruction here to register and get your own user key:

https://www.omdbapi.com/apikey.aspx

Then, you can set the environment variable `OMDB_API_KEY` to your key:
```bash
export OMDB_API_KEY=your_key
```

## Examples

This sample code creates an environment implemented in LLF-Bench, and creates an agent that interacts with it. The agent simply prints each observation to the console and takes console input as actions to be relayed to the environment.

```python
import llfbench as gym

# Environments in the benchmark are registered following
# the naming convention of llf-*

env = gym.make('llf-Gridworld-v0')

done = False
cumulative_reward = 0.0

# First observation is acquired by resetting the environment

observation = env.reset()

while not done:
    # Observation is dict having 'observation', 'instruction', 'feedback'
    # Here we print the observation and ask the user for an action

    action = input( observation['observation'] + '\n' +
                    observation['instruction'] + '\n' +
                    observation['feedback'] + '\n' +
                    'Action: ' )

    # Gridworld has a text action space, so TextWrapper is not needed
    # to parse a valid action from the input string

    observation, reward, terminated, truncated, info = env.step(action)

    # reward is never revealed to the agent; only used for evaluation

    cumulative_reward += reward

    # terminated and truncated follow the same semantics as in Gymnasium

    done = terminated or truncated

print(f'Episode reward: {cumulative_reward}')
```


## Testing

The `tests` folder in the repo contains a few helpful scripts for testing the functionality of LLF-Bench. 
- *test_agents.py*: Creates a `UserAgent` that prints the 'observation' and 'feedback' produced by an LLF-Bench environment to the console, and reads user input from the console as an 'action'.
- *test_basic_agents.py*: For a subset of LLF-Bench environments that support either a finite action space or admit a pre-built expert optimal policy, this script creates a `RandomActionAgent` and `ExpertActionAgent` to test supported LLF-Bench environments.
- *test_envs.py*: Syntactically tests environments added to the LLF-Bench environment registry so as to be compatible with the expected semantics of LLF-Bench. This is a useful script to run on any new environments that are added or existing environments are customized in the benchmark.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
