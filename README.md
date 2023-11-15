# Verbal-Gym: A benchmark for evaluating learning agents based on verbal feedback

This repository provides a collection of benchmarks for evaluating agents that learn from verbal feedback.

Each benchmark environment here is *goal-oriented* and follows the gym api.

    observation_dict, info = env.reset()
    observation_dict, reward, terminated, truncated, info = env.step(action)

`observation_dict` contains three fields:

- 'observation': a (partial) observation of the environment's state
- 'instruction': a natural language description of the task, including the objective and information about the action space, etc.
- 'feedback':  a natural language feedback to help the agent to better learn to solve the task.
When a field is missing, its value is represented as None. For example, 'instruction' is only given by `reset` whereas 'feedback' is only given by `step`.

`reward` is intended for evaluating an agent's performance. It should **not** be passed to the learning agent.

`terminated` indicates whether a task has been solved (i.e., the goal has been reached) or not.
`truncated` indicates whether the maximal episode length has been reached.
`info` returns an additional info dict of the environment.


## Principle

We design verbal-gym as a benchmark to test the "learning" ability of interactive agents.

We design each environment in verbal-gym such that, from 'observation' and 'instruction' in `observation_dict`, it is suffcient (for a human) to tell that the agent has reached the goal when the task is indeed solved. Therefore, a policy that operates based purely on 'observation' and 'instruction' can solve these problems.

However, we also design these environments such that 'observation' and 'instruction' are not suffcient for designing or *efficiently* learning the goal-reaching policies. Each environment here is designed to have some ambiguities and latent characteristics in the dynamics, reward, termination, so that the agent cannot figure out the optimal policy just based on 'instruction' without learning. In addition, since 'observation' and 'instruction' together only provides sparse informaiton about success, learning the optimal policy based on them can be exponentially hard.

These features are designed to test an agent's *learning* ability, especially, the ability to learn from verbal feedback. Verbal feedback is a generalization of reward in reinforcement learning. It can provide information about reward/success, but it can also convey more expressive feedback such as explanations and suggestions. The verbal feedback is implemented as the field 'feedback' in `observation_dict`, which is an accelerator to help learning the policy faster.





## Installation

Create conda env.

    conda create -n verbal-gym python=3.8 -y
    conda activate verbal-gym

Install the repo.

    pip install -e.
or
    pip install -e.[#option1,#option2,etc.]

Some valid options:

    metaworld: for using metaworld envs
    alfworld: for using alfworld envs

For example, to use metaworld, install the repo by `pip install -e.[metaworld]`.

Note that the alfworld option requires building/compiling from source files. Please ensure that your development environment has the appropriate utilities for C/C++ development (cmake, C compiler, etc.). On Linux or WSL, this can be accomplished by

    sudo apt-get update
    sudo apt-get install cmake build-essential

## Examples

TODO



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
