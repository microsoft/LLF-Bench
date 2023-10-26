# verbal-gym

A collection of gym environments with verbal feedback.

On top of the standard gym interface, each envronment has a
`docstring` attribute that describes the environment's problem or prior information in texts. In addition, the `info` dict returned by `step` function has a key 'feedback' that describes the verbal feedback given to the agent.


### Installation

Create conda env.

    conda create -n verbal-gym python=3.8 -y
    conda activate verbal-gym

Install the repo.

    pip install -e.
or
    pip install -e.[#option1,#option2,etc.]

Some valid options:

    ray: for using parallel evaluation.
    openai: for using openai ChatCompletion backend.

For example, to use openai+ray, install the repo by `pip install -e.[openai,ray]`.

### Example scripts

First, set the environment variables.

    export AZURE_OPENAI_KEY = <azure endpoint key>  # if using azure endpoint
    export OPENAI_KEY_PATH = <a file containing openai key>  # if using openai endpoint directly

An example command using the openai endpoint would be

     python scripts/basic_agent.py  --n_workers 0  --env_name verbal-SyllableConstrainedPoem-v0  --verbose  --model gpt-3.5-turbo

To use the azure endpoint, add `azure:` as prefix to the model name, e.g., `azure:gpt-35-turbo`. Set `n_workers` to be >1 to use ray to parallelize the evaluation.


### Benchmark Agents

`scripts/benchmark.py` provides an example code to benchmark multiple agents and or do ablations involving varying hps. It is based on a decorator function `batch_exp`, which can be used to wrap a function to run batch experiments. `batch_exp` can be used to batch run other functions too.

An example command is

    python scripts/benchmark.py --config configs/exp_configs/poem_envs.yaml  --n_workers 5 --log_dir <directory to log the results>



`scripts/benchmark.py` takes a config yaml as input, which specifies the inputs (`agent_config` and `env_name`) to the `run_experiment` method in `scripts/benchmark.py`. `agent_config` config is a dict used by `create_agent`, and `env_name` is the name of the gym environment to create.

The config yaml specifies a list of values that the batch experiment should span. `agent_config` can have arbitrary structures needed in `create_agent`, so long as we keep the values we want to vary as lists. The only difference between `agent_config` in the config yaml and the one input to  `create_agent` is: the `agent_config` in the config yaml should contains a `file` key to specify where a yaml file that specifies the default of the `agent_config` used by `create_agent`, whereas `agent_config` used by `create_agent` has a key `agent_name` instead of `file`. We can add new agents to `create_agent` by using `agent_config` as the way to pass parameters.


## Plot Experimental Results

Once the experiments are done, one can plot the reuslts by

    python analyses/plot.py  <directory where the results of benchmark.py are logged.>

We can specify the title of each subplot by passing `--plot_name`. E.g., `--plot_name  env_config:env_name+env_config:feedback` means each subplot would be defined by a combination of `config['env_config']['feedback']`, where `+` is the connector and `:` is the separator to read values from the saved config. Similarly, we can specify what goes into each subplot by specifying `--legend_name` using the same syntax.


## Create LLM

Please use `make_llm(model_name, system_prompt=<system promopt>)` in `verbal_gym.llm` to create LLMs, where `model_name` is `<backend>:<model>`, where `<backend>` can be 'gcr', 'azure', 'openai', and `<model>` is something like gpt-3, gpt-35-turbo, gpt-4.

To use 'azure' backend, one needs to set the environment variable `AZURE_OPENAI_KEY`.
To use 'openai' backend, one needs to set the environment variable `OPENAI_KEY_PATH`. (The path to a file containing the openai key).
To use 'gcr' backend, one needs to set the environment variables `GCR_GPT_KEY` and `GCR_GPT_URL`.


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