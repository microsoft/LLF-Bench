# verbal-gym

A collecation of gym environemnts with verbal feedback.

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

    export AZURE_OPEANAI_KEY = <azure endpoint key>  # if using azure endpoint
    export OPEANAI_KEY_PATH = <a file containing openai key>  # if using openai endpoint directly

An example command using the openai endpoint would be

     python scripts/basic_agent.py  --n_workers 0  --env_name verbal-SyllableConstrainedPoem-v0  --verbose  --model gpt-3.5-turbo

To use the azure endpoint, add `azure:` as prefix to the model name, e.g., `azure:gpt-35-turbo`. Set `n_workers` to be >1 to use ray to parallelize the evaluation.