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