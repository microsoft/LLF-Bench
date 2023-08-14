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
    pip install -e.[#BENCHMARK]

where #BENCHMARK denotes the name of the benchmark (which requires separate installation steps, due to version conflicts.)
