from gym.envs.registration import register

register(
    id='gridworld',
    entry_point='verbal_gym.envs.gridworld.randomized:RandomizedGridworld',
)
