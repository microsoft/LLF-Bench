
# This file contains the prompts for the verbal instructions and feedback.

highway_instruction = (
    "Your goal is to control a vehicle to park in a desired location, while ensuring that it does not collide with any obstacles or other vehicles. "+
    "You will receive the observation of the vehicle's state, the world state, as well as the desired parking location represented by a json string. "+
    "Your action is a 2-dim vector, where the first dimension controls the throttle input, and the last dimension controls the steering input. "+
     "Throttle is a number between -5 and 5, representing acceleration in units of m/s^2. "+
     "Steering is a number between -pi/4 and pi/4, representing the steering angle in radians.",
)

b_instruction = (
    "Output a good action in the form of [throttle input, steering input].",
    "Render a proper action in the style of [throttle input, steering input].",
    "Supply a good action in the form of [throttle input, steering input].",
    "Indicate a good action using [throttle input, steering input] as the format.",
    "Come up with a good action in the form of [throttle input, steering input].",
    "Present a correct action in the form of [throttle input, steering input].",
)

r_feedback = (
    "You received a reward of {reward}.",
    "Your reward is {reward}.",
    "The reward you get is {reward}.",
    "You get a reward of {reward}.",
    "The reward is {reward}.",
    "You have obtained a reward of {reward}.",
)

hp_feedback = (
    "This is the best action, as it leads to the highest expected reward.",
)

hn_feedback = (
    "This is a bad action, because the vehicle crashed.",
    "This action is unfavorable, given that the vehicle crashed.",
    "This action is problematic, because it caused the vehicle to crash.",
    "This is an inappropriate action, as it resulted in the vehicle crashing.",
)
