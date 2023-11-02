"""
Different from loss-descent where we optimize f(x)
Functions in here are in the form of f(g(x))
"""

import re
import sys
import gym
from jax import grad
import jax.numpy as jnp
import numpy as np
from textwrap import dedent, indent

from gym.utils import seeding


def global_callable_func(l2, theta2, l1, theta1, goal_l2, goal_theta2):
    # l2, theta2 = x

    u = l1 * jnp.cos(theta1) + l2 * jnp.cos(theta1 + theta2)
    v = l1 * jnp.sin(theta1) + l2 * jnp.sin(theta1 + theta2)

    u_goal = l1 * jnp.cos(theta1) + goal_l2 * jnp.cos(theta1 + goal_theta2)
    v_goal = l1 * jnp.sin(theta1) + goal_l2 * jnp.sin(theta1 + goal_theta2)

    return jnp.sqrt((u_goal - u) ** 2 + (v_goal - v) ** 2)

def end_position_to_goal(u, v, l1, theta1, goal_l2, goal_theta2):

    u_goal = l1 * jnp.cos(theta1) + goal_l2 * jnp.cos(theta1 + goal_theta2)
    v_goal = l1 * jnp.sin(theta1) + goal_l2 * jnp.sin(theta1 + goal_theta2)

    return jnp.sqrt((u_goal - u) ** 2 + (v_goal - v) ** 2)


# https://www.sfu.ca/~ssurjano/robot.html
class RoboticArmFunction(gym.Env):
    def __init__(self, feedback=0, seed=None, precision_digit=2, horizon=10,
                 intermediate_feedback_only=True):
        # min_y, optimal_sol,
        # callable_func: a function that takes in a list
        # we truncate the floating point precision to 2 decimal places

        # we sample first segment length and angle
        # first segment does not matter
        # we only choose length and angle for the second segment

        # guess work is involved (since goal is unknown)
        # gradient on distance over (u, v) will help

        super().__init__()

        self.feedback = feedback
        assert self.feedback in {0, 0.5, 1}

        self.action_space = gym.spaces.Text(sys.maxsize)
        self.observation_space = gym.spaces.Text(sys.maxsize)

        self._np_random = None

        # in addition to text, we also allow a few keywords that means
        # the agent wants to terminate the environment
        self.stop_keywords = ['reach', 'stay', 'stop']

        self.prev_x = None
        self.left_attempts = horizon
        # self.min_y = min_y
        # self.optimal_sol = optimal_sol
        self.precision_digit = precision_digit

        self.horizon = horizon

        self.intermediate_feedback_only = intermediate_feedback_only

        self._seed = self.seed(seed)

        # Note: currently we treat the first line as "instruction"
        self.docstring = dedent("""
        You are controlling a robotic arm with two segments.
        Your goal is for the second segment's endopint to reach the goal. 
        
        The first segment is fixed at the origin.
        You can control the second segment, which is attached to the first segment.
        
        The second segment is retractable, so you can control its length.
        The second segment is also rotatable, so you can control its angle.
        
        You get to observe the position of the endopint and distance to the goal.
        You need to minimize the distance to the goal.

        The range of length is [0, 1].
        The range of angle is [0, 6.28].
        
        Do not choose outside of these ranges.

        Choose them within {} attempts.
        You can also choose to stop at any time.

        Output format:
        second segment length = 0.1
        second segment angle = 1.1
        """)

        self.docstring = self.docstring.strip()
        self.docstring = self.docstring.format(self.horizon)

    def get_end_position(self, l2, theta2, round=False):
        l1 = self.l1
        theta1 = self.theta1

        u = l1 * jnp.cos(theta1) + l2 * jnp.cos(theta1 + theta2)
        v = l1 * jnp.sin(theta1) + l2 * jnp.sin(theta1 + theta2)

        if round:
            u, v = float(u), float(v)
            u = np.round(u, self.precision_digit)
            v = np.round(v, self.precision_digit)

        return (u, v)

    def get_optimal_solution(self):
        return (self.goal_l2, self.goal_theta2)

    def reset(self, **kwargs):
        # we sample the initial state from the uniform distribution
        # x = self.np_random.uniform(self.x_low, self.x_high, size=2)
        self.l1 = self.np_random.uniform(0, 1)
        self.theta1 = self.np_random.uniform(0, 2 * np.pi)

        l2 = 0.1
        theta2 = self.np_random.uniform(0, 2 * np.pi)

        self.l1 = np.round(self.l1, self.precision_digit)
        self.theta1 = np.round(self.theta1, self.precision_digit)
        theta2 = np.round(theta2, self.precision_digit)

        # we sample the goal position
        self.goal_l2 = self.np_random.uniform(0, 1)
        self.goal_theta2 = self.np_random.uniform(0, 2 * np.pi)

        # we round the floating point precision to 2 decimal places
        x = (l2, theta2)
        self.prev_x = x

        y = global_callable_func(l2, theta2, self.l1, self.theta1, self.goal_l2, self.goal_theta2)

        self.left_attempts = self.horizon

        u, v = self.get_end_position(l2, theta2, round=True)
        y = np.round(y, self.precision_digit)

        obs = "second segment length = {}\nsecond segment angle = {}\nSecond segment endpoint = {}\nDistance to goal = {:.2f}\n\nYou have {} attempts in total!".format(
            l2, theta2,
            (u, v), y,
            self.left_attempts)
        # obs += '\n\nChoose the next length and angle for the second robotic arm segment to minimize the distance to the goal:'

        # setup the function
        theta1 = self.theta1
        l1 = self.l1
        goal_l2 = self.goal_l2
        goal_theta2 = self.goal_theta2

        u_goal = l1 * np.cos(theta1) + goal_l2 * np.cos(theta1 + goal_theta2)
        v_goal = l1 * np.sin(theta1) + goal_l2 * np.sin(theta1 + goal_theta2)

        # def callable_func(l2, theta2):
        #
        #     u = l1 * jnp.cos(theta1) + l2 * jnp.cos(theta1 + theta2)
        #     v = l1 * jnp.sin(theta1) + l2 * jnp.sin(theta1 + theta2)
        #
        #     return jnp.sqrt((u_goal - u) ** 2 + (v_goal - v) ** 2)

        callable_func = lambda l2, theta2: jnp.sqrt(
            (u_goal - (l1 * jnp.cos(theta1) + l2 * jnp.cos(theta1 + theta2))) ** 2 + (
                        v_goal - (l1 * jnp.sin(theta1) + l2 * jnp.sin(theta1 + theta2))) ** 2)

        end_position_to_goal = lambda u, v: jnp.sqrt((u_goal - u) ** 2 + (v_goal - v) ** 2)

        self.distance_to_input_grad_0 = grad(callable_func, argnums=0)
        self.distance_to_input_grad_1 = grad(callable_func, argnums=1)

        self.distance_to_endpoint_grad_0 = grad(end_position_to_goal, argnums=0)
        self.distance_to_endpoint_grad_1 = grad(end_position_to_goal, argnums=1)

        self.callable_func = callable_func
        self.end_position_to_goal = end_position_to_goal

        return obs

    def seed(self, seed=None):
        """Seed the PRNG of this space and possibly the PRNGs of subspaces."""
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def np_random(self):
        """Lazily seed the PRNG since this is expensive and only needed if sampling from this space."""
        if self._np_random is None:
            self.seed()
        return self._np_random  # type: ignore  ## self.seed() call guarantees right type.

    def text_extract(self, text):
        # return np.array([x1, x2]), agent decides to stop
        for stop_word in self.stop_keywords:
            if stop_word in text:
                return None, True

        pattern = re.compile(r'\b\d+\.?\d*\b')
        numbers = pattern.findall(text)
        if len(numbers) == 0:
            return None, False
        else:
            numbers = [float(g) for g in numbers]
            return np.array(numbers), False

    def step(self, action):
        # observation, reward, terminal, info
        didactic_feedback_dict = {}
        self.left_attempts = self.left_attempts - 1

        x, stop = self.text_extract(action)
        if x is None and stop is False:
            raise ValueError(f'Invalid action: {action}')

        if stop:
            return None, global_callable_func(self.prev_x[0], self.prev_x[1], self.l1, self.theta1, self.goal_l2, self.goal_theta2), True, {}

        dist = global_callable_func(x[0], x[1], self.l1, self.theta1, self.goal_l2, self.goal_theta2)
        u, v = self.get_end_position(x[0], x[1], round=True)

        dist = np.round(float(dist), self.precision_digit)

        if np.abs(dist - 0) < 1e-2:
            # r_pos
            didactic_feedback_dict['r_pos'] = 'You have reached the minimum!'
            return "Second segment endpoint = {}\nDistance to goal = {}\nYou have reached the goal!".format((u, v),
                                                                                                            0), 0, True, {
                'feedback': 'You have reached the goal!'}

        # r_neg
        u_str, v_str = np.round(float(u), self.precision_digit), np.round(float(v), self.precision_digit)
        obs = "Second segment endpoint = {}\nDistance to goal = {}\nYou have {} attempts left!".format((u_str, v_str),
                                                                                                       dist,
                                                                                                       self.left_attempts)
        obs += '\n\nChoose the next length and angle for the second robotic arm segment to minimize the distance to the goal:'

        feedback = "The second segment endpoint has not reached the goal yet. Keep going!\n"  # y is not minimized yet. Keep going!

        if self.feedback != 0:
            if self.intermediate_feedback_only:
                # calculate gradients with respect to u, v
                # say how u, v should change to minimize y
                # dx = self.distance_to_endpoint_grad(u, v, self.l1, self.theta1, self.goal_l2, self.goal_theta2)
                dx1 = self.distance_to_endpoint_grad_0(u, v)
                dx2 = self.distance_to_endpoint_grad_1(u, v)

                u, v = self.get_end_position(x[0], x[1], round=True)

                if self.feedback == 0.5:
                    feedback += '\n\n'
                    if np.abs(dx1) > np.abs(dx2):
                        feedback += f"The current endpoint of the second arm is {(u, v)}. However, change arm's length and angle to alter the first coordinate {u} will get you closer to the goal."
                    else:
                        feedback += f"The current endpoint of the second arm is {(u, v)}. However, change arm's length and angle to alter the second coordinate {v} will get you closer to the goal."
                elif self.feedback == 1:
                    feedback += '\n\n'
                    x1_direction = 'smaller' if dx1 > 0 else 'larger'  # take the opposite of gradient
                    x2_direction = 'smaller' if dx2 > 0 else 'larger'
                    feedback += f"The current endpoint of the second arm is {(u, v)}. Choose arm's length and angle to make the first coordinate {u} {x1_direction} will get you closer to the goal."
                    feedback += f"The current endpoint of the second arm is {(u, v)}. Choose arm's length and angle to make the second coordinate {v} {x2_direction} will get you closer to the goal."
            else:
                # calculate gradients w.r.t. l2, theta2
                # say how l2, theta2 should change to minimize y
                # dx = self.distance_to_input_grad(x[0], x[1], self.l1, self.theta1, self.goal_l2, self.goal_theta2)
                dx1 = self.distance_to_input_grad_0(x[0], x[1])
                dx2 = self.distance_to_input_grad_1(x[0], x[1])
                # dx1, dx2 = dx[0], dx[1]

                if self.feedback == 0.5:
                    feedback += '\n\n'
                    if np.abs(dx1) > np.abs(dx2):
                        feedback += f"You chose {action}\nHowever, try a different arm segment length {x[0]} will get you closer to the goal."
                    else:
                        feedback += f"Try a different arm segment angle {x[1]} will get you closer to the goal."
                elif self.feedback == 1:
                    feedback += '\n\n'
                    x1_direction = 'smaller' if dx1 > 0 else 'larger'  # take the opposite of gradient
                    x2_direction = 'smaller' if dx2 > 0 else 'larger'
                    feedback += f"You chose {action}\nOutput a {x1_direction} arm segment length than {x[0]} to get closer to the goal.\n"
                    feedback += f"Output a {x2_direction} arm segment angle than {x[1]} to get closer to the goal."

        self.prev_x = x
        return obs, -dist, False, {'feedback': feedback, 'didactic_feedback': None}
