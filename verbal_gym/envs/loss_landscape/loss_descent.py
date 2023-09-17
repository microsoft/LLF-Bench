import re
import sys
import gym
from jax import grad
import jax.numpy as jnp
import numpy as np
from textwrap import dedent, indent

from gym.utils import seeding

class LossLandscapeBase(gym.Env):
    def __init__(self, callable_func, x_low, x_high, min_y, optimal_sol,
                 feedback=0, seed=None, precision_digit=2, horizon=10):
        # callable_func: a function that takes in a list
        # we truncate the floating point precision to 2 decimal places

        super().__init__()
        self.x_low = x_low
        self.x_high = x_high

        self.feedback = feedback
        assert self.feedback in {0, 0.5, 1}

        self.action_space = gym.spaces.Text(sys.maxsize)
        self.observation_space = gym.spaces.Text(sys.maxsize)

        self._np_random = None

        # in addition to text, we also allow a few keywords that means
        # the agent wants to terminate the environment
        self.stop_keywords = ['reach', 'stay', 'stop']

        self.callable_func = callable_func
        self.grad_func = grad(callable_func)

        self.prev_x = None
        self.left_attempts = horizon
        self.min_y = min_y
        self.optimal_sol = optimal_sol
        self.precision_digit = precision_digit

        self.horizon = horizon

        self._seed = self.seed(seed)

    def get_optimal_solution(self):
        return self.optimal_sol

    def reset(self, **kwargs):
        # we sample the initial state from the uniform distribution
        x = self.np_random.uniform(self.x_low, self.x_high, size=2)
        # we round the floating point precision to 2 decimal places
        x = np.round(x, self.precision_digit)
        self.prev_x = x

        self.left_attempts = self.horizon

        self.task_description = dedent("""
        You are trying to minimize the output (y) of a function by choosing input (x).
        You get to observe y once you choose the value of x, where x is a 2-dimensional vector.
        This means x = [x1, x2], where x1 and x2 are real numbers.
        The goal is to choose x such that y is as small as possible.

        The range of x1 and x2 is [{}, {}].
        Please do not choose x outside of this range.
        
        You are starting at x = [{}, {}].
        The output of the function at this point is y = {}.
        
        Choose x within {} attempts.
        You can choose to stop at any time.
        
        Output format:
        x = [x1, x2]
        
        Output the next x that will make this function output the smallest y.
        x =
        """)
        self.task_description = self.task_description.strip()
        y = self.callable_func(x)
        self.task_description = self.task_description.format(self.x_low, self.x_high, x[0], x[1], y,
                                                             self.horizon)

        return self.task_description

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

        pattern = r'\[(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]'
        match = re.search(pattern, text)
        if match is None:
            return None, False
        else:
            numbers = [float(g) for g in match.groups()]
            return np.array(numbers), False

    def step(self, action):
        # observation, reward, terminal, info

        x, stop = self.text_extract(action)
        if x is None and stop is False:
            raise ValueError(f'Invalid action: {action}')

        if stop:
            return None, self.callable_func(self.prev_x), True, {}

        loss = self.callable_func(x)

        if np.abs(loss - self.min_y) < 1e-2:
            return "Function outputs y: {}\nYou have reached the minimum!".format(self.min_y), -self.min_y, True, {'feedback': 'You have reached the minimum!'}

        obs = "Function outputs y = {}\nYou have {} attempts left!".format(loss, self.left_attempts)
        feedback = "y is not minimized yet. Keep going!"

        if self.feedback != 0:
            dx = self.grad_func(x)
            dx1, dx2 = dx[0], dx[1]

        if self.feedback == 0.5:
            feedback += '\n\n'
            if np.abs(dx1) > np.abs(dx2):
                feedback += "Changing x1 (the first dimension of x) will decrease y faster than changing x2."
            else:
                feedback += "Changing x2 (the second dimension of x) will decrease y faster than changing x1."
        elif self.feedback == 1:
            feedback += '\n\n'
            x1_direction = 'decrease' if dx1 > 0 else 'increase' # take the opposite of gradient
            x2_direction = 'decrease' if dx2 > 0 else 'increase'
            feedback += f"You should {x1_direction} x1 (the first dimension of x) to decrease y.\n"
            feedback += f"You should {x2_direction} x2 (the second dimension of x) to decrease y."

        self.prev_x = x
        self.left_attempts -= 1
        return obs, -loss, False, {'feedback': feedback}

# now we wrap all loss functions by inheriting this class
"""
Bowl-Shaped functions:
- [Bohachevsky Functions](https://www.sfu.ca/~ssurjano/boha.html)
- [Rotated Hyper-Ellipsoid Function](https://www.sfu.ca/~ssurjano/rothyp.html)

Plate-Shaped functions:
- [Booth Function](https://www.sfu.ca/~ssurjano/booth.html)
- [Matyas Function](https://www.sfu.ca/~ssurjano/matya.html)
- [McCormick Function](https://www.sfu.ca/~ssurjano/mccorm.html)

Valley shaped functions:
- [Rosenbrock Function](https://www.sfu.ca/~ssurjano/rosen.html)
- [Six-Hump Camel Function](https://www.sfu.ca/~ssurjano/camel6.html)
- [Three-Hump Camel Function](https://www.sfu.ca/~ssurjano/camel3.html)
"""

class Bohachevsky(LossLandscapeBase):
    def __init__(self, func_choice=1, feedback=0, seed=None, horizon=10):
        assert func_choice in [1, 2, 3], "func_choice must be 1, 2, or 3"
        if func_choice == 1:
            func = lambda x: x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * jnp.cos(3 * jnp.pi * x[0]) - 0.4 * jnp.cos(4 * jnp.pi * x[1]) + 0.7
        elif func_choice == 2:
            func = lambda x: x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * jnp.cos(3 * jnp.pi * x[0]) * jnp.cos(4 * jnp.pi * x[1]) + 0.3
        else:
            func = lambda x: x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * jnp.cos(3 * jnp.pi * x[0] + 4 * jnp.pi * x[1]) + 0.3

        super().__init__(callable_func=func,
                            x_low=-100, x_high=100, min_y=0, optimal_sol=np.zeros(2),
                            feedback=feedback, seed=seed, horizon=horizon, precision_digit=4)

class RotatedHyperEllipsoid(LossLandscapeBase):
    def __init__(self, feedback=0, seed=None, horizon=10):
        func = lambda x: x[0] ** 2 + (x[0] ** 2 + x[1] ** 2)
        super().__init__(callable_func=func,
                            x_low=-65.536, x_high=65.536, min_y=0, optimal_sol=np.zeros(2),
                            feedback=feedback, seed=seed, horizon=horizon)

class Booth(LossLandscapeBase):
    def __init__(self, feedback=0, seed=None, horizon=10):
        func = lambda x: (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
        super().__init__(callable_func=func,
                            x_low=-10, x_high=10, min_y=0, optimal_sol=np.array([1, 3]),
                            feedback=feedback, seed=seed, horizon=horizon)

class Matyas(LossLandscapeBase):
    def __init__(self, feedback=0, seed=None, horizon=10):
        func = lambda x: 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]
        super().__init__(callable_func=func,
                            x_low=-10, x_high=10, min_y=0, optimal_sol=np.zeros(2),
                            feedback=feedback, seed=seed, horizon=horizon, precision_digit=4)

class McCormick(LossLandscapeBase):
    def __init__(self, feedback=0, seed=None, horizon=10):
        func = lambda x: np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1
        super().__init__(callable_func=func,
                            x_low=-1.5, x_high=4, min_y=-1.9133, optimal_sol=np.array([-0.54719, -1.54719]),
                            feedback=feedback, seed=seed, horizon=horizon, precision_digit=4)

class Rosenbrock(LossLandscapeBase):
    def __init__(self, a=1, b=100, feedback=0, seed=None, horizon=10):
        # https://en.wikipedia.org/wiki/Rosenbrock_function
        # all of them are lambda functions that expect Numpy array of shape (2,)
        two_dim_rosenbrock = lambda x: (a - x[0])**2 + b * (x[1] - x[0]**2)**2
        super().__init__(callable_func=two_dim_rosenbrock,
                         x_low=-5, x_high=10, min_y=0, optimal_sol=np.ones(2),
                         feedback=feedback, seed=seed, horizon=horizon)

class SixHumpCamel(LossLandscapeBase):
    def __init__(self, feedback=0, seed=None, horizon=10):
        func = lambda x: (4 - 2.1 * x[0]**2 + (x[0]**4)/3) * x[0]**2 + x[0] * x[1] + (-4 + 4 * x[1]**2) * x[1]**2
        # note that SixHumpCamel has two global minima
        # also the range on x is x1 = [-3, 3], x2 = [-2, 2]
        # but we use x1 = [-2, 2], x2 = [-3, 3] for simplicity
        super().__init__(callable_func=func,
                            x_low=-2, x_high=2, min_y=-1.0316, optimal_sol=[np.array([0.0898, -0.7126]), np.array([-0.0898, 0.7126])],
                            feedback=feedback, seed=seed, horizon=horizon, precision_digit=4)

class ThreeHumpCamel(LossLandscapeBase):
    def __init__(self, feedback=0, seed=None, horizon=10):
        func = lambda x: 2 * x[0]**2 - 1.05 * x[0]**4 + (x[0]**6)/6 + x[0] * x[1] + x[1]**2
        super().__init__(callable_func=func,
                            x_low=-5, x_high=5, min_y=0, optimal_sol=np.array([0, 0]),
                            feedback=feedback, seed=seed, horizon=horizon, precision_digit=4)