# Loss Optimization

Library dependency:
```bash
pip install jax
```

The problems are taken from [Virtual Library of Simulation Experiments: Test Functions and Datasets](https://www.sfu.ca/~ssurjano/index.html).

We took the following functions from the [Optimization Test Problems](https://www.sfu.ca/~ssurjano/optimization.html):

All problems specify that x is two-dimensional.

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

It is easy to add any function to the wrapper class we provide.

This is the specification for the environment:
- Starting state: task description, and a randomly sampled starting point x and f(x).
- Reward signal: -f(x)
- Feedback:
  - Directional: the direction of the gradient of f(x) at x.
  - Non-directional feedback: Compare partial derivative of f(x) to two dimensions of x -- and provide information on which dimension should be changed for faster descend.
