"""
Some solvers to solve the differential equation of motion for the projectile on
a ballistic trajectory.

Most if not all of these can also be found in:
https://github.com/denzelliuku/slow_math/blob/master/numeric_methods/diff_eqs.py
"""

import numpy as np

from typing import Any, Callable


def rk4(diff_eq: Callable, y0: list | np.ndarray, t: int | float, dt: int | float,
        *args: Any, **kwargs: Any) -> np.ndarray:
    """
    Solves a system of first-order differential equations using the
    classic Runge-Kutta method
    :param diff_eq: Function which returns the righthandside of the
    equations making up the system of equations
    :param y0: Initial values for y and y" (i.e. the terms in the system
    of equations)
    :param t:
    :param dt:
    :param args: Any additional paramaters for the differential equation
    :param kwargs:
    :return: A m x n size matrix, where m is the amount of equations
    and n is the amount of timesteps. Contains values for each equation
    at each timestep.
    """
    if not isinstance(y0, np.ndarray):
        y0 = np.array(y0)
    y = y0
    k1 = diff_eq(y, t, *args, **kwargs)
    k2 = diff_eq(y + dt * k1 / 2, t + dt / 2, *args, **kwargs)
    k3 = diff_eq(y + dt * k2 / 2, t + dt / 2, *args, **kwargs)
    k4 = diff_eq(y + dt * k3, t + dt, *args, **kwargs)
    y += 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return y
