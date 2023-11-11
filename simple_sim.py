"""
A simpler version of the ballistic simulation. Basically this uses a constant
drag coefficient for the projectile and constant air density in the simulation.
Does not need/use the objects and classes and stuff that's used in the "fuller"
simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from sim import speed2velocity, vec_len


def _diff_eq(y0: np.ndarray, t: int | float, a: int | float) -> np.ndarray:
    """
    Differential equation for an object following ballistic trajectory
    :param y0:
    :param t:
    :param a:
    :return:
    """
    _ = t  # Unused variable, this suppresses the warning
    x, v = y0
    dydt = [v, -a * vec_len(v) * v + np.array([0, -9.81])]
    return np.array(dydt)


def rk4(diff_eq: Callable, y0: list | np.ndarray, trange: np.ndarray,
        *args, **kwargs) -> np.ndarray:
    """
    Solves a system of first-order differential equations using the
    classic Runge-Kutta method
    :param diff_eq: Function which returns the righthandside of the
    equations making up the system of equations
    :param y0: Initial values for y and y" (i.e. the terms in the system
    of equations)
    :param trange: Time points for which the equation is solved
    :param args: Any additional paramaters for the differential equation
    :param kwargs:
    :return: A m x n size matrix, where m is the amount of equations
    and n is the amount of timesteps. Contains values for each equation
    at each timestep.
    """
    m = trange.size
    dt = trange[1] - trange[0]
    if not isinstance(y0, np.ndarray):
        y0 = np.array(y0)
    n = y0.shape
    sol = np.zeros((m, *n))
    sol[0, :] = y0
    for i, t in enumerate(trange[1:], start=1):
        y = sol[i - 1, :]
        k1 = diff_eq(y, t, *args, **kwargs)
        k2 = diff_eq(y + dt * k1 / 2, t + dt / 2, *args, **kwargs)
        k3 = diff_eq(y + dt * k2 / 2, t + dt / 2, *args, **kwargs)
        k4 = diff_eq(y + dt * k3, t + dt, *args, **kwargs)
        y += 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        sol[i, :] = y
    return sol


def simulate(p0: np.ndarray, v0: np.ndarray, tspan: np.ndarray, m: int | float,
             c_d: int | float, area: int | float) -> np.ndarray:
    """
    :param p0:
    :param v0:
    :param tspan:
    :param m:
    :param c_d:
    :param area:
    :return:
    """
    y0 = np.vstack((p0, v0))
    rho = 1.2  # Density of air at sea level or thereabouts [kg*m^-3]
    a = 1 / (2 * m) * rho * area * c_d
    return rk4(diff_eq=_diff_eq, y0=y0, trange=tspan, a=a)


def main() -> None:
    # Initial settings
    m = 3.7  # Mass of the projectile [kg]
    c_d = .23  # Drag coefficient of the projectile [-]
    area = np.pi * (.057 / 2) ** 2  # Projected area of the projectile [m^2]
    dt = .01  # Timestep [s]
    end = 80  # Ending time (duration of time to simulate) [s]
    tspan = np.linspace(0, end, int(end / dt) + 1)  # [s]
    p0 = np.array([0, 0])
    v0_mag = 1035
    angle = 45
    v0 = speed2velocity(v0=v0_mag, angle=angle)

    # Run the simulation and show the results
    sol = simulate(p0=p0, v0=v0, tspan=tspan, m=m, c_d=c_d, area=area)
    pos = sol[:, 0]
    vel = sol[:, 1]

    _ = plt.figure(figsize=(6, 5))
    plt.plot(pos[:, 0], pos[:, 1])
    plt.title("Position")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid()

    _ = plt.figure(figsize=(6, 5))
    plt.title("Velocities")
    plt.plot(tspan, vel[:, 0], label="x velocity")
    plt.plot(tspan, vel[:, 1], label="y velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
