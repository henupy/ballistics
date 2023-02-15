"""
Some shiet regarding ballistic trajectories
"""

import numpy as np
import matplotlib.pyplot as plt

from projectiles import ProjectileData, Sphere

# For typing
num = int | float


def g_acc(h: num) -> np.ndarray:
    """
    Calculates the acceleration due to gravital attraction
    between the Earth and the projectile
    :param h: Height of the projectile related to Earth's surface
    :return: Acceleration due to gravity as a vector
    """
    if h == 0:
        return np.array([0, 0])
    big_g = 6.67259e-11
    r_e = 6378.14e3
    m_e = 5.974e24
    return np.array([0, -big_g * m_e / ((r_e + h) * (r_e + h))])


def drag_acc(c_d: num, m_p: num, rho: num, vel: np.ndarray, area: num)\
        -> np.ndarray:
    """
    Calculates the acceleration due to air resistance
    :param c_d: Drag coefficient
    :param m_p: Mass of the projectile
    :param rho: Density of air
    :param vel: Velocity of the projectile as a vector
    :param area: Cross sectional area of the projectile
    :return: Acceleration due to drag as a vector
    """
    return -0.5 * c_d * rho * (vel * vel) * area / m_p


def _simulate(v0: num, alpha: num, dt: num) -> np.ndarray:
    """
    Calculates the trajectory of the projectile without air resistance
    :param v0: Magnitude of the initial velocity
    :param alpha: Elevation angle
    :param dt: Timestep
    :return: Array of x- and y-coordinates
    """
    vel = np.array([v0 * np.cos(alpha), v0 * np.sin(alpha)])
    sol = np.zeros((1, 2))
    x, y = 0, 0
    while True:
        acc_g = g_acc(y)
        vel = vel + acc_g * dt
        dp = vel * dt
        x += dp[0]
        y += dp[1]
        sol = np.vstack((sol, np.array([x, y])))
        if y < 0:
            break

    return sol


def simulate(proj: ProjectileData, dt: num) -> np.ndarray:
    """
    Calculates the trajectory of the projectile with air resistance, if
    all necessary parameters are provided. Otherwise air resistance is
    neglected.
    :param proj: A ProjectileData-object
    :param dt: Timestep
    :return: Array of x- and y-coordinates
    """
    if None in (proj.c_d, proj.rho, proj.area):
        return _simulate(proj.v0, proj.angle, dt)
    vel = np.array([np.cos(proj.angle), np.sin(proj.angle)]) * proj.v0
    sol = np.zeros((1, 2))
    x, y = 0, 0
    while True:
        acc_g = g_acc(y)
        acc_drag = drag_acc(proj.c_d, proj.m, proj.rho, vel, proj.area)
        vel = vel + (acc_g + acc_drag) * dt
        dp = vel * dt
        x += dp[0]
        y += dp[1]
        sol = np.vstack((sol, np.array([x, y])))
        if y < 0:
            break

    return sol


def display_results(coords: np.ndarray, dt: num) -> None:
    """
    Prints out and plots some key characteristics of the trajectory
    :param coords:
    :param dt:
    :return:
    """
    x, y = coords[:, 0], coords[:, 1]
    print()
    print(f'Total distance (in x-direction:) {np.max(x):.3f} m')
    print(f'Highest point: {np.max(y):.3f} m')
    print(f'Flight time: {x.shape[0] * dt:.3f} s')
    plt.plot(coords[:, 0], coords[:, 1])
    plt.grid()
    plt.show()


def main():
    m_p, v0, alpha = 9, 500, np.deg2rad(40)
    rho, c_d, r = 1.2, 0.4, 0.088
    dt = 0.001  # [s]
    ball = Sphere(m_p, v0, alpha, c_d, rho, r)
    coords = simulate(ball, dt)
    display_results(coords, dt)


if __name__ == '__main__':
    main()
