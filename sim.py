"""
Some shiet regarding ballistic trajectories. At this moment, the projectile
is assumed to be launched from the surface of the Earth.
"""

import numpy as np
import matplotlib.pyplot as plt

from projectiles import ProjectileData, Sphere

# For typing
numeric = int | float


def _rho(pres: numeric, temp: numeric) -> numeric:
    """
    Calculates air density from air pressure and temperature according to the
    Nasa Earth Atmosphere Model
    (https://www.grc.nasa.gov/www/k-12/rocket/atmos.html)
    :param pres: Air pressure [lbs/sq ft]
    :param temp: Air temperature [°F]
    :return: Air density [slugs/cu ft]
    """
    return pres / (1718 * (temp + 459.7))


def _get_density(y: numeric) -> numeric:
    """
    Returns the air density at the given height using Nasa's Earth Atmosphere
    Model (https://www.grc.nasa.gov/www/k-12/rocket/atmos.html). This works when
    y < 82345 feet (~25099 meters).
    :param y: Height of the projectile [m]
    :return:
    """
    # Convert the height from meters to feet
    y /= 0.3048
    if y < 36152:
        temp = 59 - .00356 * y
        pres = 2116 * np.power((temp + 459.7) / 518.6, 5.256)
        rho = _rho(pres=pres, temp=temp)
    elif 36152 < y < 82345:
        temp = -70
        pres = 473.1 * np.exp(1.73 - .000048 * y)
        rho = _rho(pres=pres, temp=temp)
    else:
        raise ValueError('Height is too large')
    # Convert the density to metric units [kg/m^3]
    return rho * 515.3788184


def _g_acc(h: numeric) -> np.ndarray:
    """
    Calculates the acceleration due to gravital attraction
    between the Earth and the projectile
    :param h: Height of the projectile related to Earth's surface
    :return: Acceleration due to gravity as a vector
    """
    if h == 0:
        return np.array([0, 0])
    big_g = 6.67259e-11  # Gravitational constant [Nm^2/kg^2]
    r_e = 6378.14e3  # Radius of the Earth [m]
    m_e = 5.974e24  # Mass of the earth [kg]
    r = r_e + h
    return np.array([0, -big_g * m_e / (r * r)])


def _drag_acc(c_d: numeric, m_p: numeric, rho: numeric, vel: np.ndarray,
              area: numeric) -> np.ndarray:
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


def _simple_sim(v0: numeric, alpha: numeric, dt: numeric) -> np.ndarray:
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
        acc_g = _g_acc(y)
        vel = vel + acc_g * dt
        dp = vel * dt
        x += dp[0]
        y += dp[1]
        sol = np.vstack((sol, np.array([x, y])))
        if y < 0:
            break

    return sol


def simulate(proj: ProjectileData, dt: numeric) -> np.ndarray:
    """
    Calculates the trajectory of the projectile with air resistance, if
    all necessary parameters are provided with the ProjectileData object.
    Otherwise air resistance is neglected. The simulation is continued until
    the projectile hits the ground.
    :param proj: A ProjectileData-object
    :param dt: Timestep
    :return: Array of x- and y-coordinates
    """
    if None in (proj.c_d, proj.area):
        return _simple_sim(v0=proj.v0, alpha=proj.angle, dt=dt)
    vel = np.array([np.cos(proj.angle), np.sin(proj.angle)]) * proj.v0
    pos = np.zeros(shape=(2, ))
    sol = np.zeros(shape=(1, 2))
    sol[0] = pos
    while True:
        acc = np.zeros(shape=(2, ))
        acc += _g_acc(pos[1])
        rho = _get_density(pos[1])
        acc += _drag_acc(c_d=proj.c_d, m_p=proj.m, rho=rho, vel=vel, area=proj.area)
        vel += acc * dt
        pos += vel * dt
        sol = np.vstack((sol, pos))
        if pos[1] < 0:
            break

    return sol


def display_results(coords: np.ndarray, dt: numeric) -> None:
    """
    Prints out and plots some key characteristics of the trajectory
    :param coords:
    :param dt:
    :return:
    """
    x, y = coords[:, 0], coords[:, 1]
    print('Flight data:')
    print(f'Total distance (in x-direction): {np.max(x):.3f} m')
    print(f'Highest point: {np.max(y):.3f} m')
    print(f'Flight time: {x.shape[0] * dt:.3f} s')
    plt.plot(coords[:, 0], coords[:, 1])
    plt.grid()
    plt.show()


def main():
    m_p = 9  # Mass of the projectile [kg]
    v0 = 500  # Initial velocity of the projectile [m/s]
    alpha = 40  # Launch angle [deg]
    c_d = 0.4  # Drag coefficient [-]
    r = 0.088  # Radius [m]
    dt = 0.001  # Timestep [s]
    ball = Sphere(m=m_p, v0=v0, angle=alpha, c_d=c_d, r=r)
    coords = simulate(proj=ball, dt=dt)
    display_results(coords=coords, dt=dt)


if __name__ == '__main__':
    main()
