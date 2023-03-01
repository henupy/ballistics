"""
Some shiet regarding ballistic trajectories. At this moment, the projectile
is assumed to be launched from the surface of the Earth.
"""

import constants
import numpy as np
import matplotlib.pyplot as plt

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


def _f2k(temp: numeric) -> numeric:
    """
    Converts Fahrenheits to Kelvin
    :param temp: Air temperature [K]
    :return:
    """
    return (temp + 459.67) * 5 / 9


def _density_and_temp(y: numeric) -> tuple[numeric, numeric]:
    """
    Returns the air density and temperature at the given height using Nasa's
    Earth Atmosphere Model (https://www.grc.nasa.gov/www/k-12/rocket/atmos.html).
    This works when y < 82345 feet (~25099 meters).
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
    return _f2k(temp), rho * 515.3788184


def _visc(rho: numeric, temp: numeric) -> numeric:
    """
    Calculates the dynamic viscosity of air as a function of density and
    temperature using the correlation found in Journal of Physical and Chemical
    Reference Data 14, 947 (1985).
    :param rho: Air density [kg/m^3]
    :param temp: [K]
    :return:
    """
    temp /= constants.t_star
    rho /= constants.rho_star

    # Calculate the excess viscosity
    bb = [constants.b_1, constants.b_2, constants.b_3, constants.b_4]
    v_excess = 0
    for i, b in enumerate(bb):
        v_excess += b * np.power(rho, i + 1)

    # Calculate the sum term for the 'temperature viscosity'
    aa = [constants.a_0, constants.a__1, constants.a__2, constants.a__3, constants.a__4]
    sum_term = 0
    for i, a in enumerate(aa):
        sum_term += a * np.power(temp, -i)

    # Calculate the full 'temperature viscosity'
    v_temp = constants.a_1 * temp + constants.a_05 * np.power(temp, 0.5) + sum_term

    # Return the total viscosity
    return constants.h * (v_temp + v_excess)


def _vec_len(v: np.ndarray) -> numeric:
    """
    Length of a vector
    :param v:
    :return:
    """
    return np.sqrt(np.dot(v, v))


def _reynolds(size: numeric, rho: numeric, temp: numeric,
              vel: np.ndarray) -> numeric:
    """
    Calculates the Reynolds number of the flow around a sphere
    :param size: The characteristic size of the project (for a sphere this
    is usually the diameter) [m]
    :param rho: Air density [kg/m^3]
    :param temp: Air temperature [K]
    :param vel: Velocity of the projectile [m/s]
    :return:
    """
    visc = _visc(rho=rho, temp=temp)
    vmag = _vec_len(vel)
    return rho * vmag * size / visc


def _g_acc(h: numeric) -> np.ndarray:
    """
    Calculates the acceleration due to gravital attraction
    between the Earth and the projectile
    :param h: Height of the projectile related to Earth's surface [m]
    :return: Acceleration due to gravity as a vector [m/s^2]
    """
    if h == 0:
        return np.array([0, 0])
    r = constants.r_e + h
    return np.array([0, -constants.big_g * constants.m_e / (r * r)])


def _cd_sphere(size: numeric, rho: numeric, temp: numeric, vel: np.ndarray) -> numeric:
    """
    Calculates the drag coefficient for a sphere using the correlation in
    Morrison (2016)
    (https://pages.mtu.edu/~fmorriso/DataCorrelationForSphereDrag2016.pdf)
    It is not recommended to use the correlation for cases when Re > 1e6, but
    we shall ignore it for now
    :param size: The characteristic size of the project (for a sphere this
    is usually the diameter) [m]
    :param rho: Air density [kg/m^3]
    :param temp: Air temperature [K]
    :param vel: Velocity of the projectile [m/s]
    :return:
    """
    # Determine the Reynolds number
    re = _reynolds(size=size, rho=rho, temp=temp, vel=vel)

    # Let's define the correlation term by term
    term1 = 24 / re
    re1 = re / 5
    term2 = 2.6 * re1 / (1 + np.power(re1, 1.52))
    re2 = re / 2.63e5
    term3 = 0.411 * np.power(re2, -7.94) / (1 + np.power(re2, -8))
    re3 = re / 1e6
    term4 = .25 * re3 / (1 + re3)
    return term1 + term2 + term3 + term4


def _drag_acc(m: numeric, size: numeric, area: numeric, rho: numeric, temp: numeric,
              vel: np.ndarray) -> np.ndarray:
    """
    Calculates the acceleration due to air resistance
    :param m: Mass of the projectile [kg]
    :param size: The characteristic size of the project (for a sphere this
    is usually the diameter) [m]
    :param area: Cross sectional (or projected) area of the projectile [m^2]
    :param rho: Density of air [kg/m^3]
    :param temp: Temperature of the air [K]
    :param vel: Velocity of the projectile as a vector [m/s]
    :return: Acceleration due to drag as a vector [m/s^2]
    """
    c_d = _cd_sphere(size=size, rho=rho, temp=temp, vel=vel)
    return -0.5 * c_d * rho * (vel * vel) * area / m


def _simple_sim(v0: np.ndarray, dt: numeric) -> np.ndarray:
    """
    Calculates the trajectory of the projectile without air resistance
    :param v0: Initial velocity of the projectile
    :param dt: Timestep
    :return: Array of x- and y-coordinates
    """
    vel = v0
    pos = np.zeros(shape=(2, ))
    sol = np.zeros(shape=(1, 2))
    sol[0] = pos
    while True:
        acc = _g_acc(pos[1])
        vel += acc * dt
        pos += vel * dt
        sol = np.vstack((sol, pos))
        if pos[1] < 0:
            break

    return sol


def simulate(m: numeric, v0: np.ndarray, dt: numeric, size: numeric = None,
             area: numeric = None) -> np.ndarray:
    """
    Calculates the trajectory of the projectile with air resistance, if
    all necessary parameters are provided with the ProjectileData object.
    Otherwise air resistance is neglected. The simulation is continued until
    the projectile hits the ground.
    :param m: Mass of the projectile [kg]
    :param v0: Inital velocity of the projectile [m/s]
    :param dt: Timestep [s]
    :param size: Size of the projectile (for a sphere the diameter) [m]
    :param area: Cross sectional (or projected) area of the projectile [m^2]
    :return: Array of x- and y-coordinates [m]
    """
    # If the necessary parameters to calculate drag aren't provided, let's
    # ignore it
    if None in (size, area):
        return _simple_sim(v0=v0, dt=dt)
    vel = v0
    pos = np.zeros(shape=(2, ))
    sol = np.zeros(shape=(1, 2))
    sol[0] = pos
    while True:
        acc = np.zeros(shape=(2, ))
        acc += _g_acc(pos[1])
        temp, rho = _density_and_temp(pos[1])
        acc += _drag_acc(m=m, size=size, area=area, rho=rho, temp=temp, vel=vel)
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
    # These values are for the Flak 88 anti-aircraft/anti-tank artillery
    # Taken from https://en.wikipedia.org/wiki/8.8_cm_Flak_18/36/37/41
    m_p = 9.2  # Mass of the projectile [kg]
    v0_mag = 840  # Magnitude of the velocity of the projectile [m/s]
    r = 0.088  # Radius [m]
    alpha = 30  # Launch angle [deg]
    v0 = np.array([np.cos(np.deg2rad(alpha)), np.sin(np.deg2rad(alpha))]) * v0_mag
    dt = 0.001  # Timestep [s]
    size = 2 * r
    area = np.pi * r * r
    coords = simulate(m=m_p, v0=v0, dt=dt, size=size, area=area)
    display_results(coords=coords, dt=dt)


if __name__ == '__main__':
    main()
