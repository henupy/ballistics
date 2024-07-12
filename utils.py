"""
File containing some utility functions used in multiple places
"""

import atmos
import constants
import numpy as np


def vec_len(v: np.ndarray) -> int | float:
    """
    Length of a vector
    :param v:
    :return:
    """
    return np.sqrt(np.dot(v, v))


def speed2velocity(speed: int | float, angle: int | float) -> np.ndarray:
    """
    Creates a velocity vector from the given speed and launch angle
    :param speed: Speed in units of [m/s]
    :param angle: Angle in degrees
    :return:
    """
    angle = np.deg2rad(angle)
    return np.array([np.cos(angle), np.sin(angle)]) * speed


def reynolds(size: int | float, rho: int | float, temp: int | float,
             vel: np.ndarray) -> int | float:
    """
    Calculates the Reynolds number of the flow around a sphere
    :param size: The characteristic size of the project (for a sphere this
    is usually the diameter) [m]
    :param rho: Air density [kg/m^3]
    :param temp: Air temperature [K]
    :param vel: Velocity of the projectile [m/s]
    :return:
    """
    visc = atmos.visc(rho=rho, temp=temp)
    vmag = vec_len(vel)
    return rho * vmag * size / visc


def grav_force(m: int | float, h: int | float) -> np.ndarray:
    """
    Calculates the acceleration due to gravital attraction
    between the Earth and the projectile
    :param m: Mass of the projectile [kg]
    :param h: Height of the projectile related to Earth's surface [m]
    :return: Gravitational force as a vector [N]
    """
    r = constants.r_e + h
    return np.array([0, -constants.big_g * constants.m_e * m / (r * r)])


def drag_force(rho: int | float, area: int | float, c_d: int | float,
               vel: np.ndarray) -> np.ndarray:
    """
    Calculates the drag force on the onject
    :param rho: Air density [kg/m^3]
    :param area: Cross sectional area of the projectile [m^2]
    :param c_d: Drag coefficient [-]
    :param vel: Velocity of the projectile as a vector [m/s]
    :return: Drag force as a vector [N]
    """
    k = .5 * rho * area * c_d
    return -k * vec_len(vel) * vel

