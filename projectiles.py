"""
File containing objects for different projectiles
"""

import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod

# For typing
numeric = int | float


@dataclass
class ProjectileData(ABC):
    m: numeric
    v0: numeric
    angle: numeric
    c_d: numeric = None
    area: numeric = None

    @abstractmethod
    def _area(self, *args):
        """
        The cross sectional area of the projectile
        :param args:
        :return:
        """


class Sphere(ProjectileData):
    def __init__(self, m: numeric, v0: numeric, angle: numeric, c_d: numeric = None,
                 r: numeric = None):
        """
        :param m: Mass [kg]
        :param v0: Initial velocity [m/s]
        :param angle: Launch angle [deg]
        :param c_d: Drag coefficient [-]
        :param r: Radius [m]
        """
        self.m = m
        self.v0 = v0
        self.angle = np.deg2rad(angle)
        self.c_d = c_d
        self.area = self._area(r)

    def _area(self, r: numeric) -> numeric:
        """
        The projected area of the sphere
        :param r: Radius [m]
        :return:
        """
        return np.pi * r * r
