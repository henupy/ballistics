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
    m: numeric  # Mass of the projectile [kg]
    v0: np.ndarray  # Initial velocity of the projectile [m/s]
    size: numeric = None  # The characteristic length/size of the projectile [m]
    area: numeric = None  # The cross sectional/projected area of the body [m^2]

    @abstractmethod
    def _area(self, *args):
        """
        The cross sectional area of the projectile
        :param args:
        :return:
        """


class Sphere(ProjectileData):
    def __init__(self, m: numeric, v0: np.ndarray, r: numeric = None) -> None:
        """
        :param m: Mass [kg]
        :param v0: Initial velocity [m/s]
        :param r: Radius [m]
        """
        self.m = m
        self.v0 = v0
        self.size = 2 * r
        self.area = self._area(r)

    def _area(self, r: numeric) -> numeric:
        """
        The projected area of the sphere
        :param r: Radius [m]
        :return:
        """
        return np.pi * r * r
