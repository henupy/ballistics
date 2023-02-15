"""
File containing objects for different projectiles
"""

import numpy as np

from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# For typing
num = int | float


@dataclass
class ProjectileData(ABC):
    m: num
    v0: num
    angle: num
    c_d: num = None
    rho: num = None
    area: num = None

    @abstractmethod
    def _area(self, *args):
        """
        The cross sectional area of the projectile
        :param args:
        :return:
        """


class Sphere(ProjectileData):
    def __init__(self, m: num, v0: num, angle: num, c_d: num = None,
                 rho: num = None, r: num = None):
        self.m = m
        self.v0 = v0
        self.angle = angle
        self.c_d = c_d
        self.rho = rho
        self.area = self._area(r)

    def _area(self, r: num) -> Optional[num]:
        """
        :param r: Radius of a sphere
        :return:
        """
        if r is None:
            return None
        return np.pi * r * r
