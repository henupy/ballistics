"""
File containing objects for different projectiles
"""

import numpy as np

from typing import Optional
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
        self.m = m
        self.v0 = v0
        self.angle = angle
        self.c_d = c_d
        self.area = self._area(r)

    def _area(self, r: numeric) -> Optional[numeric]:
        """
        :param r: Radius of a sphere
        :return:
        """
        if r is None:
            return None
        return np.pi * r * r
