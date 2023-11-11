"""
File containing objects for different projectiles
"""

import numpy as np

from collections import namedtuple
from abc import ABC, abstractmethod


class Projectile(ABC):
    """
    Abstract class to define a template for the classes of the different
    projectiles.
    """
    m: int | float
    p0: np.ndarray
    v0: np.ndarray
    size: int | float
    name: str = None

    @property
    @abstractmethod
    def proj_area(self) -> int | float:
        """
        The cross sectional area of the projectile
        :return:
        """

    @property
    @abstractmethod
    def surf_area(self) -> int | float:
        """
        The surface area of the projectile
        :return:
        """

    @property
    @abstractmethod
    def volume(self) -> int | float:
        """
        Volume of the projectile
        :return:
        """

    @abstractmethod
    def __str__(self) -> str:
        """
        :return:
        """


class Sphere(Projectile):
    def __init__(self, m: int | float, p0: np.ndarray, v0: np.ndarray,
                 r: int | float, name: str = None) -> None:
        """
        :param m: Mass [kg]
        :param p0: Initial position [m]
        :param v0: Initial velocity [m/s]
        :param r: Radius [m]
        :param name: Optional name for the projectile, will be used in the legend
        of the plots so that the projectile can be identified.
        """
        self.m = m
        self.p0 = p0
        self.v0 = v0
        self.r = r
        self.name = name
        self.size = 2 * r

    @property
    def proj_area(self) -> int | float:
        """
        The projected area of the sphere
        :return:
        """
        return np.pi * self.r * self.r

    @property
    def surf_area(self) -> int | float:
        """
        THe surface area of a sphere
        :return:
        """
        return 4 * np.pi * self.r * self.r

    @property
    def volume(self) -> int | float:
        """
        Volume of a sphere
        :return:
        """
        return 4 / 3 * np.pi * self.r * self.r * self.r

    def __str__(self) -> str:
        """
        :return:
        """
        if self.name is None:
            return f"{self.__class__.__name__}"
        return self.name


class Cube(Projectile):
    def __init__(self, m: int | float, p0: np.ndarray, v0: np.ndarray,
                 d: int | float, name: str = None) -> None:
        """
        :param m: Mass [kg]
        :param p0: Initial position [m]
        :param v0: Initial velocity [m/s]
        :param d: Side length [m]
        :param name:
        """
        self.m = m
        self.p0 = p0
        self.v0 = v0
        self.d = d
        self.size = d
        self.name = name

    @property
    def proj_area(self) -> int | float:
        """
        The projected area of a cube (one plane normal to the flow)
        :return:
        """
        return self.d * self.d

    @property
    def surf_area(self) -> int | float:
        """
        THe surface area of a cube
        :return:
        """
        return self.d * self.d * 6

    @property
    def volume(self) -> int | float:
        """
        Volume of a cube
        :return:
        """
        return self.d * self.d * self.d

    def __str__(self) -> str:
        """
        :return:
        """
        if self.name is None:
            return f"{self.__class__.__name__}"
        return self.name


SimObject = namedtuple(typename="SimObject", field_names=["proj", "drag_corr"])
