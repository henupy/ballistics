"""
File containing objects for different projectiles
"""

import numpy as np

from collections import namedtuple
from abc import ABC, abstractmethod

# Basically just a definition for a "Simulation object" that is used in the
# simulations to bind together a projectile and a drag correlation
SimObject = namedtuple(typename="SimObject", field_names=["proj", "drag_corr"])


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


class Shell(Sphere):
    """
    A class for an projectile shaped like a bullet or an artillery shell.
    This is essentially just a sphere but the the more aerodynamic shape
    of the bullet/shell is modelled by using a smaller diameter for the
    sphere than what would be the shell's real diameter/caliber.

    Instead of radius being diameter / 2, it's diameter / 2.25. The 2.25
    value was chosen because with it it seemed that the simulated flights
    matched pretty closely to the actual ones for some artillery shells.
    See https://en.wikipedia.org/wiki/8.8_cm_Flak_18/36/37/41 and
    https://en.wikipedia.org/wiki/Bofors_57_mm_Naval_Automatic_Gun_L/70.
    """
    def __init__(self, m: int | float, p0: np.ndarray, v0: np.ndarray,
                 d: int | float, name: str = None) -> None:
        """
        :param m:
        :param p0:
        :param v0:
        :param d:
        :param name:
        """
        super().__init__(m=m, p0=p0, v0=v0, r=d / 2.25, name=name)

# The currently available drag correlations don't work well with the
# elongated shape of a bullet/artillery shell or something like that
# (they seem to overestimate the drag) so the implementation for an
# 'actual' shell below is disabled for now.

# class Shell(Projectile):
#     """
#     A projectile that has the shape of a bullet or an artillery shell
#     """
#     def __init__(self, m: int | float, p0: np.ndarray, v0: np.ndarray,
#                  d: int | float, len1: int | float, len2: int | float,
#                  name: str = None) -> None:
#         """
#         :param m: Mass [kg]
#         :param p0: Initial position [m]
#         :param v0: Initial velocity [m/s]
#         :param d: The diameter of the shell (basically the caliber) [m]
#         :param len1: Length of the straight part of the shell [m]
#         :param len2: Length of the curved part of the shell [m]
#         :param name:
#         """
#         self.m = m
#         self.p0 = p0
#         self.v0 = v0
#         self.d = d
#         self.r = d / 2
#         self.size = d
#         self.len1 = len1
#         self.len2 = len2
#         self.name = name
#
#     @property
#     def proj_area(self) -> int | float:
#         """
#         The projected area of a shell (assumed to be perpendicular
#         to the flow always)
#         :return:
#         """
#         return np.pi * self.r * self.r
#
#     @property
#     def surf_area(self) -> int | float:
#         """
#         Surface area of the shell. The straight part is just a round
#         cylinder and the curved part is a paraboloid
#         (https://en.wikipedia.org/wiki/Paraboloid), I think?
#         :return:
#         """
#         cylinder_a = np.pi * self.r * self.r * self.len1
#         len_sq = self.len2 * self.len2
#         r2 = self.r * self.r
#         t1 = np.sqrt(np.power(r2 + 4 * len_sq, 3))
#         t2 = np.pi * self.r * (t1 - r2 * self.r)
#         curved_a = t2 / (6 * len_sq)
#         return cylinder_a + curved_a
#
#     @property
#     def volume(self) -> int | float:
#         """
#         Volume of the shell. The straight part is just a round cylinder
#         and the curved part is a paraboloid
#         (https://en.wikipedia.org/wiki/Paraboloid), I think?
#         :return:
#         """
#         return np.pi / 2 * self.r * self.r * self.len2
#
#     def __str__(self) -> str:
#         """
#         :return:
#         """
#         if self.name is None:
#             return f"{self.__class__.__name__}"
#         return self.name
