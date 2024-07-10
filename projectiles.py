"""
File containing objects for different projectiles
"""

import numpy as np


from typing import Callable, Type
from abc import ABC, abstractmethod
from drag_correlations import DragCorrelation

# TODO: Probably bad idea to use keyword arguments in user provided functions
# and classes

class Projectile(ABC):
    """
    Abstract class to define a template for the classes of the different
    projectiles.
    """
    m: int | float
    p0: np.ndarray
    v0: np.ndarray
    size: int | float
    drag_corr = Type[DragCorrelation]
    name: str = None

    @property
    @abstractmethod
    def proj_area(self) -> int | float:
        """
        The cross sectional area of the projectile
        :return:
        """
        raise NotImplementedError("Abstract method called")

    @property
    @abstractmethod
    def surf_area(self) -> int | float:
        """
        The surface area of the projectile
        :return:
        """
        raise NotImplementedError("Abstract method called")

    @property
    @abstractmethod
    def volume(self) -> int | float:
        """
        Volume of the projectile
        :return:
        """
        raise NotImplementedError("Abstract method called")

    @abstractmethod
    def get_cd(re: int | float) -> int | float:
        """
        Calculates the drag coefficient
        :param re: Reynolds number [-]
        :return:
        """
        raise NotImplementedError("Abstract method called")

    @abstractmethod
    def __str__(self) -> str:
        """
        :return:
        """
        raise NotImplementedError("Abstract method called")


class Sphere(Projectile):
    def __init__(self, m: int | float, p0: np.ndarray, v0: np.ndarray,
                 r: int | float, drag_corr: Type[DragCorrelation],
                 name: str = None) -> None:
        """
        :param m: Mass [kg]
        :param p0: Initial position [m]
        :param v0: Initial velocity [m/s]
        :param r: Radius [m]
        :param drag_corr: Some DragCorrelation class
        :param name: Optional name for the projectile, will be used in the legend
        of the plots so that the projectile can be identified.
        :return:
        """
        self.m = m
        self.p0 = p0
        self.v0 = v0
        self.r = r
        self.drag_corr = drag_corr(surf_area=self.surf_area, volume=self.volume,
                                   proj_area=self.proj_area)
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

    def get_cd(self, re: int | float) -> int | float:
        """
        Calculates the drag coefficient based on the Reynolds number using
        the drag correlation attribute
        :param re: Reynolds number [-]
        :return:
        """
        return self.drag_corr.eval(re=re)

    def __str__(self) -> str:
        """
        :return:
        """
        if self.name is None:
            return f"{self.__class__.__name__}"
        return self.name


class Cube(Projectile):
    def __init__(self, m: int | float, p0: np.ndarray, v0: np.ndarray,
                 d: int | float, drag_corr: Type[DragCorrelation],
                 name: str = None) -> None:
        """
        :param m: Mass [kg]
        :param p0: Initial position [m]
        :param v0: Initial velocity [m/s]
        :param d: Side length [m]
        :param drag_corr: Some DragCorrelation class
        :param name: Optional name for the projectile, will be used in the legend
        of the plots so that the projectile can be identified.
        :return:
        """
        self.m = m
        self.p0 = p0
        self.v0 = v0
        self.d = d
        self.size = d
        self.drag_corr = drag_corr(surf_area=self.surf_area, volume=self.volume,
                                   proj_area=self.proj_area)
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
        The surface area of a cube
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

    def get_cd(self, re: int | float) -> int | float:
        """
        Calculates the drag coefficient based on the Reynolds number using
        the drag correlation attribute
        :param re: Reynolds number [-]
        :return:
        """
        return self.drag_corr.eval(re=re)

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
                 d: int | float, drag_corr: Type[DragCorrelation],
                 name: str = None) -> None:
        """
        :param m: Mass of the projectile [kg]
        :param p0: Initial position of the projectile [m]
        :param v0: Initial velocity of the projectile [m/s]
        :param d: Diameter of the sphere [m]
        :param drag_corr: Some DragCorrelation class
        :param name: Optional name for the projectile, will be used in the legend
        of the plots so that the projectile can be identified.
        :return:
        """
        super().__init__(m=m, p0=p0, v0=v0, r=d / 2.25, drag_corr=drag_corr,
                         name=name)


class Rocket(Projectile):
    """
    The rocket is here assumed to essentially be a shell shaped thing that
    has a changing mass and non-zero thrust.
    """
    def __init__(self, p0: np.ndarray, d: int | float, length: int | float,
                 mass_fun: Callable, thrust_fun: Callable,
                 drag_corr: Type[DragCorrelation], name: str = None) -> None:
        """
        :param p0: Initial position [m]
        :param d: Diameter [m]
        :param length: Length [m]
        :param mass_fun: Function that describes the mass of the rocket
        as a function of time [kg]
        :param thrust_fun: Function that describes the thrust produced
        by the rocket as a function of time [N]
        :param drag_corr: Some DragCorrelation class
        :oaram name: Optional name for the projectile, will be used in the legend
        of the plots so that the projectile can be identified.
        :return:
        """
        self.p0 = p0
        self.d = d
        self.r = d / 2
        self.length = length
        self.mass_fun = mass_fun
        self.thrust_fun = thrust_fun
        self.drag_corr = drag_corr(surf_area=self.surf_area, volume=self.volume,
                                   proj_area=self.proj_area)
        self.name = name

    @property
    def proj_area(self) -> int | float:
        """
        The projected area of the rocket (assumed to be parallel to the flow)
        :return:
        """
        return np.pi * self.r * self.r

    @property
    def surf_area(self) -> int | float:
        """
        The surface area of the rocket (here assumed to be a cylinder)
        :return:
        """
        return self.proj_area * 2 + np.pi * self.d * self.length

    @property
    def volume(self) -> int | float:
        """
        Volume of a cube
        :return:
        """
        return self.proj_area * self.length

    def get_cd(self, re: int | float) -> int | float:
        """
        Calculates the drag coefficient based on the Reynolds number using
        the drag correlation attribute
        :param re: Reynolds number [-]
        :return:
        """
        return self.drag_corr.eval(re=re)

    def get_mass(self, t: int | float) -> int | float:
        """
        Gets the mass at the given time
        :param t: Time [s]
        :return:
        """
        return self.mass_fun(t=t)

    def get_thrust(self, t: int | float) -> int | float:
        """
        Gets the thrust at the given 01:42
        :param t: Time [s]
        :return:
        """
        return self.thrust_fun(t=t)

    def __str__(self) -> str:
        """
        :return:
        """
        if self.name is None:
            return f"{self.__class__.__name__}"
        return self.name


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
