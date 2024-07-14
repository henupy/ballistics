"""
Some correlations for drag coefficients of arbitrary objects

Correlations from: El Hasadi & Padding - A Generalized Model for Predicting
the Drag Coefficient of Arbitrary Bluff Shaped Bodies at High Reynolds
Numbers (2023)
(Should be) Available at: https://arxiv.org/pdf/2308.05272.pdf
"""

import numpy as np

from abc import ABC, abstractmethod


class DragCorrelation(ABC):

    @abstractmethod
    def eval(self, re: int | float) -> float:
        """
        :param re: Reynolds number [-]
        :return:
        """
        raise NotImplementedError("Abstract method called")


class HaiderLevenspiel(DragCorrelation):
    def __init__(self, surf_area: int | float, volume: int | float,
                 proj_area: int | float = None) -> None:
        """
        :param surf_area: Total surface area of the projectile [m^2]
        :param volume: Volume of the projectile [m^3]
        :param proj_area: Projected surface area of the projectile [m^2]
        :return:
        """
        # Projected area is not needed for this correlation, but it should accept
        # it anyway, so that this accepts the same parameters as HolzerSommerfeld 
        _ = proj_area
        self.surf_area = surf_area
        self.volume = volume
        self.phi = self._calc_sphericity()
        self.a = self._calc_a()
        self.b = self._calc_b()
        self.c = self._calc_c()
        self.d = self._calc_d()

    def _calc_sphericity(self) -> float:
        """
        Calculates the sphericity of the object as it is defined for the
        correlation
        :return:
        """
        # Radius of a sphere of the same volume as the given projectile
        r = np.cbrt(self.volume * 3 / 4 / np.pi)
        # Surface area of this volume equivalent sphere
        s_area = 4 * np.pi * r * r
        # Sphericity is the ratio between the spheres and the projectiles
        # surface areas
        phi = s_area / self.surf_area
        # The sphericity should never be above 1, let's add a guard for it
        # in case I made a mistake
        if phi > 1:
            raise ValueError("Too large sphericity")
        return phi

    def _calc_a(self) -> float:
        """
        The coefficient A
        :return:
        """
        phi2 = self.phi * self.phi
        return np.exp(2.3288 - 6.4581 * self.phi + 2.4486 * phi2)

    def _calc_b(self) -> float:
        """
        The coefficient B
        :return:
        """
        return .0964 + 0.5565 * self.phi

    def _calc_c(self) -> float:
        """
        The coefficient C
        :return:
        """
        phi2 = self.phi * self.phi
        phi3 = phi2 * self.phi
        return np.exp(4.905 - 13.8944 * self.phi + 18.4222 * phi2 - 10.2599 * phi3)

    def _calc_d(self) -> float:
        """
        :return:
        """
        phi2 = self.phi * self.phi
        phi3 = phi2 * self.phi
        return np.exp(1.4681 + 12.2584 * self.phi - 20.7322 * phi2 + 15.8855 * phi3)

    def eval(self, re: int | float) -> float:
        """
        The correlation itself I suppose
        :param re: Reynolds number [-]
        :return:
        """
        if re <= .1:
            return 0.
        # "t" stands for "term"
        t1 = 24 / re
        t2 = 1 + self.a * np.power(re, self.b)
        t3 = self.c / (1 + self.d / re)
        return t1 * t2 + t3


class HolzerSommerfeld(DragCorrelation):
    def __init__(self, surf_area: int | float, volume: int | float,
                 proj_area: int | float) -> None:
        """
        :param surf_area: Total surface area of the projectile [m^2]
        :param volume: Volume of the projectile [m^3]
        :param proj_area: Projected surface area of the projectile [m^2]
        :return:
        """
        self.surf_area = surf_area
        self.volume = volume
        self.proj_area = proj_area
        self.phi = self._calc_sphericity()
        self.psi = self._calc_cw_sphericity()

    def _calc_sphericity(self) -> float:
        """
        Calculates the sphericity of the object as it is defined for the
        correlation
        :return:
        """
        # Radius of a sphere of the same volume as the given projectile
        r = np.cbrt(self.volume * 3 / 4 / np.pi)
        # Surface area of this volume equivalent sphere
        s_area = 4 * np.pi * r * r
        # Sphericity is the ratio between the sphere's and the projectile's
        # surface areas
        phi = s_area / self.surf_area
        # The sphericity should never be above 1, let's add a guard for it
        # in case I made a mistake
        if phi > 1:
            raise ValueError("Too large sphericity")
        return phi

    def _calc_cw_sphericity(self) -> float:
        """
        Calculates the crosswise sphericity of the projectile as it is defined
        in the correlation
        :return:
        """
        # Radius of a sphere of the same volume as the given projectile
        r = np.cbrt(self.volume * 3 / 4 / np.pi)
        # Cross-sectional area of the volume equivalent sphere
        proj_area = np.pi * r * r
        return proj_area / self.proj_area

    def eval(self, re: int | float) -> float:
        """
        The correlation itself or something
        :param re: Reynolds number [-]
        :return:
        """
        if re <= .1:
            return 0.
        # "t" stands for "term"
        t1 = 8 / re * 1 / np.sqrt(self.psi)
        t2 = 16 / re * 1 / np.sqrt(self.phi)
        t3 = 3 / np.sqrt(re) * 1 / np.power(self.phi, 3 / 4)
        e = .4 * np.power(-np.log(self.phi), .2)  # Exponent in t4
        t4 = .42 * np.power(10, e) * 1 / self.psi
        return t1 + t2 + t3 + t4

