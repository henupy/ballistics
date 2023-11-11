"""
File for a ProjectileData object which is the object that's actually
used in the simulations
"""
import numpy as np

from projectiles import Projectile


class ProjectileData:
    def __init__(self, proj: Projectile, coords: np.ndarray, vel: np.ndarray,
                 cd: np.ndarray, rey: np.ndarray, dt: int | float) -> None:
        """
        :param proj: A Projectile object representing the projectile itself
        :param coords:
        :param vel:
        :param cd:
        :param rey:
        projectile
        """
        self.proj = proj
        self.coords = coords
        self.vel = vel
        self.c_d = cd
        self.re = rey
        self.dt = dt
        x, y = coords[:, 0], coords[:, 1]
        self.x_dist = x[-1] - x[0]
        self.y_max = np.max(y)
        self.time = x.shape[0] * dt
