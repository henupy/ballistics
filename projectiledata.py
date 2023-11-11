"""
File for a ProjectileData object which is the object that's actually
used in the simulations
"""
import numpy as np

from projectiles import Projectile


def _speed(vel: np.ndarray) -> np.ndarray:
    """
    :param vel:
    :return:
    """
    vx, vy = vel[:, 0], vel[:, 1]
    return np.sqrt(np.power(vx, 2) + np.power(vy, 2))


class ProjectileData:
    def __init__(self, proj: Projectile, coords: np.ndarray, vel: np.ndarray,
                 cd: np.ndarray, rey: np.ndarray, dt: int | float) -> None:
        """
        :param proj: A Projectile object representing the projectile itself
        :param coords: Coordinates of the trajectory
        :param vel: Velocity of the projectile at different timepoints
        :param cd: Drag coefficient at different timepoints
        :param rey: Reynolds number at different timepoints
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
        self.speed = _speed(vel=vel)
        self.ke = .5 * proj.m * np.power(self.speed, 2)
