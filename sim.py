"""
Some shiet regarding ballistic trajectories. The simulation tries to be a bit more
realistic by modelling the variation of the density of the earth's atmosphere as a
function of altitude and the variation of the drag coefficient of the projectiles
(of different shapes), which varies as a function of the Reynolds number.

This simulation assumes that the projectile moves freely with some initial
velocity, meaning that the projectile does not have any thrust.
"""

import atmos
import constants
import numpy as np
import matplotlib.pyplot as plt

from solvers import rk4
from typing import Callable
from projectiledata import ProjectileData
from projectiles import Shell, Sphere, SimObject
from drag_correlations import HolzerSommerfeld


def vec_len(v: np.ndarray) -> int | float:
    """
    Length of a vector
    :param v:
    :return:
    """
    return np.sqrt(np.dot(v, v))


def reynolds(size: int | float, rho: int | float, temp: int | float,
              vel: np.ndarray) -> int | float:
    """
    Calculates the Reynolds number of the flow around a sphere
    :param size: The characteristic size of the project (for a sphere this
    is usually the diameter) [m]
    :param rho: Air density [kg/m^3]
    :param temp: Air temperature [K]
    :param vel: Velocity of the projectile [m/s]
    :return:
    """
    visc = atmos.visc(rho=rho, temp=temp)
    vmag = vec_len(vel)
    return rho * vmag * size / visc


def _grav_force(m: int | float, h: int | float) -> np.ndarray:
    """
    Calculates the acceleration due to gravital attraction
    between the Earth and the projectile
    :param m: Mass of the projectile [kg]
    :param h: Height of the projectile related to Earth's surface [m]
    :return: Gravitational force as a vector [N]
    """
    r = constants.r_e + h
    return np.array([0, -constants.big_g * constants.m_e * m / (r * r)])


def _drag_force(rho: int | float, area: int | float, c_d: int | float,
                vel: np.ndarray) -> np.ndarray:
    """
    Calculates the drag force on the onject
    :param rho: Air density [kg/m^3]
    :param area: Cross sectional area of the projectile [m^2]
    :param c_d: Drag coefficient [-]
    :param vel: Velocity of the projectile as a vector [m/s]
    :return: Drag force as a vector [N]
    """
    k = .5 * rho * area * c_d
    return -k * vec_len(vel) * vel


def _diff_eq(y0: np.ndarray, t: int | float, m: int | float, d_f: int | float,
             g_f: int | float) -> np.ndarray:
    """
    Differential equation for an object following ballistic trajectory
    :param y0: Initial conditions as a vector of vectors [position, velocity]
    :param t: Time [s] (unused)
    :param m: Mass of the projectile [kg]
    :param d_f: Drag force affecting the projectile [N]
    :param d_g: Gravitational force affecting the projectile [N]
    :return:
    """
    _ = t  # Unused variable, this suppresses the warning
    x, v = y0
    dydt = [v, (d_f + g_f) / m]
    return np.array(dydt)


def _solve(obj: SimObject, solver: Callable, dt: float,
           max_steps: int | float) -> ProjectileData:
    """
    :param obj: The SimObject of the projectile
    :param solver: The solver used to solve the differential equation of motion
    :param dt: Timestep size [s]
    :param max_steps: Maximum amount of timesteps to take
    :return: A ProjectileData object containing the simulation results
    """
    proj = obj.proj
    pos = np.zeros(shape=(1, 2))
    vel = np.zeros(shape=(1, 2))
    pos[0] = proj.p0
    vel[0] = proj.v0
    cd = []
    rey = []
    n = 0
    while n <= max_steps:
        grav_f = _grav_force(m=proj.m, h=float(pos[n, 1]))
        temp, _, rho = atmos.get_atmos_data(h=pos[n, 1])
        re = reynolds(proj.size, rho=rho, temp=temp, vel=vel[n])
        rey.append(re)
        c_d = obj.drag_corr.eval(re=re)
        cd.append(c_d)
        drag_f = _drag_force(rho=rho, area=proj.proj_area, c_d=c_d, vel=vel[n])
        t = dt * (n + 1)
        n_pos, n_vel = solver(diff_eq=_diff_eq, y0=np.vstack((pos[n], vel[n])), t=t,
                              dt=dt, m=proj.m, d_f=drag_f, g_f=grav_f)
        pos = np.vstack((pos, n_pos))
        vel = np.vstack((vel, n_vel))
        n += 1
        if pos[n, 1] <= 0:
            break

    if n >= max_steps:
        print(f"INFO: Maximum amount of iterations reached for "
              f"projectile {obj.proj}.")
    return ProjectileData(proj=obj.proj, coords=pos, vel=vel, cd=np.array(cd),
                          rey=np.array(rey), dt=dt)


def simulate(*args: SimObject, solver: Callable, dt: int | float,
             max_steps: int = 1e5) -> list[ProjectileData]:
    """
    Calculates the trajectory of the projectile with air resistance, if
    all necessary parameters are provided with the ProjectileData object.
    Otherwise air resistance is neglected. The simulation is continued until
    the projectile hits the ground or the max_steps amount of timesteps are
    simulated.
    :param args: Arbitrary amount of SimObjects to run the simulation for
    :param solver: The solver function used to solve the differential equation
    of motion
    :param dt: Timestep [s]
    :param max_steps: A hard limit to the number of timesteps to simulate so that
    the simulation does not run too long
    :return: List of ProjectileData objects containing the simulation results
    for all given args
    """
    data_objs = []
    for obj in args:
        data_obj = _solve(obj=obj, solver=solver, dt=dt, max_steps=max_steps)
        data_objs.append(data_obj)
    return data_objs


def display_results(*args: ProjectileData) -> None:
    """
    Prints out and plots some key characteristics of the trajectory
    :param args:
    :return:
    """
    # TODO: Some refactoring
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    for data_obj in args:
        tspan = np.linspace(0, data_obj.time, int(data_obj.time / data_obj.dt))
        print(f"Flight data for {data_obj.proj}:")
        print(f"Total distance (in x-direction): {data_obj.x_dist:.3f} m")
        print(f"Highest point: {data_obj.y_max:.3f} m")
        print(f"Flight time: {data_obj.time:.3f} s")
        print()
        plt.figure(fig1)
        plt.plot(data_obj.coords[:, 0], data_obj.coords[:, 1],
                 label=f"{data_obj.proj}")
        plt.figure(fig2)
        plt.plot(tspan, data_obj.vel[:, 0], label=f"X velocity, {data_obj.proj}")
        plt.plot(tspan, data_obj.vel[:, 1], label=f"Y velocity, {data_obj.proj}")
        plt.plot(tspan, data_obj.speed, label=f"Total velocity, {data_obj.proj}")
        plt.figure(fig3)
        plt.plot(tspan[1:], data_obj.c_d, label=f"{data_obj.proj}")
        plt.figure(fig4)
        plt.plot(tspan[1:], data_obj.re, label=f"{data_obj.proj}")
        plt.figure(fig5)
        plt.plot(tspan, data_obj.ke, label=f"{data_obj.proj}")

    ax1.set_title("Flight path")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.legend()
    ax1.grid()

    ax2.set_title("Velocities and speed as a function of time")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Speed [m/s]")
    ax2.legend()
    ax2.grid()

    ax3.set_title("Drag coefficient as a function of time")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Drag coefficient [-]")
    ax3.legend()
    ax3.grid()

    ax4.set_title("Reynolds number as a function of time")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Reynolds number [-]")
    ax4.legend()
    ax4.grid()

    ax5.set_title("Kinetic energy as a function of time")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Kinetic energy [J]")
    ax5.legend()
    ax5.grid()

    plt.show()


def speed2velocity(v0: int | float, angle: int | float) -> np.ndarray:
    """
    Creates a velocity vector from the given speed and launch angle
    :param v0:
    :param angle:
    :return:
    """
    angle = np.deg2rad(angle)
    return np.array([np.cos(angle), np.sin(angle)]) * v0


def main() -> None:
    # Define some initial values
    m = 9.4  # Mass of the projectiles [kg]
    d = .088  # Diameter of the shell [m]
    angle = 45  # Launch angle of the projectiles [deg]
    v0_mag = 840  # Initial speed of the projectiles [m/s]
    # Initial positions [m]
    p0_1 = np.array([0, 0], dtype=np.float64)
    p0_2 = np.array([0, 0], dtype=np.float64)
    # Initial velocities
    v0_1 = speed2velocity(v0=v0_mag, angle=angle)
    v0_2 = speed2velocity(v0=v0_mag, angle=angle)
    dt = .01  # Timestep [s]

    # Create some projectiles and stuff
    shell = Shell(m=m, p0=p0_1, v0=v0_1, d=d, name="Shell")
    ball = Sphere(m=m, p0=p0_2, v0=v0_2, r=d / 2, name="Sphere")
    shell_corr = HolzerSommerfeld(proj=shell)
    ball_corr = HolzerSommerfeld(proj=ball)
    shell_obj = SimObject(proj=shell, drag_corr=shell_corr)
    ball_obj = SimObject(proj=ball, drag_corr=ball_corr)

    # Simulate
    solver = rk4
    shell_data, ball_data = simulate(shell_obj, ball_obj, solver=solver, dt=dt)
    display_results(shell_data, ball_data)


if __name__ == "__main__":
    main()

