"""
Some shiet regarding ballistic trajectories. The simulation tries to be a bit more
realistic by modelling the variation of the density of the earth's atmosphere as a
function of altitude and the variation of the drag coefficient of the projectiles
(of different shapes), which varies as a function of the Reynolds number.

This simulation assumes that the projectile moves freely with some initial
velocity, meaning that the projectile does not have any thrust.
"""

import atmos
import utils
import constants
import numpy as np
import matplotlib.pyplot as plt

from solvers import rk4
from typing import Callable
from projectiledata import ProjectileData
from projectiles import Projectile, Shell, Sphere
from drag_correlations import HaiderLevenspiel, HolzerSommerfeld


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


def _solve(proj: Projectile, solver: Callable, dt: float,
           max_steps: int | float) -> ProjectileData:
    """
    :param proj: Some Projectile object
    :param solver: The solver used to solve the differential equation of motion
    :param dt: Timestep size [s]
    :param max_steps: Maximum amount of timesteps to take
    :return: A ProjectileData object containing the simulation results
    """
    pos = np.zeros(shape=(1, 2))
    vel = np.zeros(shape=(1, 2))
    pos[0] = proj.p0
    vel[0] = proj.v0
    cd = []
    rey = []
    n = 0
    while n <= max_steps:
        grav_f = utils.grav_force(m=proj.m, h=float(pos[n, 1]))
        temp, _, rho = atmos.get_atmos_data(h=pos[n, 1])
        re = utils.reynolds(proj.size, rho=rho, temp=temp, vel=vel[n])
        rey.append(re)
        c_d = proj.get_cd(re=re)
        cd.append(c_d)
        drag_f = utils.drag_force(rho=rho, area=proj.proj_area, c_d=c_d, vel=vel[n])
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
    return ProjectileData(proj=proj, coords=pos, vel=vel, cd=np.array(cd),
                          rey=np.array(rey), dt=dt)


def simulate(*args: Projectile, solver: Callable, dt: int | float,
             max_steps: int = 1e5) -> list[ProjectileData]:
    """
    Calculates the trajectory of the projectile with air resistance, if
    all necessary parameters are provided with the ProjectileData object.
    Otherwise air resistance is neglected. The simulation is continued until
    the projectile hits the ground or the max_steps amount of timesteps are
    simulated.
    :param args: Arbitrary amount of Projectiles to run the simulation for
    :param solver: The solver function used to solve the differential equation
    of motion
    :param dt: Timestep [s]
    :param max_steps: A hard limit to the number of timesteps to simulate so that
    the simulation does not run too long
    :return: List of ProjectileData objects containing the simulation results
    for all given args
    """
    data_objs = []
    for proj in args:
        data_obj = _solve(proj=proj, solver=solver, dt=dt, max_steps=max_steps)
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
    v0_1 = utils.speed2velocity(speed=v0_mag, angle=angle)
    v0_2 = utils.speed2velocity(speed=v0_mag, angle=angle)
    dt = .01  # Timestep [s]

    # Create some projectiles and stuff
    shell_corr = HolzerSommerfeld
    ball_corr = HaiderLevenspiel 
    shell = Shell(m=m, p0=p0_1, v0=v0_1, d=d, drag_corr=shell_corr, name="Shell")
    ball = Sphere(m=m, p0=p0_2, v0=v0_2, r=d / 2, drag_corr=ball_corr, name="Sphere")

    # Simulate
    solver = rk4
    shell_data, ball_data = simulate(shell, ball, solver=solver, dt=dt)
    display_results(shell_data, ball_data)


if __name__ == "__main__":
    main()

