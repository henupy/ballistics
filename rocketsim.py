"""
Ballistic simulation for some kind of a rocket or missile or some shit. Takes into
account the changing atmospheric properties as a function of altitude, and the Earth's
curvature is also taken into account (at some point maybe). The projectile can have
some thrust, which can also be time-dependent. The depletion of the mass during
flight is also taken into account, on some crude level of detail.
"""

import atmos
import utils
import functools
import numpy as np
import matplotlib.pyplot as plt

from solvers import rk4
from typing import Callable
from projectiles import Rocket
from projectiledata import ProjectileData
from drag_correlations import HaiderLevenspiel, HolzerSommerfeld


def flight_angle(t: int | float) -> int | float:
    """
    Returns the angle that the rocket is pointing towards as a function of time,
    90 degrees is directly away from the Earth, 0 degrees is parallel to ground.
    :param t: Flight time [s]
    :return: 
    """
    if t < 30:
        return 90
    t -= 30
    turning_rate = .25  # [deg/s]
    return max(10, 90 - turning_rate * t)


def f9_mass_fun(t: int | float, payload: int | float) -> int | float:
    """
    Returns the mass of a Spacex Falcon 9 rocket as a function
    of flight time

    Specs from from https://en.wikipedia.org/wiki/Falcon_9
    :param t: Flight time [s]
    :param payload: Payload mass [kg]
    :return:
    """
    if t < 0:
        raise ValueError(f"Time must be positive, now got {t}")
    stage1_empty_mass = 25600  # [kg]
    stage1_fuel_mass = 395700
    stage1_burn_time = 162  # [s]
    stage2_empty_mass = 3900
    stage2_fuel_mass = 92670
    stage2_burn_time = 397
    total = stage1_empty_mass + stage1_fuel_mass + stage2_empty_mass + \
        stage2_fuel_mass + payload
    # Stage 1
    if t <= stage1_burn_time:
        return total - t * (stage1_fuel_mass / stage1_burn_time)
    # Stage 2, stage 1 has been dropped
    elif stage1_burn_time < t <= (stage1_burn_time + stage2_burn_time):
        t -= stage1_burn_time
        loss_rate = stage2_fuel_mass / stage2_burn_time
        return (stage2_empty_mass + stage2_fuel_mass + payload) - t * loss_rate
    else:
        return stage2_empty_mass + payload


def f9_thrust_fun(t: int | float, angle_fun: Callable) -> np.ndarray:
    """
    Returns the thrust of a Spacex Falcon 9 rocket as a function of flight time

    Specs from from https://en.wikipedia.org/wiki/Falcon_9
    :param t: Flight time [s]
    :param angle_fun: Function that returns the rocket direction as a function of
    time
    :return:
    """
    if t < 0:
        raise ValueError(f"Time must be positive, now got {t}")
    stage1_burn_time = 162  # [s]
    stage2_burn_time = 397
    stage1_thrust = 7607e3  # At sea level [N]
    stage2_thrust = 981e3  # [N]
    angle = angle_fun(t)
    direc = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    if t <= stage1_burn_time:
        return direc * stage1_thrust
    elif stage1_burn_time < t <= (stage1_burn_time + stage2_burn_time):
        return direc * stage2_thrust
    else:
        return 0


def _diff_eq(y0: np.ndarray, t: int | float, m: int | float, drag: int | float,
             gravity: int | float, thrust: int | float) -> np.ndarray:
    """
    Differential equation for an object following ballistic trajectory
    :param y0: Initial conditions as a vector of vectors [position, velocity]
    :param t: Time [s] (unused)
    :param m: Mass of the projectile [kg]
    :param drag: Drag force affecting the projectile [N]
    :param gracity: Gravitational force affecting the projectile [N]
    :param thrust: Thrust force [N]
    :return:
    """
    _ = t  # Unused variable, this suppresses the warning
    _, v = y0
    dydt = [v, (drag + gravity + thrust) / m]
    return np.array(dydt)


def _solve(proj: Rocket, solver: Callable, dt: float,
           time: int | float) -> ProjectileData:
    """
    :param proj: A Rocket object
    :param solver: The solver used to solve the differential equation of motion
    :param dt: Timestep size [s]
    :param time: Simulation time [s]
    :return: A ProjectileData object containing the simulation results
    """
    steps = int(time / dt)
    pos = np.zeros(shape=(steps, 2), dtype=float)
    vel = np.zeros(shape=(steps, 2), dtype=float)
    pos[0] = proj.p0
    for n in range(1, steps):
        t = dt * n
        mass = proj.mass_fun(t)
        thrust = proj.thrust_fun(t)
        grav_f = utils.grav_force(m=mass, h=float(pos[n - 1, 1]))
        temp, _, rho = atmos.get_atmos_data(h=pos[n - 1, 1])
        re = utils.reynolds(proj.size, rho=rho, temp=temp, vel=vel[n - 1])
        c_d = proj.get_cd(re=re)
        drag_f = utils.drag_force(rho=rho, area=proj.proj_area, c_d=c_d, vel=vel[n - 1])
        n_pos, n_vel = solver(diff_eq=_diff_eq, y0=np.vstack((pos[n - 1], vel[n - 1])), t=t,
                              dt=dt, m=mass, drag=drag_f, gravity=grav_f, thrust=thrust)
        pos[n] = n_pos
        vel[n] = n_vel
        if pos[n, 1] < 0:
            print(f"INFO: Simulation ended early due to height being < 0")
            break

    return ProjectileData(proj=proj, coords=pos[:n], vel=vel[:n], dt=dt)


def simulate(*args: Rocket, solver: Callable, dt: int | float,
             time: int | float) -> list[ProjectileData]:
    """
    Calculates the trajectory of the projectile. The simulation is continued until
    the projectile hits the ground or the max_steps amount of timesteps are
    simulated.
    :param args: Arbitrary amount of Rocket objects to run the simulation for
    :param solver: The solver function used to solve the differential equation
    of motion
    :param dt: Timestep [s]
    :param time: The length of the simulation time [s]
    :return: List of ProjectileData objects containing the simulation results
    for all given args
    """
    data_objs = []
    for proj in args:
        data_obj = _solve(proj=proj, solver=solver, dt=dt, time=time)
        data_objs.append(data_obj)
    return data_objs


def display_results(*args: ProjectileData) -> None:
    """
    Prints out and plots some key characteristics of the trajectory
    :param args:
    :return:
    """
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for data_obj in args:
        tspan = np.linspace(0, data_obj.time, int(data_obj.time / data_obj.dt))
        print(f"    Flight data for {data_obj.proj}:")
        print(f"    Total distance (in x-direction): {data_obj.x_dist:.3f} m")
        print(f"    Highest point: {data_obj.y_max:.3f} m")
        print(f"    Flight time: {data_obj.time:.3f} s")
        print()
        plt.figure(fig1)
        plt.plot(tspan, data_obj.coords[:, 1] / 1e3,
                 label=f"{data_obj.proj}")
        plt.figure(fig2)
        plt.plot(tspan, data_obj.vel[:, 0], label=f"X velocity, {data_obj.proj}")
        plt.plot(tspan, data_obj.vel[:, 1], label=f"Y velocity, {data_obj.proj}")
        plt.plot(tspan, data_obj.speed, linestyle="--",
                 label=f"Total velocity, {data_obj.proj}")

    ax1.set_title("Altitude as a function of time")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("y [km]")
    ax1.legend()
    ax1.grid()

    ax2.set_title("Velocities and speed as a function of time")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Speed [m/s]")
    ax2.legend()
    ax2.grid()

    plt.show()


def main() -> None:
    # Initial conditions and shit
    p0 = np.array([0, 0])  # Initial position [m]
    d = 3.7  # Diameter [m]
    length = 69.8  # Total length [m]
    payload = 20e3  # [kg]
    drag_corr = HolzerSommerfeld
    dt = 1  # [s]
    time = 60 * 120 # [s]
    solver = rk4

    # Projectile creation with functions and stuff
    m_fun = functools.partial(f9_mass_fun, payload=payload)
    t_fun = functools.partial(f9_thrust_fun, angle_fun=flight_angle)
    proj = Rocket(p0=p0, d=d, length=length, mass_fun=m_fun, thrust_fun=t_fun,
                  drag_corr=drag_corr, name="Falcon 9")
   
    # Simulate
    flight_data = simulate(proj, solver=solver, dt=dt, time=time)
    display_results(flight_data[0])



if __name__ == "__main__":
    main()

