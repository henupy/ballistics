"""
Some shiet regarding ballistic trajectories. The simulation tries to be a bit more
realistic by modelling the variation of the density of the earth's atmosphere as a
function of altitude and the variation of the drag coefficient of the projectiles
(of different shapes), which varies as a function of the Reynolds number.

This simulation assumes that the projectile moves freely with some initial
velocity, meaning that the projectile does not have any thrust.
"""

import constants
import numpy as np
import matplotlib.pyplot as plt

from solvers import rk4
from typing import Callable
from projectiledata import ProjectileData
from projectiles import Shell, Sphere, SimObject
from drag_correlations import HolzerSommerfeld


def _get_geopotential_height(h: int | float) -> int | float:
    """
    Converts the geometric height h to the geopotential height as it is defined
    in http://www.braeunig.us/space/atmmodel.htm
    :param h:
    :return:
    """
    # TODO: The model says that we should use the radius of the Earth at the
    # latitude where the object is flying to be more exact
    r0 = constants.r_e  # Mean radius of the earth
    return r0 * h / (r0 + h)


def _lower_atmosphere(h: int | float) -> tuple[float, float, float]:
    """
    Atmospheric data for the lower atmosphere (0 ... 86 km), see
    http://www.braeunig.us/space/atmmodel.htm
    :param h: geopotential height [km]
    :return: Temperature, pressure, density
    """
    # Convert height to geopotential height
    h = _get_geopotential_height(h=h)
    r = constants.r_gas  # Specific gas constant
    a = 34.1632  # A constant used in multiple places
    if h < 11:
        t = 288.15 - 6.5 * h
        p = 101325 * np.power((288.15 / (288.15 - 6.5 * h)), a / -6.5)
    elif 11 <= h < 20:
        t = 216.65
        p = 22632.06 * np.exp(-a * (h - 11) / 216.65)
    elif 20 <= h < 32:
        t = 196.65 + h
        p = 5474.889 * np.power((216.65 / (216.65 + (h - 20))), a)
    elif 32 <= h < 47:
        t = 139.05 + 2.8 * h
        p = 868.0187 * np.power((228.85 / (228.65 + 2.8 * (h - 32))), a / 2.8)
    elif 47 <= h < 51:
        t = 270.65
        p = 110.9063 * np.exp(-a * (h - 47) / 270.65)
    elif 51 <= h < 71:
        t = 413.45 - 2.8 * h
        p = 66.93887 * np.power((270.65 / 270.65 - 2.8 * (h - 51)), a / -2.8)
    elif 71 <= 84.852:
        t = 356.65 - 2 * h
        p = 3.956429 * np.power((214.65 / (214.65 - 2 * (h - 71))), a / -2)
    else:
        raise ValueError("Invalid geopotential height {h}")

    rho = p / (r * t)
    return t, 0, rho


def _base_eq(h: int | float, a: int | float, b: int | float, c: int | float,
             d: int | float, e: int | float) -> float:
    """
    :param h: Geometric height [km]
    :param a:
    :param b:
    :param c:
    :param d:
    :param e:
    :return:
    """
    h2 = h * h
    h3 = h2 * h
    h4 = h3 * h
    return np.exp(a * h4 + b * h3 + c * h2 + d * h + e)


def get_atmos_data(h: int | float) -> tuple[float, float, float]:
    """
    Computes the temperature, pressure, and density of the atmosphere using the
    model defined in http://www.braeunig.us/space/atmmodel.htm
    :param h: Height of the object as measured from the ground [m]
    :return: A tuple of form (temperature, pressure, density)
    """
    if h < 0:
        raise ValueError("Height must be >= 0, now got {h}")
    # Convert height to kilometers
    h *= 1e-3
    a = (h - 120) * (6356.766 + 120) / (6356.766 + h)  # Used in multiple places for temp
    if h < 86:
        return _lower_atmosphere(h=h)
    if 86 <= h < 91:
        t = 186.8673
        pa = 0
        pb = 2.159582e-6
        pc = -4.836957e-4
        pd = -0.1425192
        pe = 13.47530
        rhoa = 0
        rhob = -3.322622e-6
        rhoc = 9.11146e-4
        rhod = -0.2609971
        rhoe = 5.944694
    elif 91 <= h < 100:
        t = 263.1905 - 76.3232 * np.sqrt(1 - np.power((h - 91) / -19.9429, 2))
        pa = 0
        pb = 3.304895e-5
        pc = -0.00906273
        pd = 0.6516698
        pe = -11.03037
        rhoa = 0
        rhob = 2.873405e-5
        rhoc = -0.008492037
        rhod = 0.6541179
        rhoe = -23.6201
    elif 100 <= h < 110:
        t = 263.1905 - 76.3232 * np.sqrt(1 - np.power((h - 91) / -19.9429, 2))
        pa = 0
        pb = 6.693926e-5
        pc = -0.01945388
        pd = 1.71908
        pe = -47.7503
        rhoa = -1.240774e-5
        rhob = 0.005162063
        rhoc = -0.8048342
        rhod = 55.55996
        rhoe = -1443.338
    elif 110 <= h < 120:
        t = 240 + 12 * (h - 110)
        pa = 0
        pb = -6.539316e-5
        pc = 0.02485568
        pd = -3.22362
        pe = 135.9355
        rhoa = 0
        rhob = -8.854164e-5
        rhoc = 0.03373254
        rhod = -4.390837
        rhoe = 176.5294
    elif 120 <= h < 150:
        t = 1000 - 640 * np.exp(-0.01875 * a)
        pa = 2.283506e-7
        pb = -1.343221e-4
        pc = 0.02999016
        pd = -3.055446
        pe = 113.5764
        rhoa = 3.661771e-7
        rhob = -2.154344e-4
        rhoc = 0.04809214
        rhod = -4.884744
        rhoe = 172.3597
    elif 150 <= h < 200:
        t = 1000 - 640 * np.exp(-0.01875 * a)
        pa = 1.209434e-8
        pb = -9.692458e-6
        pc = 0.003002041
        pd = -0.4523015
        pe = 19.19151
        rhoa = 1.906032e-8
        rhob = -1.527799e-5
        rhoc = 0.04724294
        rhod = -0.6992340
        rhoe = 20.50921
    elif 200 <= h < 300:
        t = 1000 - 640 * np.exp(-0.01875 * a)
        pa = 8.113942e-10
        pb = -9.822568e-7
        pc = 4.687616e-4
        pd = -0.1231710
        pe = 3.067409
        rhoa = 1.199282e-9
        rhob = -1.451051e-6
        rhoc = 6.910474e-4
        rhod = -0.1736220
        rhoe = 5.321644
    elif 300 <= h < 500:
        t = 1000 - 640 * np.exp(-0.01875 * a)
        pa = 9.814674e-11
        pb = -1.654439e-7
        pc = 1.148115e-4
        pd = -0.05431334
        pe = -2.011365
        rhoa = 1.140564e-10
        rhob = -2.130756e-7
        rhoc = 1.570762e-4
        rhod = -0.07029296
        rhoe = -12.89844
    elif 500 <= h < 750:
        t = 1000 - 640 * np.exp(-0.01875 * a)
        pa = -7.835161e-11
        pb = 1.964589e-7
        pc = -1.657213e-4
        pd = 0.04305869
        pe = -14.77132
        rhoa = 8.105631e-12
        rhob = -2.358417e-9
        rhoc = -2.635110e-6
        rhod = -0.01562608
        rhoe = -20.02246
    elif 750 <= h < 1000:
        t = 1000 - 640 * np.exp(-0.01875 * a)
        pa = 2.813255e-11
        pb = -1.120689e-7
        pc = 1.695568e-4
        pd = -0.1188941
        pe = 14.56718
        rhoa = -3.701195e-12
        rhob = -8.608611e-9
        rhoc = 5.118829e-5
        rhod = -0.06600998
        rhoe = -6.137674
    else:
        raise ValueError("Invalid height {h}")

    # p = _base_eq(h=h, a=pa, b=pb, c=pc, d=pd, e=pe)
    rho = _base_eq(h=H, a=rhoa, b=rhob, c=rhoc, d=rhod, e=rhoe)
    return t, 0, rho


def _visc(rho: int | float, temp: int | float) -> int | float:
    """
    Calculates the dynamic viscosity of air as a function of density and
    temperature using the correlation found in Journal of Physical and Chemical
    Reference Data 14, 947 (1985).
    :param rho: Air density [kg/m^3]
    :param temp: [K]
    :return:
    """
    temp /= constants.t_star
    rho /= constants.rho_star
    # Calculate the excess viscosity
    bb = [constants.b_1, constants.b_2, constants.b_3, constants.b_4]
    v_excess = 0
    for i, b in enumerate(bb):
        v_excess += b * np.power(rho, i + 1)
    # Calculate the sum term for the "temperature viscosity"
    aa = [constants.a_0, constants.a__1, constants.a__2, constants.a__3,
          constants.a__4]
    sum_term = 0
    for i, a in enumerate(aa):
        sum_term += a * np.power(temp, -i)
    # Calculate the full "temperature viscosity"
    v_temp = constants.a_1 * temp + constants.a_05 * np.power(temp, 0.5) + sum_term
    # Return the total viscosity
    return constants.h * (v_temp + v_excess)


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
    visc = _visc(rho=rho, temp=temp)
    vmag = vec_len(vel)
    return rho * vmag * size / visc


def _grav_force(m: int | float, h: int | float) -> np.ndarray:
    """
    Calculates the acceleration due to gravital attraction
    between the Earth and the projectile
    :param m:
    :param h: Height of the projectile related to Earth's surface [m]
    :return: Acceleration due to gravity as a vector [m/s^2]
    """
    r = constants.r_e + h
    return np.array([0, -constants.big_g * constants.m_e * m / (r * r)])


def _diff_eq(y0: np.ndarray, t: int | float, rho: int | float, area: int | float,
             g_f: int | float, c_d: int | float, m: int | float) -> np.ndarray:
    """
    Differential equation for an object following ballistic trajectory
    :param y0:
    :param t:
    :param
    :return:
    """
    _ = t  # Unused variable, this suppresses the warning
    x, v = y0
    v_len = vec_len(v)
    k = .5 * rho * area * c_d
    dydt = [v, (-k * v_len * v + g_f) / m]
    return np.array(dydt)


def _solve(obj: SimObject, solver: Callable, dt: float,
           max_steps: int | float) -> ProjectileData:
    """
    :return:
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
        g_f = _grav_force(m=proj.m, h=float(pos[n, 1]))
        temp, _, rho = get_atmos_data(h=pos[n, 1])
        re = reynolds(proj.size, rho=rho, temp=temp, vel=vel[n])
        rey.append(re)
        c_d = obj.drag_corr.eval(re=re)
        cd.append(c_d)
        t = dt * (n + 1)
        n_pos, n_vel = solver(diff_eq=_diff_eq, y0=np.vstack((pos[n], vel[n])), t=t,
                              dt=dt, rho=rho, area=proj.proj_area, c_d=c_d, g_f=g_f,
                              m=proj.m)
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
    :param args:
    :param solver:
    :param dt: Timestep [s]
    :param max_steps: A hard limit to the number of timesteps to simulate so that
    the simulation does not run too long
    :return: Array of x- and y-coordinates [m]
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

