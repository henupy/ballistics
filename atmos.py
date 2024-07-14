"""
File that contains the atmospheric "model", i.e., function(s) for computing
atmospheric temperature, density, pressure, and viscosity as a function of
altitude.

References:
http://www.braeunig.us/space/atmmodel.htm

Journal of Physical and Chemical Reference Data 14, 947 (1985)
"""

import constants
import numpy as np


def _get_geopotential_height(h: int | float) -> int | float:
    """
    Converts the geometric height h to the geopotential height as it is defined
    in http://www.braeunig.us/space/atmmodel.htm
    :param h:
    :return:
    """
    # TODO: The model says that we should use the radius of the Earth at the
    # latitude where the object is flying to be more exact
    r0 = constants.r_e * 1e-3  # Mean radius of the Earth [km]
    return r0 * h / (r0 + h)


def _lower_atmosphere(h: int | float) -> tuple[float, float, float]:
    """
    Atmospheric data for the lower atmosphere (0 ... 86 km), see
    http://www.braeunig.us/space/atmmodel.htm
    :param h: Geometric height [km]
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
        p = 66.93887 * np.power((270.65 / (270.65 - 2.8 * (h - 51))), a / -2.8)
    elif 71 <= 84.852:
        t = 356.65 - 2 * h
        p = 3.956429 * np.power((214.65 / (214.65 - 2 * (h - 71))), a / -2)
    else:
        raise ValueError(f"Invalid geopotential height {h}")

    rho = p / (r * t)
    return t, p, rho


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
        rhoc = 0.004724294
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
        rhoe = -5.321644
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
        # Return some semi random stuff for h > 1000 km
        return 1000., 0., 0.

    p = _base_eq(h=h, a=pa, b=pb, c=pc, d=pd, e=pe)
    rho = _base_eq(h=h, a=rhoa, b=rhob, c=rhoc, d=rhod, e=rhoe)
    return t, p, rho


def visc(rho: int | float, temp: int | float) -> int | float:
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

