"""Small sag cable functions."""

import numpy as np
from scipy.optimize import root

from slenderpy.future import floatArrayLike
from slenderpy.future._constant import _GRAVITY
from slenderpy.future.cable.static import blondel


def _f(z: float) -> float:
    """Primitive used to compute the length of a parabola."""
    q = np.sqrt(1 + z**2)
    return 0.5 * (z * q + np.log(z + q))


def _g(x, a, b):
    """Primitive used to compute an integral."""
    y = x + b
    s = np.sqrt(a + y**2)
    return 0.5 * (y * s + a * np.log(y + s))


def shape(
    x: floatArrayLike,
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
) -> floatArrayLike:
    """Parabola equation for cable.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    x : horizontal position (m, should be in [0, lspan] range)
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    g : gravitational acceleration (m.s**-2)

    Returns
    -------
    Vertical position of the cable (m). Return array has the same size as
    the given inputs.

    """
    a = tension / (linm * g)
    return 0.5 * x / a * (x + 2.0 * a * sld / lspan - lspan)


def length(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
) -> floatArrayLike:
    """Compute suspended cable length using a parabola model.

    If more than one arg is an array, they must have the same size (no check).

    An approximation of the result is:
        l * [1 + (h/l)**2 / 2 + (l/a)**2 / 24]
    with
        - a = tension / (linm * g)
        - l = lspan
        - h = sld

    Parameters
    ----------
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    g : gravitational acceleration (m.s**-2)

    Returns
    -------
    An estimation of the cable length (m). Return array has the same size as
    the given inputs.

    """
    a = tension / (linm * g)
    z1 = sld / lspan - 0.5 * lspan / a
    z2 = lspan / a + z1
    return a * (_f(z2) - _f(z1))


def argsag(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
) -> floatArrayLike:
    """Sag position.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    g : gravitational acceleration (m.s**-2)

    Returns
    -------
    Horizontal position of the cable's lowest point (m). Return array has the
    same size as the given inputs.

    """
    return np.minimum(
        np.maximum(0.5 * lspan - tension * sld / (linm * g * lspan), 0.0), lspan
    )


def sag(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
) -> floatArrayLike:
    """Cable sag.

    The sag is the vertical distance between the lowest point of the cable and
    the  line that joins the two suspensions points. When the support level
    difference is important, it can be equal to zero (the lowest point is one of
    the anchor points).

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    g : gravitational acceleration (m.s**-2)

    Returns
    -------
    Sag value (m). Return array has the same size as the given inputs.

    """
    # NB : exact formula -> put this in docstring ?
    # a = tension / (linm * g)
    # l = lspan
    # h = sld
    # sag = 0.5 / a * (l / 2 - a*h/l) * (l / 2 + a*h/l)
    x0 = argsag(lspan, tension, sld, linm, g=g)
    return sld * x0 / lspan - shape(x0, lspan, tension, sld, linm)


def max_chord(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
) -> floatArrayLike:
    """Maximum value taken by chord length.

    A chord is a vertical line between a point on the cable and the line that
    joins the two suspensions points. The maximum chord length is the largest
    chord possible. It is often used as an approximation for sag (and equal to
    sag if sld=0).

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    g : gravitational acceleration (m.s**-2)

    Returns
    -------
    Max chord length (m) Return array has the same size as the given inputs.

    """
    return 0.125 * linm * g * lspan**2 / tension


def stress(
    x: floatArrayLike,
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
) -> floatArrayLike:
    """Stress modulus along cable.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    x : horizontal position (m, should be in [0, lspan] range)
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    g : gravitational acceleration (m.s**-2)

    Returns
    -------
    Stress along x position (N). Return array has the same size as
    the given inputs.

    """
    a = tension / (linm * g)
    b = 0.5 * lspan / a - sld / lspan
    N = tension * np.sqrt(1.0 + (x / a - b) ** 2)
    return N


def mean_stress(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
) -> floatArrayLike:
    """Average stress in cable.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    g : gravitational acceleration (m.s**-2)

    Returns
    -------
    Average stress (N). Return array has the same size as the given inputs.

    """
    a = tension / (linm * g)
    b = 0.5 * lspan / a - sld / lspan
    N = tension * a / lspan * (_f(lspan / a - b) - _f(-b))
    return N


_RTOL = 1.0e-12
_MAXITER = 16


def thermal_expansion_tension(
    lspan: floatArrayLike,
    tension_i: floatArrayLike,
    sld: floatArrayLike,
    temperature_i: floatArrayLike,
    temperature_f: floatArrayLike,
    linm_i: floatArrayLike,
    alpha: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
):
    """Compute new tension with temperature change.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension_i : initial mechanical tension (N)
    sld : support level difference (m)
    temperature_i : initial temperature of cable (K)
    temperature_f : final temperature of cable (K)
    linm_i : initial linear mass (kg.m**-1)
    axs : axial stiffness (N)
    alpha : thermal expansion coefficient (K**-1)
    g : gravitational acceleration (m.s**-2)
    rtol : relative tolerance for quasi-newton algorithm
    maxiter : maximum number of iterations in quasi-newton algorithm

    Returns
    -------
    Mechanical tension in final state (N). Return array has the same size as
    the given inputs.

    """
    length_i = length(lspan, tension_i, sld, linm_i, g)
    dl = 1.0 + alpha * (temperature_f - temperature_i)
    length_f = length_i * dl
    weight = linm_i * g * length_i

    def fun(tension):
        linm_f = linm_i * dl
        return np.abs(length(lspan, tension, sld, linm_f, g) - length_f)

    tension_guess = blondel.tension(
        weight, tension_i, temperature_i, temperature_f, 1.0e12, alpha
    )

    sol = root(fun, tension_guess)

    return sol.x


def thermal_expansion_temperature(
    lspan: floatArrayLike,
    tension_i: floatArrayLike,
    tension_f: floatArrayLike,
    sld: floatArrayLike,
    temperature_i: floatArrayLike,
    linm_i: floatArrayLike,
    alpha: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
):
    """Inverse of thermexp_tension, ie compute new temperature with tension change.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension_i : initial mechanical tension (N)
    tension_f : final mechanical tension (N)
    sld : support level difference (m)
    temperature_i : initial temperature of cable (K)
    linm_i : initial linear mass (kg.m**-1)
    axs : axial stiffness (N)
    alpha : thermal expansion coefficient (K**-1)
    g : gravitational acceleration (m.s**-2)
    rtol : relative tolerance for quasi-newton algorithm
    maxiter : maximum number of iterations in quasi-newton algorithm

    Returns
    -------
    Mechanical tension in final state (N). Return array has the same size as
    the given inputs.

    """

    length_i = length(lspan, tension_i, sld, linm_i, g)
    weight = linm_i * g * length_i

    def fun(temperature):
        dl = 1.0 + alpha * (temperature - temperature_i)
        length_f = length_i * dl
        linm_f = linm_i * dl
        return np.abs(length(lspan, tension_f, sld, linm_f, g) - length_f)

    temperature_guess = blondel.temperature(
        weight, tension_i, tension_f, temperature_i, 1.0e12, alpha
    )

    sol = root(fun, temperature_guess)

    return sol.x
