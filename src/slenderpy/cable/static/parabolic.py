"""Small sag cable functions."""

import numpy as np
from scipy.optimize import root

from slenderpy import floatArrayLike
from slenderpy._constant import _GRAVITY
from slenderpy.cable.static import blondel


def _f(z: float) -> float:
    """Primitive used to compute the length of a parabola."""
    q = np.sqrt(1 + z**2)
    return 0.5 * (z * q + np.log(z + q))


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
    x
        Horizontal position (m, should be in [0, lspan] range).
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    g
        Gravitational acceleration (m.s**-2).

    Returns
    -------
    float or array
        Vertical position of the cable (m). Same shape as the broadcast inputs.

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
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    g
        Gravitational acceleration (m.s**-2).

    Returns
    -------
    float or array
        An estimation of the cable length (m). Same shape as the broadcast inputs.

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
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    g
        Gravitational acceleration (m.s**-2).

    Returns
    -------
    float or array
        Horizontal position of the cable's lowest point (m). Same shape as the broadcast inputs.

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
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    g
        Gravitational acceleration (m.s**-2).

    Returns
    -------
    float or array
        Sag value (m). Same shape as the broadcast inputs.

    """
    # NB : exact formula -> put this in docstring ?
    # a = tension / (linm * g)
    # l = lspan
    # h = sld
    # sag = 0.5 / a * (l / 2 - a*h/l) * (l / 2 + a*h/l)
    x0 = argsag(lspan, tension, sld, linm, g=g)
    return sld * x0 / lspan - shape(x0, lspan, tension, sld, linm, g=g)


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
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    g
        Gravitational acceleration (m.s**-2).

    Returns
    -------
    float or array
        Max chord length (m). Same shape as the broadcast inputs.

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
    x
        Horizontal position (m, should be in [0, lspan] range).
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    g
        Gravitational acceleration (m.s**-2).

    Returns
    -------
    float or array
        Stress along x position (N). Same shape as the broadcast inputs.

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
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    g
        Gravitational acceleration (m.s**-2).

    Returns
    -------
    float or array
        Average stress (N). Same shape as the broadcast inputs.

    """
    a = tension / (linm * g)
    b = 0.5 * lspan / a - sld / lspan
    N = tension * a / lspan * (_f(lspan / a - b) - _f(-b))
    return N


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
    lspan
        Span length (m).
    tension_i
        Initial mechanical tension (N).
    sld
        Support level difference (m).
    temperature_i
        Initial temperature of cable (K).
    temperature_f
        Final temperature of cable (K).
    linm_i
        Initial linear mass (kg.m**-1).
    alpha
        Thermal expansion coefficient (K**-1).
    g
        Gravitational acceleration (m.s**-2).

    Returns
    -------
    float or array
        Mechanical tension in final state (N). Same shape as the broadcast inputs.

    """
    length_i = length(lspan, tension_i, sld, linm_i, g)
    dl = 1.0 + alpha * (temperature_f - temperature_i)
    length_f = length_i * dl
    weight = linm_i * g * length_i

    def fun(tension):
        linm_f = linm_i / dl
        return length(lspan, tension, sld, linm_f, g) - length_f

    tension_guess = blondel.tension(
        weight, tension_i, temperature_i, temperature_f, 1.0e12, alpha
    )

    sol = root(fun, tension_guess)

    return np.reshape(sol.x, np.shape(tension_guess))


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
    """Inverse of thermal_expansion_tension: new temperature from a tension change.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan
        Span length (m).
    tension_i
        Initial mechanical tension (N).
    tension_f
        Final mechanical tension (N).
    sld
        Support level difference (m).
    temperature_i
        Initial temperature of cable (K).
    linm_i
        Initial linear mass (kg.m**-1).
    alpha
        Thermal expansion coefficient (K**-1).
    g
        Gravitational acceleration (m.s**-2).

    Returns
    -------
    float or array
        Temperature in final state (K). Same shape as the broadcast inputs.

    """

    length_i = length(lspan, tension_i, sld, linm_i, g)
    weight = linm_i * g * length_i

    def fun(temperature):
        dl = 1.0 + alpha * (temperature - temperature_i)
        length_f = length_i * dl
        linm_f = linm_i / dl
        return length(lspan, tension_f, sld, linm_f, g) - length_f

    temperature_guess = blondel.temperature(
        weight, tension_i, tension_f, temperature_i, 1.0e12, alpha
    )

    sol = root(fun, temperature_guess)

    return np.reshape(sol.x, np.shape(temperature_guess))
