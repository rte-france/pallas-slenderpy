"""Small sag cable functions."""

from typing import Union

import numpy as np
from slenderpy.future._constant import _GRAVITY


def _f(z: float) -> float:
    """Primitive used to compute the length of a parabola."""
    q = np.sqrt(1 + z**2)
    return 0.5 * (z * q + np.log(z + q))


def shape(x: Union[float, np.ndarray], lspan: Union[float, np.ndarray],
          tension: Union[float, np.ndarray], sld: Union[float, np.ndarray],
          linm: Union[float, np.ndarray], g=_GRAVITY) -> Union[float, np.ndarray]:
    """Parabola equation for cable.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    x : curvilinear abscissa along cable (m, before applying load)
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
    return 0.5 * x / a * (x + 2. * a * sld / lspan - lspan)


def length(lspan: Union[float, np.ndarray], tension: Union[float, np.ndarray],
           sld: Union[float, np.ndarray], linm: Union[float, np.ndarray],
           g=_GRAVITY) -> Union[float, np.ndarray]:
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


def argsag(lspan: Union[float, np.ndarray], tension: Union[float, np.ndarray],
           sld: Union[float, np.ndarray], linm: Union[float, np.ndarray],
           g=_GRAVITY) -> Union[float, np.ndarray]:
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
    Curvilinear abscissa (before applying load) of the cable where the lowest
    point is met (m). Return array has the same size as the given inputs.

    """
    return np.minimum(np.maximum(0.5 * lspan - tension * sld / (linm * g * lspan), 0.), lspan)


def sag(lspan: Union[float, np.ndarray], tension: Union[float, np.ndarray],
        sld: Union[float, np.ndarray], linm: Union[float, np.ndarray],
        g=_GRAVITY) -> Union[float, np.ndarray]:
    """Cable sag.

    The sag is the level difference between the lowest point of the cable and
    the lowest anchor point. When the support level difference is important, it
    can be equal to zero (the lowest point is one of the anchor points).

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


def max_chord(lspan: Union[float, np.ndarray], tension: Union[float, np.ndarray],
              sld: Union[float, np.ndarray], linm: Union[float, np.ndarray],
              g=_GRAVITY) -> Union[float, np.ndarray]:
    """Maximum value taken by chord length.

    The maximum chord length is the maximum distance between the cable's lowest
    point and the line that joins the two suspensions points. It is often used
    as an approximation for sag (and equal to sag if sld=0).

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
