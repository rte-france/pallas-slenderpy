"""Small sag cable functions."""

from typing import Union

import numpy as np

from slenderpy.future._constant import _GRAVITY


def _f(z: float) -> float:
    """Primitive used to compute the length of a parabola."""
    q = np.sqrt(1 + z**2)
    return 0.5 * (z * q + np.log(z + q))


def _g(x, a, b):
    """Primitive used to compute an integral."""
    y = x + b
    s = np.sqrt(a + y**2)
    return 0.5 * (y * s + a * np.log(y + s))


def shape(x: Union[float, np.ndarray], lspan: Union[float, np.ndarray],
          tension: Union[float, np.ndarray], sld: Union[float, np.ndarray],
          linm: Union[float, np.ndarray], g=_GRAVITY) -> Union[float, np.ndarray]:
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
    Horizontal position of the cable's lowest point (m). Return array has the
    same size as the given inputs.

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


def stress(x: Union[float, np.ndarray], lspan: Union[float, np.ndarray],
           tension: Union[float, np.ndarray], sld: Union[float, np.ndarray],
           linm: Union[float, np.ndarray], g=_GRAVITY) -> Union[float, np.ndarray]:
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
    N = tension * np.sqrt(1. + (x / a - b)**2)
    return N


def mean_stress(lspan: Union[float, np.ndarray], tension: Union[float, np.ndarray],
                sld: Union[float, np.ndarray], linm: Union[float, np.ndarray],
                g=_GRAVITY) -> Union[float, np.ndarray]:
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
    linw = linm * g
    a_ = (tension / linw)**2
    b_ = -(0.5 * lspan - tension * sld / (linw * lspan))
    N = linw / lspan * (_g(lspan, a_, b_) - _g(0., a_, b_))
    return N
