"""Catenary equation for suspended cables."""

from typing import Tuple

import numpy as np

from slenderpy.future import floatArrayLike
from slenderpy.future._constant import _GRAVITY


def _mechparam(
    tension: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
) -> floatArrayLike:
    return tension / (linm * g)


def _qfactor(l, h):
    return np.log((l + h) / (l - h))


def _length(ls, a, sld):
    return np.sqrt(sld**2 + (2.0 * a * np.sinh(0.5 * ls / a)) ** 2)


def length(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
):
    a = _mechparam(tension, linm, g=g)
    l = _length(lspan, a, sld)
    return l


def _lax0(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
) -> Tuple[floatArrayLike, floatArrayLike, floatArrayLike]:
    a = _mechparam(tension, linm, g=g)
    l = _length(lspan, a, sld)
    q = _qfactor(l, sld)
    x0 = a * q - lspan
    return l, a, x0


def shape(
    x: floatArrayLike,
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
) -> floatArrayLike:
    _, a, x0 = _lax0(lspan, tension, sld, linm, g)
    return 2.0 * a * np.sinh(0.5 * (x + x0) / a) * np.sinh(0.5 * x / a)


def argsag(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
) -> floatArrayLike:
    _, _, x0 = _lax0(lspan, tension, sld, linm, g)
    return np.minimum(np.maximum(-0.5 * x0, 0.0), lspan)


def sag(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
) -> floatArrayLike:
    x = argsag(lspan, tension, sld, linm, g)
    return sld * x / lspan - shape(x, lspan, tension, sld, linm, g=g)


def max_chord(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
) -> floatArrayLike:
    _, a, x0 = _lax0(lspan, tension, sld, linm, g)
    x = a * np.arcsinh(sld / lspan) - 0.5 * x0
    return sld * x / lspan - shape(x, lspan, tension, sld, linm, g=g)


def stress():
    raise NotImplementedError


def mean_stress():
    raise NotImplementedError


def thermal_expansion_tension():
    raise NotImplementedError


def thermal_expansion_temperature():
    raise NotImplementedError
