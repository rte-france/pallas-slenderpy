"""Analytic natural frequencies of a taut beam under different end conditions.

The functions take plain float or array arguments (span length, tension, mass
per unit length and bending stiffness); object-oriented wrappers call them
later. Three models are provided: a vibrating string (no bending stiffness), a
pinned beam (rotation free at both ends) and a clamped beam (rotation blocked at
both ends).
"""

import numpy as np
import scipy as sp

from slenderpy.future import floatArrayLike


def natural_frequencies(
    length: floatArrayLike,
    tension: floatArrayLike,
    mass: floatArrayLike,
    n: int,
) -> np.ndarray:
    """Compute the n first natural frequencies of the vibrating string (Hz)."""
    return 0.5 * np.linspace(1, n, n) / length * np.sqrt(tension / mass)


def natural_frequency(
    length: floatArrayLike,
    tension: floatArrayLike,
    mass: floatArrayLike,
) -> float:
    """Compute the fundamental natural frequency of the vibrating string (Hz)."""
    return natural_frequencies(length, tension, mass, n=1)[0]


def natural_frequencies_hinged(
    length: floatArrayLike,
    tension: floatArrayLike,
    mass: floatArrayLike,
    ei: floatArrayLike,
    n: int,
) -> np.ndarray:
    """Compute the n first natural frequencies for a pinned beam (Hz).

    Rotation is free at both ends; ``ei`` is the bending stiffness.
    """
    ep = ei / (tension * length**2)
    nn = np.linspace(1, n, n)
    Wn = nn * np.sqrt(1.0 + ep * (np.pi * nn) ** 2)
    return Wn * natural_frequency(length, tension, mass)


def natural_frequencies_clamped(
    length: floatArrayLike,
    tension: floatArrayLike,
    mass: floatArrayLike,
    ei: floatArrayLike,
    n: int,
) -> np.ndarray:
    """Compute the n first natural frequencies for a clamped beam (Hz).

    Rotation is blocked at both ends; ``ei`` is the bending stiffness. The
    frequencies solve a transcendental equation, computed here with a Newton
    iteration.
    """
    ep = ei / (tension * length**2)
    f0 = natural_frequency(length, tension, mass)
    nn = np.linspace(1, n, n)

    Wg = 1.0 + np.sqrt(ep) + (1.0 + 0.5 * (np.pi * nn) ** 2) * ep
    rs = np.zeros_like(nn)

    def sqe(x):
        return np.sqrt(0.25 / ep**2 + np.pi**2 * x**2 / ep)

    def k1L(x):
        return np.sqrt(sqe(x) + 0.5 / ep)

    def k2L(x):
        return np.sqrt(sqe(x) - 0.5 / ep)

    def fun(x):
        return np.tan(k2L(x)) - k2L(x) / k1L(x)

    def dk1(x):
        return x * np.pi**2 / (ep * sqe(x) * 2 * k1L(x))

    def dk2(x):
        return x * np.pi**2 / (ep * sqe(x) * 2 * k2L(x))

    def dfn(x):
        return (
            dk2(x) / np.cos(k2L(x)) ** 2
            + (dk2(x) * k1L(x) - dk1(x) * k2L(x)) / k1L(x) ** 2
        )

    for k in range(n):
        rs[k] = sp.optimize.newton(fun, Wg[k], fprime=dfn)

    return f0 * nn * rs
