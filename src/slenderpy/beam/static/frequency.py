"""Analytic natural frequencies of a taut beam under different end conditions.

The functions take plain float or array arguments (span length, tension, mass
per unit length and bending stiffness); object-oriented wrappers call them
later. Three models are provided: a vibrating string (no bending stiffness), a
pinned beam (rotation free at both ends) and a clamped beam (rotation blocked at
both ends).
"""

import numpy as np
import scipy as sp

from slenderpy import floatArrayLike


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
    length: float,
    tension: float,
    mass: float,
    ei: float,
    n: int,
) -> np.ndarray:
    """Compute the n first natural frequencies for a clamped beam (Hz).

    Rotation is blocked at both ends; ``ei`` is the bending stiffness. With
    ``ep = ei / (tension * length**2)``, the mode shapes combine ``cosh(a s)``,
    ``sinh(a s)``, ``cos(b s)`` and ``sin(b s)`` (``s`` in [0, 1]) with
    ``a**2 = b**2 + 1/ep``, and the clamped-clamped condition is

        2ab(1 - cosh a cos b) + (a**2 - b**2) sinh a sin b = 0,

    solved here divided by ``cosh a`` to avoid overflow. The frequency is
    ``b * sqrt(1 + ep * b**2) / pi`` times the string fundamental; ``b = n pi``
    gives the pinned beam, and the n-th clamped root lies in
    ``(n pi, (n + 1) pi)``, where it is bracketed.
    """
    ep = ei / (tension * length**2)
    f0 = natural_frequency(length, tension, mass)

    def determinant(b):
        a = np.sqrt(b**2 + 1.0 / ep)
        # 1 / cosh(a) written with exp(-a), which does not overflow
        sech = 2.0 * np.exp(-a) / (1.0 + np.exp(-2.0 * a))
        return 2.0 * a * b * (sech - np.cos(b)) + (a**2 - b**2) * np.tanh(a) * np.sin(b)

    b = np.array(
        [
            sp.optimize.brentq(determinant, k * np.pi, (k + 1) * np.pi, xtol=1.0e-14)
            for k in range(1, n + 1)
        ]
    )
    return f0 * b * np.sqrt(1.0 + ep * b**2) / np.pi
