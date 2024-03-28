"""Using Blondel formulae to compute changes of tension or temperature."""

import numpy as np
from pyntb.polynomial import solve_p3_v
from typing import Union


def tension(weight: Union[float, np.ndarray],
            tension_i: Union[float, np.ndarray],
            temperature_i: Union[float, np.ndarray],
            temperature_f: Union[float, np.ndarray],
            axs: Union[float, np.ndarray],
            alpha: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute new tension with temperature change.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    weight : cable weight (N)
    tension_i : initial mechanical tension (N)
    temperature_i : initial temperature of cable (K)
    temperature_f : final temperature of cable (K)
    axs : axial stiffness (N)
    alpha : thermal expansion coefficient (K**-1)

    Returns
    -------
    Mechanical tension in final state (N). Return array has the same size as
    the given inputs.

    """
    a = 1. / axs
    b = alpha * (temperature_f - temperature_i) - tension_i / axs + (weight / tension_i)**2 / 24
    c = 0.
    d = - weight**2 / 24
    tension_f, _, _ = solve_p3_v(a, b, c, d)
    return tension_f


def temperature(weight: Union[float, np.ndarray],
                tension_i: Union[float, np.ndarray],
                tension_f: Union[float, np.ndarray],
                temperature_i: Union[float, np.ndarray],
                axs: Union[float, np.ndarray],
                alpha: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Inverse of tension function, ie compute new temperature given tension change.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    weight : cable weight (N)
    tension_i : initial mechanical tension (N)
    tension_f : final mechanical tension (N)
    temperature_i : initial temperature of cable (K)
    axs : axial stiffness (N)
    alpha : thermal expansion coefficient (K**-1)

    Returns
    -------
    Final temperature of cable (K). Return array has the same size as the given
    inputs.

    """
    return temperature_i + (weight**2 / 24 * (1. / tension_f**2 - 1. / tension_i**2) -
                            (tension_f - tension_i) / axs) / alpha
