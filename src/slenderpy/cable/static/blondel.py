"""Using Blondel formulae to compute changes of tension or temperature."""

import numpy as np
from pyntb.polynomial import solve_p3_v

from slenderpy import floatArrayLike


def tension(
    weight: floatArrayLike,
    tension_i: floatArrayLike,
    temperature_i: floatArrayLike,
    temperature_f: floatArrayLike,
    axs: floatArrayLike,
    alpha: floatArrayLike,
) -> floatArrayLike:
    """Compute new tension with temperature change.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    weight
        Cable weight (N).
    tension_i
        Initial mechanical tension (N).
    temperature_i
        Initial temperature of cable (K).
    temperature_f
        Final temperature of cable (K).
    axs
        Axial stiffness (N).
    alpha
        Thermal expansion coefficient (K**-1).

    Returns
    -------
    float or array
        Mechanical tension in final state (N). Same shape as the broadcast
        inputs.

    """
    a = 1.0 / axs
    b = (
        alpha * (temperature_f - temperature_i)
        - tension_i / axs
        + (weight / tension_i) ** 2 / 24
    )
    c = 0.0
    d = -(weight**2) / 24

    # a > 0 and d < 0: by Descartes' rule of signs the cubic has exactly one
    # positive root, and solve_p3_v returns it first (the only real root, or the
    # largest of three). It evaluates sqrt(delta) on both branches of a
    # np.where, hence the ignored warning when delta < 0
    with np.errstate(invalid="ignore"):
        tension_f, _, _ = solve_p3_v(a, b, c, d)

    return tension_f


def temperature(
    weight: floatArrayLike,
    tension_i: floatArrayLike,
    tension_f: floatArrayLike,
    temperature_i: floatArrayLike,
    axs: floatArrayLike,
    alpha: floatArrayLike,
) -> floatArrayLike:
    """Inverse of tension function, ie compute new temperature given tension change.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    weight
        Cable weight (N).
    tension_i
        Initial mechanical tension (N).
    tension_f
        Final mechanical tension (N).
    temperature_i
        Initial temperature of cable (K).
    axs
        Axial stiffness (N).
    alpha
        Thermal expansion coefficient (K**-1).

    Returns
    -------
    float or array
        Final temperature of cable (K). Same shape as the broadcast inputs.

    """
    return (
        temperature_i
        + (
            weight**2 / 24 * (1.0 / tension_f**2 - 1.0 / tension_i**2)
            - (tension_f - tension_i) / axs
        )
        / alpha
    )
