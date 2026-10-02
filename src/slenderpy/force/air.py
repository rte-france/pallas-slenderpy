"""Air properties and the drag coefficient of a circular cylinder.

Formulas ported from :mod:`slenderpy.wind`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slenderpy.components import _check_positive


@dataclass(frozen=True)
class Air:
    """Air state.

    Attributes:
        temperature: Temperature (K).
        pressure: Pressure (Pa).
        humidity: Relative humidity, in [0, 1].
    """

    temperature: float = 293.15
    pressure: float = 1.013e05
    humidity: float = 0.0

    def __post_init__(self) -> None:
        _check_positive("temperature", self.temperature)
        _check_positive("pressure", self.pressure)
        # written so that nan fails too
        if not 0.0 <= self.humidity <= 1.0:
            raise ValueError(f"humidity must be in [0, 1], got {self.humidity}")

    @property
    def density(self) -> float:
        """Volumic mass (kg/m^3)."""
        t, p, phi = self.temperature, self.pressure, self.humidity
        vapour = 230.617 * phi * np.exp(17.5043 * (t - 273.15) / (t - 31.95))
        return float((p - vapour) / (287.06 * t))

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity (m^2/s): dynamic viscosity over density."""
        t = self.temperature
        dynamic = 8.8848e-15 * t**3 - 3.2398e-11 * t**2 + 6.2657e-08 * t + 2.3543e-06
        return float(dynamic / self.density)


def cylinder_drag(reynolds):
    """Drag coefficient of a circular cylinder.

    From https://kdusling.github.io/teaching/Applied-Fluids/DragCoefficient.html,
    valid for ``reynolds < 2e5``.

    Parameters
    ----------
    reynolds : float or numpy.ndarray
        Reynolds number.

    Returns
    -------
    float or numpy.ndarray
        Drag coefficient.
    """
    reynolds = np.asarray(reynolds, dtype=float)
    return (
        11.0 * np.power(reynolds, -0.75)
        + 0.9 * (1.0 - np.exp(-1000.0 / reynolds))
        + 1.2 * (1.0 - np.exp(-np.power(reynolds / 4500.0, 0.7)))
    )
