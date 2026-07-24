"""Frozen dataclasses describing a conductor and a span.

These value objects group the physical parameters shared by the cable and beam
models of :mod:`slenderpy.future`. Most parameters are common to both models; a
few are model-specific and default to ``None``:

    - ``axial_stiffness`` is used by the cable model only.
    - ``ei_min``, ``ei_max`` and ``chi0`` describe the beam bending behaviour;
      the choice between a constant-``EI`` and a Bouc-Wen sub-model is made by
      the solvers, not by the data.
    - ``boundary_conditions`` is used by the beam model only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _check_positive(name: str, value: float) -> None:
    """Raise ValueError if value is not strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _check_optional_positive(name: str, value: Optional[float]) -> None:
    """Raise ValueError if value is provided and not strictly positive."""
    if value is not None:
        _check_positive(name, value)


@dataclass(frozen=True)
class Conductor:
    """Physical properties of a conductor, expressed per unit length.

    Attributes:
        mass: Linear mass, i.e. mass per unit length (kg/m).
        diameter: Conductor diameter (m). Used by dynamic-force computations.
        axial_stiffness: Axial stiffness EA (N). Cable model only.
        ei_min: Minimum bending stiffness (N.m^2). Beam model.
        ei_max: Maximum bending stiffness (N.m^2). Beam model.
        chi0: Critical curvature (1/m). Beam model (Bouc-Wen).
        thermal_expansion: Linear thermal-expansion coefficient alpha (1/K).
            Cable thermal computations only.
    """

    mass: float
    diameter: Optional[float] = None
    axial_stiffness: Optional[float] = None
    ei_min: Optional[float] = None
    ei_max: Optional[float] = None
    chi0: Optional[float] = None
    thermal_expansion: Optional[float] = None

    def __post_init__(self) -> None:
        _check_positive("mass", self.mass)
        _check_optional_positive("diameter", self.diameter)
        _check_optional_positive("axial_stiffness", self.axial_stiffness)
        _check_optional_positive("ei_min", self.ei_min)
        _check_optional_positive("ei_max", self.ei_max)
        _check_optional_positive("chi0", self.chi0)
        if (
            self.ei_min is not None
            and self.ei_max is not None
            and self.ei_max < self.ei_min
        ):
            raise ValueError(
                f"ei_max ({self.ei_max}) must be >= ei_min ({self.ei_min})"
            )
