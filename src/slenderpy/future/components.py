"""Frozen dataclasses describing a conductor and a span.

These value objects group the physical parameters shared by the cable and beam
models of :mod:`slenderpy.future`. Most parameters are common to both models; a
few are model-specific and default to ``None``:

    - ``axial_stiffness`` is used by the cable model only.
    - ``ei_min``, ``ei_max`` and ``beta_flexion`` describe the beam bending behavior;
      the choice between a constant-``EI`` and a Bouc-Wen sub-model is made by
      the solvers, not by the data.
    - ``boundary_conditions`` is used by the beam model only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from slenderpy.future.boundary_condition import BoundaryCondition


def _check_positive(name: str, value: float) -> None:
    """Raise ValueError if value is not a finite value strictly greater than 0."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite value > 0, got {value}")


def _check_optional_positive(name: str, value: float | None) -> None:
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
        beta_flexion: Flexion compliance (J^-1). Beam model (Bouc-Wen); the
            critical curvature is derived as chi0 = beta_flexion * tension.
        thermal_expansion: Linear thermal-expansion coefficient alpha (1/K).
            Cable thermal computations only.
    """

    mass: float
    diameter: float | None = None
    axial_stiffness: float | None = None
    ei_min: float | None = None
    ei_max: float | None = None
    beta_flexion: float | None = None
    thermal_expansion: float | None = None

    def __post_init__(self) -> None:
        _check_positive("mass", self.mass)
        _check_optional_positive("diameter", self.diameter)
        _check_optional_positive("axial_stiffness", self.axial_stiffness)
        _check_optional_positive("ei_min", self.ei_min)
        _check_optional_positive("ei_max", self.ei_max)
        _check_optional_positive("beta_flexion", self.beta_flexion)
        if (
            self.ei_min is not None
            and self.ei_max is not None
            and self.ei_max < self.ei_min
        ):
            raise ValueError(
                f"ei_max ({self.ei_max}) must be >= ei_min ({self.ei_min})"
            )


@dataclass(frozen=True)
class Span:
    """Geometry and loading of a span.

    Attributes:
        length: Span length (m).
        tension: Mechanical tension (N).
        sld: Support level difference between the two supports (m). May be
            negative; the sign encodes which support is higher.
        boundary_conditions: Boundary conditions for the beam model. ``None``
            for the cable model.
    """

    length: float
    tension: float
    sld: float = 0.0
    boundary_conditions: BoundaryCondition | None = None

    def __post_init__(self) -> None:
        _check_positive("length", self.length)
        _check_positive("tension", self.tension)
        if not math.isfinite(self.sld):
            raise ValueError(f"sld must be finite, got {self.sld}")
