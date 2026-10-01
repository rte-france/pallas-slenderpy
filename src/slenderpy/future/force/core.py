"""Force models shared by the future beam and cable solvers.

A force is any callable ``force(x, t, y, z, vy, vz) -> (fy, fz)``:

- ``x``: horizontal distance of each node from support 1 (m);
- ``t``: time (s);
- ``y``, ``z``: global position of each node (m), ``vy``, ``vz`` its global
  velocity (m/s);
- ``fy``, ``fz``: force per unit length (N/m) along global ``y`` and ``z``,
  as arrays of the nodes' shape or anything broadcastable to it.

The frame is the one of :mod:`slenderpy.future.simulation`: origin at support
1, ``x`` along the span, ``z`` upwards, ``y`` completing a right-handed triad.
The beam solver uses ``fz`` only; the cable solver projects both components on
its local normal and binormal.

The classes below are provided models; they can be summed with ``+``. A plain
function with the same signature is a valid force too, but cannot take part
in ``+``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from slenderpy.future._constant import _GRAVITY
from slenderpy.future.components import _check_positive


def _check_finite(name: str, value: float) -> None:
    """Raise ValueError if value is not finite."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


class Force(ABC):
    """Base of the provided force models."""

    @abstractmethod
    def __call__(self, x, t, y, z, vy, vz):
        """Force per unit length ``(fy, fz)`` (N/m) at the nodes."""

    def __add__(self, other):
        if not isinstance(other, Force):
            return NotImplemented
        left = self.terms if isinstance(self, ForceSum) else (self,)
        right = other.terms if isinstance(other, ForceSum) else (other,)
        return ForceSum(left + right)


@dataclass(frozen=True)
class ForceSum(Force):
    """Sum of forces, built with ``+``.

    Attributes:
        terms: The summed forces, flattened: ``a + b + c`` holds three terms.
    """

    terms: tuple[Force, ...]

    def __call__(self, x, t, y, z, vy, vz):
        fy, fz = 0.0, 0.0
        for term in self.terms:
            term_y, term_z = term(x, t, y, z, vy, vz)
            fy = fy + term_y
            fz = fz + term_z
        return fy, fz


@dataclass(frozen=True)
class Gravity(Force):
    """Weight of the structure, ``(0, -g * mass)``.

    For the beam only: the cable equilibrium already holds the weight, so the
    cable solver refuses this force.

    Attributes:
        mass: Mass per unit length (kg/m).
    """

    mass: float

    def __post_init__(self) -> None:
        _check_positive("mass", self.mass)

    def __call__(self, x, t, y, z, vy, vz):
        return 0.0, -_GRAVITY * self.mass


@dataclass(frozen=True)
class PointExcitation(Force):
    """Sinusoidal vertical force applied at one node.

    The force ``amplitude * sin(2 pi frequency (t - t_start))`` (N) is applied,
    while ``t_start <= t <= t_end``, at the node nearest ``position``, never on
    a support node. It is spread over the half-cell
    ``w = (x[i+1] - x[i-1]) / 2`` of that node, so ``fz[i] * w`` is the
    applied force.

    ``w`` is measured in horizontal ``x``. The cable model integrates per unit
    arc length, longer by the local factor ``sqrt(1 + slope**2)``, so on a
    cable the applied force is larger by that factor (about 1% at typical
    slopes).

    Attributes:
        frequency: Frequency (Hz).
        amplitude: Amplitude (N); positive goes up first.
        position: Horizontal distance from support 1 (m).
        t_start: Start of the excitation (s).
        t_end: End of the excitation (s).
    """

    frequency: float
    amplitude: float
    position: float
    t_start: float = 0.0
    t_end: float = math.inf

    def __post_init__(self) -> None:
        _check_positive("frequency", self.frequency)
        _check_finite("amplitude", self.amplitude)
        _check_finite("position", self.position)
        if self.position < 0.0:
            raise ValueError(f"position must be >= 0, got {self.position}")
        _check_finite("t_start", self.t_start)
        # written so that nan fails too
        if not self.t_end > self.t_start:
            raise ValueError(
                f"t_end ({self.t_end}) must be larger than t_start ({self.t_start})"
            )

    def __call__(self, x, t, y, z, vy, vz):
        x = np.asarray(x, dtype=float)
        fz = np.zeros_like(x)
        if self.t_start <= t <= self.t_end:
            node = int(np.argmin(np.abs(x - self.position)))
            node = min(max(node, 1), x.size - 2)
            width = 0.5 * (x[node + 1] - x[node - 1])
            phase = 2.0 * np.pi * self.frequency * (t - self.t_start)
            fz[node] = self.amplitude * np.sin(phase) / width
        return 0.0, fz
