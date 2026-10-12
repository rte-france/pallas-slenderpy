"""Force models shared by the beam and cable solvers.

A force is any callable ``force(x, t, y, z, vy, vz) -> (fy, fz)``:

- ``x``: horizontal distance of each node from support 1 (m);
- ``t``: time (s);
- ``y``, ``z``: global position of each node (m), ``vy``, ``vz`` its global
  velocity (m/s);
- ``fy``, ``fz``: force per unit length (N/m) along global ``y`` and ``z``,
  as arrays of the nodes' shape or anything broadcastable to it.

The frame is the one of [](`slenderpy.simulation`): origin at support
1, ``x`` along the span, ``z`` upwards, ``y`` completing a right-handed triad.
The beam solver uses ``fz`` only; the cable solver projects both components on
its local normal and binormal.

The classes below are provided models; they can be summed with ``+``, and
summed with or multiplied by a float (applied to both components) or a tuple
``(cy, cz)`` (applied per component). A plain function with the same
signature is a valid force too, but cannot take part in ``+`` or ``*``.
"""

from __future__ import annotations

import math
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from slenderpy._constant import _GRAVITY
from slenderpy.components import _check_positive


def _check_finite(name: str, value: float) -> None:
    """Raise ValueError if value is not finite."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _as_pair(value) -> tuple[float, float] | None:
    """``(y, z)`` pair from a real or a 2-tuple of reals, None otherwise."""
    if isinstance(value, numbers.Real):
        return float(value), float(value)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(v, numbers.Real) for v in value)
    ):
        return float(value[0]), float(value[1])
    return None


class Force(ABC):
    """Base of the provided force models."""

    @abstractmethod
    def __call__(self, x, t, y, z, vy, vz):
        """Force per unit length ``(fy, fz)`` (N/m) at the nodes."""

    def __add__(self, other):
        pair = _as_pair(other)
        if pair is not None:
            other = Constant(*pair)
        if not isinstance(other, Force):
            return NotImplemented
        left = self.terms if isinstance(self, ForceSum) else (self,)
        right = other.terms if isinstance(other, ForceSum) else (other,)
        return ForceSum(left + right)

    __radd__ = __add__

    def __mul__(self, other):
        pair = _as_pair(other)
        if pair is None:
            return NotImplemented
        return Scaled(self, *pair)

    __rmul__ = __mul__


@dataclass(frozen=True)
class ForceSum(Force):
    """Sum of forces, built with ``+``.

    Attributes
    ----------
    terms
        The summed forces, flattened: ``a + b + c`` holds three terms.
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
class Constant(Force):
    """Uniform force ``(fy, fz)``, built by adding a force and a constant.

    ``force + c`` adds ``c`` to both components, ``force + (cy, cz)`` adds
    ``cy`` to ``fy`` and ``cz`` to ``fz``.

    Attributes
    ----------
    fy
        Force per unit length along ``y`` (N/m).
    fz
        Force per unit length along ``z`` (N/m).
    """

    fy: float
    fz: float

    def __post_init__(self) -> None:
        _check_finite("fy", self.fy)
        _check_finite("fz", self.fz)

    def __call__(self, x, t, y, z, vy, vz):
        x = np.asarray(x, dtype=float)
        return np.full_like(x, self.fy), np.full_like(x, self.fz)


@dataclass(frozen=True)
class Scaled(Force):
    """Force multiplied by constant factors, built with ``*``.

    ``c * force`` multiplies both components by ``c``, ``(cy, cz) * force``
    multiplies ``fy`` by ``cy`` and ``fz`` by ``cz``.

    Attributes
    ----------
    force
        The scaled force.
    factor_y
        Factor on ``fy``.
    factor_z
        Factor on ``fz``.
    """

    force: Force
    factor_y: float
    factor_z: float

    def __post_init__(self) -> None:
        _check_finite("factor_y", self.factor_y)
        _check_finite("factor_z", self.factor_z)

    def __call__(self, x, t, y, z, vy, vz):
        fy, fz = self.force(x, t, y, z, vy, vz)
        return self.factor_y * fy, self.factor_z * fz


@dataclass(frozen=True)
class Gravity(Force):
    """Weight of the structure, ``(0, -g * mass)``.

    For the beam only: the cable equilibrium already holds the weight, so the
    cable solver refuses this force.

    Attributes
    ----------
    mass
        Mass per unit length (kg/m).
    """

    mass: float

    def __post_init__(self) -> None:
        _check_positive("mass", self.mass)

    def __call__(self, x, t, y, z, vy, vz):
        x = np.asarray(x, dtype=float)
        return np.zeros_like(x), np.full_like(x, -_GRAVITY * self.mass)


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

    Attributes
    ----------
    frequency
        Frequency (Hz).
    amplitude
        Amplitude (N); positive goes up first.
    position
        Horizontal distance from support 1 (m), at most the span length;
        checked when the force is evaluated, since the span is not known here.
    t_start
        Start of the excitation (s).
    t_end
        End of the excitation (s).
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
        # relative margin for the round-off of the last node position
        if self.position > x[-1] * (1.0 + 1.0e-12):
            raise ValueError(
                f"position ({self.position}) is past the span end ({x[-1]})"
            )
        fz = np.zeros_like(x)
        if self.t_start <= t <= self.t_end:
            node = int(np.argmin(np.abs(x - self.position)))
            node = min(max(node, 1), x.size - 2)
            width = 0.5 * (x[node + 1] - x[node - 1])
            phase = 2.0 * np.pi * self.frequency * (t - self.t_start)
            fz[node] = self.amplitude * np.sin(phase) / width
        return 0.0, fz
