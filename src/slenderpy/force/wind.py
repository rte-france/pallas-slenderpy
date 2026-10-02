"""Wind models and the wind drag force.

A wind model provides ``velocity(x, t) -> (wy, wz)``, the wind velocity (m/s)
at the horizontal positions ``x`` (m) and time ``t`` (s), in the global frame
of :mod:`slenderpy.force.core`. The mean wind blows along ``+y``.

The turbulent models are ported from :mod:`slenderpy.turbwind`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from slenderpy.components import _check_positive
from slenderpy.force.air import Air, cylinder_drag
from slenderpy.force.core import Force


def _von_karman_u(frequency, mean, std, length_scale):
    """Von Karman spectrum of the along-wind component."""
    n = (length_scale * frequency / mean) ** 2
    return 4.0 * length_scale * std**2 / mean * np.power(1.0 + 70.7 * n, -5.0 / 6.0)


def _von_karman_w(frequency, mean, std, length_scale):
    """Von Karman spectrum of the vertical component."""
    n = (length_scale * frequency / mean) ** 2
    return (
        4.0
        * length_scale
        * std**2
        / mean
        * np.power(1.0 + 282.8 * n, -11.0 / 6.0)
        * (1.0 + 753.6 * n)
    )


def _check_window(t_start: float, t_end: float, dt: float) -> None:
    """Raise ValueError on a non-finite or empty time window."""
    if not math.isfinite(t_start):
        raise ValueError(f"t_start must be finite, got {t_start}")
    _check_positive("dt", dt)
    if not (math.isfinite(t_end) and t_end - t_start >= 2.0 * dt):
        raise ValueError(
            f"t_end ({t_end}) must be finite and at least 2 dt after t_start"
        )


def _time_vector(t_start: float, t_end: float, dt: float) -> np.ndarray:
    """Sample times covering the window with a step of about dt."""
    return np.linspace(t_start, t_end, 1 + int(np.floor((t_end - t_start) / dt)))


@dataclass(frozen=True)
class ConstantWind:
    """Steady, uniform wind along ``y``.

    Attributes:
        speed: Wind speed (m/s); a negative speed blows along ``-y``.
    """

    speed: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.speed):
            raise ValueError(f"speed must be finite, got {self.speed}")

    def velocity(self, x, t):
        """Wind velocity ``(wy, wz)`` (m/s)."""
        return self.speed, 0.0


@dataclass(frozen=True)
class UniformTurbulentWind:
    """Turbulent wind along ``y``, the same at every point of the span.

    A Von Karman along-wind signal generated at init on the time window, then
    normalised to exactly ``mean`` and ``std``. Ported from
    :class:`slenderpy.turbwind.RandomWind1D`; with ``t_start = 0`` both give
    the same signal for the same seed. Between samples the speed is linearly
    interpolated; outside the window it is held at its end values.

    Attributes:
        mean: Mean wind speed (m/s), positive.
        std: Standard deviation of the wind speed (m/s).
        length_scale: Turbulence length scale (m).
        t_start: Start of the window (s).
        t_end: End of the window (s).
        dt: Sampling step (s).
        seed: Seed of the random generator; None for a random seed.
    """

    mean: float
    std: float
    length_scale: float
    t_start: float
    t_end: float
    dt: float
    seed: int | None = None

    def __post_init__(self) -> None:
        _check_positive("mean", self.mean)
        _check_positive("std", self.std)
        _check_positive("length_scale", self.length_scale)
        _check_window(self.t_start, self.t_end, self.dt)

        times = _time_vector(self.t_start, self.t_end, self.dt)
        rng = np.random.default_rng(self.seed)
        n = times.size
        half = n // 2
        if half < 3:
            raise ValueError(
                "the window must hold at least one harmonic (6 samples): "
                "increase t_end - t_start or decrease dt"
            )
        frequency = np.fft.fftfreq(n, d=np.mean(np.diff(times)))[:half]
        phase = rng.uniform(0.0, 2.0 * np.pi, half - 1)
        spectrum = _von_karman_u(frequency, self.mean, self.std, self.length_scale)
        amplitude = np.sqrt(np.diff(frequency) * (spectrum[1:] + spectrum[:-1]))
        speed = self.mean * np.ones_like(times)
        for i in range(1, half - 1):
            speed += amplitude[i] * np.cos(
                2.0 * np.pi * times * frequency[i] + phase[i]
            )
        speed = self.std * (speed - np.mean(speed)) / np.std(speed) + self.mean

        object.__setattr__(self, "times", times)
        object.__setattr__(self, "speed", speed)

    def velocity(self, x, t):
        """Wind velocity ``(wy, wz)`` (m/s)."""
        return float(np.interp(t, self.times, self.speed)), 0.0


@dataclass(frozen=True)
class TurbulentWindField:
    """Turbulent wind field along the span, along-wind and vertical components.

    Ported from :class:`slenderpy.turbwind.TurbWind3D`, restricted to the
    along-wind ``u`` (giving ``wy``) and vertical ``w`` (giving ``wz``)
    components; the along-span component is tangential and, the legacy cross
    spectral matrix being block diagonal, dropping it leaves ``u`` and ``w``
    with the same statistics. The legacy generation is kept, including the
    absolute value of the Cholesky factor and the global normalisation of each
    component to its target standard deviation; the constant
    ``sqrt(n_points)`` factor of the legacy spectral matrix is dropped since
    that normalisation cancels it. The random draws differ from legacy.

    The field is sampled on ``grid = linspace(0, length, n_points)`` and on
    the time window; ``velocity`` interpolates linearly in time then in
    ``x``, and holds the edge values outside.

    Attributes:
        mean: Mean wind speed (m/s), positive.
        length: Length of the sampled line (m).
        n_points: Number of points of the line, at least 2.
        t_start: Start of the window (s).
        t_end: End of the window (s).
        dt: Sampling step (s).
        std_u: Standard deviation of the along-wind component (m/s).
        std_w: Standard deviation of the vertical component (m/s).
        length_scale_u: Turbulence length scale of ``u`` (m).
        length_scale_w: Turbulence length scale of ``w`` (m).
        coherence_u: Coherence decay coefficient of ``u`` along the line.
        coherence_w: Coherence decay coefficient of ``w`` along the line.
        seed: Seed of the random generator; None for a random seed.
    """

    mean: float
    length: float
    n_points: int
    t_start: float
    t_end: float
    dt: float
    std_u: float = 0.3
    std_w: float = 0.1
    length_scale_u: float = 200.0
    length_scale_w: float = 30.0
    coherence_u: float = 7.0
    coherence_w: float = 7.0
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "mean",
            "length",
            "std_u",
            "std_w",
            "length_scale_u",
            "length_scale_w",
            "coherence_u",
            "coherence_w",
        ):
            _check_positive(name, getattr(self, name))
        if self.n_points < 2:
            raise ValueError(f"n_points must be at least 2, got {self.n_points}")
        _check_window(self.t_start, self.t_end, self.dt)

        duration = self.t_end - self.t_start
        frequencies = np.arange(1.0 / duration, 0.5 / self.dt, 1.0 / duration)
        if frequencies.size < 2:
            raise ValueError(
                "the window must hold at least two frequencies: increase "
                "t_end - t_start or decrease dt"
            )

        times = _time_vector(self.t_start, self.t_end, self.dt)
        grid = np.linspace(0.0, self.length, self.n_points)
        gap = np.abs(grid[:, None] - grid[None, :])
        weight = np.sqrt(2.0 * np.median(np.diff(frequencies)))
        rng = np.random.default_rng(self.seed)

        n = self.n_points
        u = np.zeros((n, times.size))
        w = np.zeros((n, times.size))
        components = (
            (u, _von_karman_u, self.std_u, self.length_scale_u, self.coherence_u),
            (w, _von_karman_w, self.std_w, self.length_scale_w, self.coherence_w),
        )
        for frequency in frequencies:
            phases = 2.0 * np.pi * rng.random(2 * n)
            for k, (values, spectrum, std, scale, coherence) in enumerate(components):
                cross = np.exp(-frequency * coherence * gap / self.mean) * spectrum(
                    frequency, self.mean, std, scale
                )
                factor = np.abs(np.linalg.cholesky(cross))
                phase = phases[k * n : (k + 1) * n]
                waves = np.cos(
                    2.0 * np.pi * frequency * times[None, :] + phase[:, None]
                )
                values += weight * factor @ waves

        u = self.std_u * (u - np.mean(u)) / np.std(u) + self.mean
        w = self.std_w * (w - np.mean(w)) / np.std(w)

        object.__setattr__(self, "times", times)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "w", w)

    def _at_time(self, values, t):
        """Values on the grid at time t, linear in time, held outside."""
        last = self.times.size - 1
        step = self.times[1] - self.times[0]
        q = min(max((t - self.times[0]) / step, 0.0), last)
        i = min(int(q), last - 1)
        a = q - i
        return (1.0 - a) * values[:, i] + a * values[:, i + 1]

    def velocity(self, x, t):
        """Wind velocity ``(wy, wz)`` (m/s)."""
        x = np.asarray(x, dtype=float)
        wy = np.interp(x, self.grid, self._at_time(self.u, t))
        wz = np.interp(x, self.grid, self._at_time(self.w, t))
        return wy, wz


@dataclass(frozen=True)
class WindDrag(Force):
    """Drag of the wind on a circular cylinder, with the relative velocity.

    With ``r = (wy - vy, wz - vz)`` the relative wind and ``|r|`` its norm,
    the force per unit length is ``0.5 * rho * diameter * cd * |r| * r``.
    ``cd`` is ``drag_coefficient`` if it is a float, else
    ``drag_coefficient(Re)`` at the local Reynolds number
    ``Re = |r| * diameter / nu``. The force is zero where ``|r|`` is zero.

    Attributes:
        diameter: Cylinder diameter (m).
        wind: Any object with ``velocity(x, t) -> (wy, wz)``.
        drag_coefficient: Constant drag coefficient, or a function of the
            Reynolds number. Default :func:`cylinder_drag`.
        air: Air state, for the density and the viscosity.
    """

    diameter: float
    wind: object
    drag_coefficient: float | Callable = cylinder_drag
    air: Air = field(default_factory=Air)

    def __post_init__(self) -> None:
        _check_positive("diameter", self.diameter)
        if not callable(self.drag_coefficient):
            _check_positive("drag_coefficient", self.drag_coefficient)

    def __call__(self, x, t, y, z, vy, vz):
        wy, wz = self.wind.velocity(x, t)
        ry = wy - np.asarray(vy, dtype=float)
        rz = wz - np.asarray(vz, dtype=float)
        speed = np.hypot(ry, rz)
        if callable(self.drag_coefficient):
            reynolds = speed * self.diameter / self.air.kinematic_viscosity
            # the default law diverges at Re = 0, where the force is zero anyway
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                cd = self.drag_coefficient(reynolds)
            cd = np.where(speed > 0.0, cd, 0.0)
        else:
            cd = self.drag_coefficient
        gamma = 0.5 * self.air.density * self.diameter * cd * speed
        return gamma * ry, gamma * rz
