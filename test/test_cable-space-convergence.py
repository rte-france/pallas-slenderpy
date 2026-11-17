"""Space convergence of the static and dynamic cable solvers across spans.

The same three spans as the time convergence tests (an ASTER 570 conductor):
a long span (440 m, 39 kN), a medium span (50 m, 5 kN) and a jumper (5 m,
50 N). The static shape is checked under a uniform horizontal load on top of
the weight; the dynamic solve, with the implicit strain, in free vibration,
under a slow load at the first taut-string frequency and under a fast one at
20 Hz.

Each case is run on a coarse grid refined 2x and 4x, against a reference
refined 8x, and compared at the span fractions ``k / 8``. The dynamic runs
keep the same time step on every grid, so the gap between two grids is the
space error of the time-discrete solution.

The loads are uniform, in plane on the long and medium spans and out of plane
on the jumper (see the time convergence tests). A point force would not do: it
is applied at the node nearest its horizontal position, and the cable nodes are
evenly spaced in arc length, so the load moves by up to half a cell from one
grid to the next.

The dynamic tension converges at first order, the end rows of the first
derivative being one-sided (see ``slenderpy.cable._model._operators``). The
position converges at second order on the static shape, on the jumper and on
the medium span in free vibration, and at first order where the dynamic
tension drives it, on the long span. The
coarse grids hold at least 8 cells per wavelength of the fast loading.
"""

import numpy as np
import pytest

from slenderpy import simulation
from slenderpy.cable import dynamic
from slenderpy.cable.static import shape
from slenderpy.components import Conductor, Span

CONDUCTOR = Conductor(mass=1.571, diameter=0.0313, axial_stiffness=3.76e07)

# span and number of cells of the coarse grid, a multiple of 8
SPANS = {
    "long": (Span(length=440.0, tension=39e3), 440),
    "medium": (Span(length=50.0, tension=5e3), 160),
    "jumper": (Span(length=5.0, tension=50.0), 160),
}

# spans loaded out of plane, see the module docstring
OUT_OF_PLANE = {"jumper"}

# frequency of the fast load (Hz)
FAST = 20.0

# horizontal load of the static shape (N/m), half the weight
STATIC_LOAD = 0.5 * 9.81 * CONDUCTOR.mass

# free: peak displacement (m) of a sin initial velocity at the first frequency;
# slow and fast: uniform load amplitude (N/m)
AMPLITUDE = {
    ("long", "free"): 0.5,
    ("long", "slow"): 0.5,
    ("long", "fast"): 0.5,
    ("medium", "free"): 0.05,
    ("medium", "slow"): 0.5,
    ("medium", "fast"): 0.5,
    ("jumper", "free"): 4.0e-3,
    ("jumper", "slow"): 0.01,
    ("jumper", "fast"): 2.0,
}

# time steps per period of the loading, the same on every grid, and periods run
STEPS = 80
PERIODS = {"free": 2, "slow": 3, "fast": 10}

# coarse grid refined 1x, 2x and 4x, against a reference refined 8x
REFINEMENTS = (1, 2, 4)
REFERENCE = 8

# over the 4x refinement the error has to drop at least 5x: first order gives
# 7 against the reference at 8x, second order 21. Every case reaches 7.0 or
# more, the jumper position about 20
_MIN_TOTAL_RATIO = 5.0


def _first_frequency(span):
    """First taut-string frequency of the span (Hz)."""
    return 0.5 / span.length * np.sqrt(span.tension / CONDUCTOR.mass)


def _uniform_load(amplitude, frequency, out_of_plane):
    """Sinusoidal load (N/m), uniform along the span."""

    def force(x, t, y, z, vy, vz):
        value = amplitude * np.sin(2.0 * np.pi * frequency * t)
        return (value, 0.0) if out_of_plane else (0.0, value)

    return force


def _run(span_name, loading, refinement):
    """Run of a (span, loading) case on the grid refined ``refinement``x.

    The output is recorded at the span fractions ``k / 8``, so the runs compare
    point by point.
    """
    span, cells = SPANS[span_name]
    ns = cells * refinement + 1
    amplitude = AMPLITUDE[(span_name, loading)]
    f = FAST if loading == "fast" else _first_frequency(span)
    out_of_plane = span_name in OUT_OF_PLANE
    kwargs = {}
    if loading == "free":
        geom = dynamic._geometry(CONDUCTOR, span, ns)
        direction = geom.eb if out_of_plane else geom.en
        kwargs["initial_velocity"] = (
            2.0 * np.pi * f * amplitude * np.sin(np.pi * geom.s) * direction
        )
    else:
        kwargs["force"] = _uniform_load(amplitude, f, out_of_plane)

    dt = 1.0 / (STEPS * f)
    parameters = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=PERIODS[loading] / f,
        dt=dt,
        dr=4.0 * dt,
        los=9,
        pp=False,
    )
    return dynamic.solve(CONDUCTOR, span, parameters, implicit_strain=True, **kwargs)


def _l2_gap(first, second):
    """Root-mean-square gap between two recorded fields."""
    return float(np.sqrt(np.mean((np.asarray(first) - np.asarray(second)) ** 2)))


def _check_errors(errors, label):
    """The error decreases at each refinement and drops enough overall."""
    assert errors[0] > errors[1] > errors[2], (label, errors)
    assert errors[0] / errors[2] > _MIN_TOTAL_RATIO, (label, errors)


@pytest.mark.parametrize("span_name", list(SPANS))
def test_static_shape_converges_in_space(span_name):
    """Refining the grid converges on the static shape under a lateral load."""
    span, cells = SPANS[span_name]

    def at_eighths(refinement):
        ns = cells * refinement + 1
        position = shape.solve(CONDUCTOR, span, STATIC_LOAD, 0.0, ns)
        return position[1:, :: cells * refinement // 8]

    reference = at_eighths(REFERENCE)
    errors = [_l2_gap(at_eighths(r), reference) for r in REFINEMENTS]
    _check_errors(errors, "yz")


@pytest.mark.parametrize(
    "span_name, loading",
    [
        pytest.param(span, loading, id=f"{span}-{loading}")
        for span in SPANS
        for loading in ("free", "slow", "fast")
    ],
)
def test_dynamic_converges_in_space(span_name, loading):
    """Refining the grid converges on every span and loading.

    Checked on the position along the loading (``y`` out of plane, ``z`` in
    plane) and on the dynamic tension. The cable has to stay taut.
    """
    span, _ = SPANS[span_name]
    reference = _run(span_name, loading, REFERENCE)
    runs = {r: _run(span_name, loading, r) for r in REFINEMENTS}

    assert np.min(span.tension + reference["dtension"].values) > 0.0

    position = "y" if span_name in OUT_OF_PLANE else "z"
    for field in (position, "dtension"):
        errors = [_l2_gap(runs[r][field], reference[field]) for r in REFINEMENTS]
        _check_errors(errors, field)
