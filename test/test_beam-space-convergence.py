"""Space convergence of the static and dynamic beam solvers across spans.

The same three spans as the time convergence tests, an ASTER 570 with clamped
ends: a long span (440 m, 39 kN), a medium span (50 m, 5 kN) and a jumper
(5 m, 50 N). The static shape under gravity is checked with both bending
models; the dynamic solve, with the varying model, in free vibration, under a
slow point force at the first frequency and under a fast one at 20 Hz.

Each case is run on a coarse grid refined 2x and 4x, against a reference
refined 8x, and compared at the span fractions ``k / 8``, nodes of every grid.
The dynamic runs keep the same time step on every grid, so the gap between two
grids is the space error of the time-discrete solution.

The position and the velocity converge at first order: the clamped ends
impose the slope with the one-sided ``(y1 - y0) / ds`` (see
``BoundaryCondition.compute``), and its error spreads to the whole span.
Hinged ends converge at second order. The hysteresis variable ``eta``
converges more irregularly, see ``_MIN_TOTAL_RATIO_ETA``.

The coarse grids resolve the bending boundary layer ``sqrt(EI / T)`` of the
medium span (0.66 m, 4 cells) and of the jumper, and the 8 m wavelength of the
fast loading on the long span (8 cells). The 0.23 m boundary layer of the long
span is not resolved. On the medium span, ``eta`` does not decrease at each
refinement from coarse grids of 80 or 160 cells.
"""

import numpy as np
import pytest

from slenderpy import simulation
from slenderpy.beam.bending import BendingModel
from slenderpy.beam.dynamic import solve_dynamic
from slenderpy.beam.static import frequency, shape
from slenderpy.boundary_condition import clamped
from slenderpy.components import Conductor, Span
from slenderpy.force.core import Gravity, PointExcitation

CONDUCTOR = Conductor(
    mass=1.57,
    diameter=31.1e-3,
    ei_min=28.28,
    ei_max=2155.07,
    beta_flexion=6.437693e-07,
)

# span and number of cells of the coarse grid, a multiple of 8
SPANS = {
    "long": (Span(length=440.0, tension=39e3, boundary_conditions=clamped()), 440),
    "medium": (Span(length=50.0, tension=5e3, boundary_conditions=clamped()), 320),
    "jumper": (Span(length=5.0, tension=50.0, boundary_conditions=clamped()), 80),
}

# frequency of the fast point force (Hz)
FAST = 20.0

# free: peak displacement (m) of a sin**2 initial velocity at the first
# frequency; slow and fast: point force amplitude (N) at a quarter span. Same
# as the time convergence tests
AMPLITUDE = {
    ("long", "free"): 0.5,
    ("long", "slow"): 200.0,
    ("long", "fast"): 50.0,
    ("medium", "free"): 5.0e-3,
    ("medium", "slow"): 2.0,
    ("medium", "fast"): 20.0,
    ("jumper", "free"): 1.0e-3,
    ("jumper", "slow"): 2.0,
    ("jumper", "fast"): 2.0,
}

# time steps per period of the loading, the same on every grid, and periods run
STEPS = 80
PERIODS = {"free": 2, "slow": 3, "fast": 10}

# coarse grid refined 1x, 2x and 4x, against a reference refined 8x
REFINEMENTS = (1, 2, 4)
REFERENCE = 8

# over the 4x refinement the error has to drop at least 5x: first order gives
# 7 against the reference at 8x, second order 21. The position and the
# velocity reach 6.9 or more on every case. The hysteresis variable only has
# to drop 4x, about order 1/2 as in the time convergence tests: the bouc-wen
# switching is not smooth. The tightest case, long-fast, reaches 4.9
_MIN_TOTAL_RATIO = 5.0
_MIN_TOTAL_RATIO_ETA = 4.0

# newton tolerance of the dynamic runs, as in the time convergence tests
_TOL = 1.0e-10


def _first_frequency(span):
    """First frequency of the clamped span with the stiff bending (Hz)."""
    return float(
        frequency.natural_frequencies_clamped(
            span.length, span.tension, CONDUCTOR.mass, CONDUCTOR.ei_max, 1
        )[0]
    )


def _static_shape(span, ns, model):
    """Static shape of the span under gravity on ``ns`` nodes."""
    x = np.linspace(0.0, span.length, ns)
    weight = Gravity(CONDUCTOR.mass)(x, 0.0, 0.0, 0.0, 0.0, 0.0)[1]
    return shape.solve(CONDUCTOR, span, np.zeros(ns) + weight, ns, model=model)


def _run(span_name, loading, refinement):
    """Varying-model run of a (span, loading) case on the grid refined ``refinement``x.

    The output is recorded at the span fractions ``k / 8``, so the runs compare
    point by point.
    """
    span, cells = SPANS[span_name]
    ns = cells * refinement + 1
    x = np.linspace(0.0, span.length, ns)
    amplitude = AMPLITUDE[(span_name, loading)]
    f = FAST if loading == "fast" else _first_frequency(span)
    gravity = Gravity(CONDUCTOR.mass)

    y0 = _static_shape(span, ns, BendingModel.VARYING)
    if loading == "free":
        v0 = 2.0 * np.pi * f * amplitude * np.sin(np.pi * x / span.length) ** 2
        force = gravity
    else:
        # a quarter span is a node of every grid
        v0 = np.zeros(ns)
        force = gravity + PointExcitation(
            frequency=f, amplitude=amplitude, position=0.25 * span.length
        )

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
    return solve_dynamic(
        CONDUCTOR,
        span,
        parameters,
        model=BendingModel.VARYING,
        initial_position=y0,
        initial_velocity=v0,
        force=force,
        tol=_TOL,
    )


def _l2_gap(first, second):
    """Root-mean-square gap between two recorded fields."""
    return float(np.sqrt(np.mean((np.asarray(first) - np.asarray(second)) ** 2)))


def _check_errors(errors, label, min_total_ratio=_MIN_TOTAL_RATIO):
    """The error decreases at each refinement and drops enough overall."""
    assert errors[0] > errors[1] > errors[2], (label, errors)
    assert errors[0] / errors[2] > min_total_ratio, (label, errors)


@pytest.mark.parametrize("span_name", list(SPANS))
@pytest.mark.parametrize("model", [BendingModel.CONSTANT, BendingModel.VARYING])
def test_static_shape_converges_in_space(span_name, model):
    """Refining the grid converges on the static shape under gravity."""
    span, cells = SPANS[span_name]

    def at_eighths(refinement):
        y = _static_shape(span, cells * refinement + 1, model)
        return y[:: cells * refinement // 8]

    reference = at_eighths(REFERENCE)
    errors = [_l2_gap(at_eighths(r), reference) for r in REFINEMENTS]
    _check_errors(errors, "z")


_CASES = [
    pytest.param(span, loading, id=f"{span}-{loading}")
    for span in SPANS
    for loading in ("free", "slow", "fast")
]


@pytest.mark.parametrize("span_name, loading", _CASES)
def test_dynamic_converges_in_space(span_name, loading):
    """Refining the grid converges on every span and loading.

    Checked on the position, the velocity and the hysteresis variable.
    """
    reference = _run(span_name, loading, REFERENCE)
    runs = {r: _run(span_name, loading, r) for r in REFINEMENTS}

    for field, min_total_ratio in (
        ("z", _MIN_TOTAL_RATIO),
        ("vz", _MIN_TOTAL_RATIO),
        ("eta", _MIN_TOTAL_RATIO_ETA),
    ):
        errors = [_l2_gap(runs[r][field], reference[field]) for r in REFINEMENTS]
        _check_errors(errors, field, min_total_ratio)
