"""Time convergence of the dynamic beam solver across spans and loadings.

The spans cover the three hysteresis regimes met in practice, all with an
ASTER 570 and clamped ends:

- a long span (440 m, 39 kN), nearly elastic: the dynamic curvature stays
  below the reference curvature ``chi0``;
- a medium span (50 m, 5 kN), around ``chi0``;
- a jumper (5 m, 50 N), whose tiny ``chi0`` keeps it saturated.

Each one is run in free vibration, under a slow point force at its first
frequency and under a fast one at 20 Hz (aeolian range).

These cases guard the time scheme against two failures met while choosing it:
a numerical damping proportional to ``dt`` (``eta`` driven half a step ahead
of the curvature), and the undamped ringing of the unresolved high-frequency
modes, which stops a Crank-Nicolson step from converging once the hysteresis
is active.
"""

import numpy as np
import pytest

from slenderpy import simulation
from slenderpy.beam.bending import BendingModel
from slenderpy.beam.dynamic import solve_dynamic
from slenderpy.beam.static import frequency, shape
from slenderpy.boundary_condition import clamped, hinged
from slenderpy.components import Conductor, Span
from slenderpy.force.core import Gravity, PointExcitation

CONDUCTOR = Conductor(
    mass=1.57,
    diameter=31.1e-3,
    ei_min=28.28,
    ei_max=2155.07,
    beta_flexion=6.437693e-07,
)

# span and number of nodes
SPANS = {
    "long": (Span(length=440.0, tension=39e3, boundary_conditions=clamped()), 441),
    "medium": (Span(length=50.0, tension=5e3, boundary_conditions=clamped()), 201),
    "jumper": (Span(length=5.0, tension=50.0, boundary_conditions=clamped()), 101),
}

# frequency of the fast point force (Hz)
FAST = 20.0

# free: peak displacement (m) of a sin**2 initial velocity at the first
# frequency; slow and fast: point force amplitude (N) at a quarter span. They
# give a peak dynamic curvature of about 0.3 chi0 (long, free) to 600 chi0
# (jumper, slow)
AMPLITUDE = {
    ("long", "free"): 0.5,
    ("long", "slow"): 200.0,
    ("long", "fast"): 50.0,
    ("medium", "free"): 5.0e-3,
    ("medium", "slow"): 2.0,
    ("medium", "fast"): 20.0,
    # 1 mm: from there a weakly damped scheme (hht alpha -0.05) turns chaotic
    ("jumper", "free"): 1.0e-3,
    ("jumper", "slow"): 2.0,
    ("jumper", "fast"): 2.0,
}

# time steps per period of the loading at the coarsest run, and periods run
STEPS = 20
PERIODS = {"free": 2, "slow": 3, "fast": 10}

# coarse runs at dt, dt/4 and dt/16, against a reference at dt/64
REFINEMENTS = (1, 4, 16)
REFERENCE = 64

# over the 16x refinement the error has to drop at least 4x, i.e. an observed
# order of 1/2 or better: the bouc-wen switching is not smooth, so the order of
# the hysteretic cases stays between 1/2 and 1 at these steps. The tightest
# case, medium-slow on eta, reaches 4.6
_MIN_TOTAL_RATIO = 4.0

# newton tolerance of the runs. At the default 1e-06 the dt/64 reference of
# the jumper moves by up to 1e-07 m with the tolerance, 15% of the dt/16 error
# it has to measure: the stopping point decides on which branch a node leaves
# a reversal. 1e-10 makes the reference independent of it
_TOL = 1.0e-10


def _first_frequency(span):
    """First frequency of the clamped span with the stiff bending (Hz)."""
    return float(
        frequency.natural_frequencies_clamped(
            span.length, span.tension, CONDUCTOR.mass, CONDUCTOR.ei_max, 1
        )[0]
    )


def _run(span_name, loading, refinement):
    """Varying-model run of a (span, loading) case at ``dt / refinement``.

    The output is recorded at the coarsest step whatever the refinement, so
    the runs compare snapshot by snapshot.
    """
    span, ns = SPANS[span_name]
    x = np.linspace(0.0, span.length, ns)
    amplitude = AMPLITUDE[(span_name, loading)]
    f = FAST if loading == "fast" else _first_frequency(span)
    gravity = Gravity(CONDUCTOR.mass)

    y0 = shape.solve(
        CONDUCTOR,
        span,
        np.zeros(ns) + gravity(x, 0.0, 0.0, 0.0, 0.0, 0.0)[1],
        ns,
        model=BendingModel.VARYING,
    )
    if loading == "free":
        # sin**2 is compatible with the clamped ends and excites few of the
        # unresolved high modes, unlike a released static shape
        v0 = 2.0 * np.pi * f * amplitude * np.sin(np.pi * x / span.length) ** 2
        force = gravity
    else:
        v0 = np.zeros(ns)
        force = gravity + PointExcitation(
            frequency=f, amplitude=amplitude, position=0.25 * span.length
        )

    coarse_dt = 1.0 / (STEPS * f)
    parameters = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=PERIODS[loading] / f,
        dt=coarse_dt / refinement,
        dr=coarse_dt,
        los=ns,
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


_CASES = [
    pytest.param(span, loading, id=f"{span}-{loading}")
    for span in SPANS
    for loading in ("free", "slow", "fast")
]


@pytest.mark.parametrize("span_name, loading", _CASES)
def test_converging_in_time_across_spans(span_name, loading):
    """Refining the time step converges on every span and loading.

    The error against the finest run decreases at each refinement and drops
    by at least ``_MIN_TOTAL_RATIO`` over the whole range, on the position,
    the velocity and the hysteresis variable.
    """
    reference = _run(span_name, loading, REFERENCE)
    runs = {r: _run(span_name, loading, r) for r in REFINEMENTS}

    for field in ("z", "vz", "eta"):
        errors = [_l2_gap(runs[r][field], reference[field]) for r in REFINEMENTS]
        assert errors[0] > errors[1] > errors[2], (field, errors)
        assert errors[0] / errors[2] > _MIN_TOTAL_RATIO, (field, errors)


def _decay_damping(ratio, steps):
    """Equivalent damping ratio of a free decay, mode 9 of a 30 m hinged span.

    The beam starts at rest in position with the velocity of mode 9, scaled so
    that its peak curvature is ``ratio * chi0``, and is left to decay for 50
    periods with ``steps`` time steps per period.
    """
    span = Span(length=30.0, tension=39.0e3, boundary_conditions=hinged())
    n, ns, n_periods = 9, 151, 50
    chi0 = CONDUCTOR.beta_flexion * span.tension
    x = np.linspace(0.0, span.length, ns)
    fn = frequency.natural_frequencies_hinged(
        span.length, span.tension, CONDUCTOR.mass, CONDUCTOR.ei_max, n
    )[-1]
    amplitude = ratio * chi0 / (n * np.pi / span.length) ** 2
    dt = 1.0 / (steps * fn)
    parameters = simulation.Parameters(
        ns=ns, tf=n_periods / fn, dt=dt, dr=dt, los=[0.5 / n], pp=False
    )
    res = solve_dynamic(
        CONDUCTOR,
        span,
        parameters,
        model=BendingModel.VARYING,
        initial_position=np.zeros(ns),
        initial_velocity=2.0
        * np.pi
        * fn
        * amplitude
        * np.sin(n * np.pi * x / span.length),
    )
    z = np.abs(res["z"].values[:, 0])
    peaks = z[: n_periods * steps].reshape(n_periods, steps).max(axis=1)
    return np.log(peaks[0] / peaks[-1]) / (2.0 * np.pi * (n_periods - 1))


@pytest.mark.parametrize("ratio", [0.03, 1.0])
def test_decay_damping_resolved_at_50_steps_per_period(ratio):
    """The damping of a free decay is resolved at 50 steps per period.

    The equivalent damping ratio is the hysteretic one of the law, which does
    not depend on the time step: at 50 steps per period it has to be within
    50% of the value at 200 steps per period, below and around ``chi0``.
    """
    coarse = _decay_damping(ratio, 50)
    fine = _decay_damping(ratio, 200)
    assert coarse == pytest.approx(fine, rel=0.5)
