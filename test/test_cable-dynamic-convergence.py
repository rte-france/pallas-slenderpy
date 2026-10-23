"""Time convergence of the cable dynamic solver across spans and loadings.

The same three spans as the beam convergence tests (an ASTER 570 conductor):
a long span (440 m, 39 kN), a medium span (50 m, 5 kN) and a jumper (5 m,
50 N), each run in free vibration, under a slow load at its first taut-string
frequency and under a fast one at 20 Hz (aeolian range).

The long and medium spans are loaded in plane, by a vertical point force. The
jumper is loaded out of plane, by a uniform horizontal load: with its 1 m sag
and the axial stiffness of the conductor, 1 mm of in-plane motion already
swings its tension to 50 times the static one, compression included, out of
the taut-cable model. The amplitudes keep every case taut, which is checked.

Crank-Nicolson suits the cable: its nonlinearity, the uniform dynamic strain,
is smooth, so the unresolved modes it leaves undamped have nothing to
destabilise (the beam needs damping of those modes because they switch the
branches of its Bouc-Wen law). With the implicit strain every case converges
at second order. The explicit strain only converges while the CFL number stays
below 1: once the strain grows the waves travel at the axial speed
``sqrt(EA/m)``, and with the slow steps of the medium span the coarse explicit
runs are off by orders of magnitude.
"""

import warnings

import numpy as np
import pytest

from slenderpy import simulation
from slenderpy.cable import dynamic
from slenderpy.components import Conductor, Span
from slenderpy.force.core import PointExcitation

CONDUCTOR = Conductor(mass=1.571, diameter=0.0313, axial_stiffness=3.76e07)

# span and number of nodes
SPANS = {
    "long": (Span(length=440.0, tension=39e3), 201),
    "medium": (Span(length=50.0, tension=5e3), 101),
    "jumper": (Span(length=5.0, tension=50.0), 51),
}

# spans loaded out of plane, see the module docstring
OUT_OF_PLANE = {"jumper"}

# frequency of the fast load (Hz)
FAST = 20.0

# free: peak displacement (m) of a sin initial velocity at the first frequency;
# slow and fast: point force amplitude (N) at a quarter span in plane, uniform
# load amplitude (N/m) out of plane
AMPLITUDE = {
    ("long", "free"): 0.5,
    ("long", "slow"): 200.0,
    ("long", "fast"): 50.0,
    ("medium", "free"): 0.05,
    ("medium", "slow"): 20.0,
    ("medium", "fast"): 20.0,
    ("jumper", "free"): 4.0e-3,
    ("jumper", "slow"): 0.01,
    ("jumper", "fast"): 2.0,
}

# time steps per period of the loading at the coarsest run, and periods run
STEPS = 20
PERIODS = {"free": 2, "slow": 3, "fast": 10}

# coarse runs at dt, dt/4 and dt/16, against a reference at dt/64
REFINEMENTS = (1, 4, 16)
REFERENCE = 64

# over the 16x refinement the error has to drop at least 4x, as for the beam.
# The implicit strain reaches more than 85 on every case, about 250 (second
# order) on most
_MIN_TOTAL_RATIO = 4.0


def _first_frequency(span):
    """First taut-string frequency of the span (Hz)."""
    return 0.5 / span.length * np.sqrt(span.tension / CONDUCTOR.mass)


def _uniform_horizontal_load(amplitude, frequency):
    """Sinusoidal horizontal load (N/m), uniform along the span."""

    def force(x, t, y, z, vy, vz):
        return amplitude * np.sin(2.0 * np.pi * frequency * t), 0.0

    return force


def _run(span_name, loading, refinement, implicit_strain):
    """Run of a (span, loading) case at ``dt / refinement``.

    The output is recorded at the coarsest step whatever the refinement, so
    the runs compare snapshot by snapshot. A CFL warning is raised as an error.
    """
    span, ns = SPANS[span_name]
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
    elif out_of_plane:
        kwargs["force"] = _uniform_horizontal_load(amplitude, f)
    else:
        kwargs["force"] = PointExcitation(
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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return dynamic.solve(
            CONDUCTOR, span, parameters, implicit_strain=implicit_strain, **kwargs
        )


def _l2_gap(first, second):
    """Root-mean-square gap between two recorded fields."""
    return float(np.sqrt(np.mean((np.asarray(first) - np.asarray(second)) ** 2)))


def _check_convergence(span_name, loading, implicit_strain):
    """The error decreases at each refinement and drops enough overall.

    Checked on the position along the loading (``y`` out of plane, ``z`` in
    plane) and on the dynamic tension. The cable has to stay taut.
    """
    span, _ = SPANS[span_name]
    reference = _run(span_name, loading, REFERENCE, implicit_strain)
    runs = {r: _run(span_name, loading, r, implicit_strain) for r in REFINEMENTS}

    assert np.min(span.tension + reference["dtension"].values) > 0.0

    position = "y" if span_name in OUT_OF_PLANE else "z"
    for field in (position, "dtension"):
        errors = [_l2_gap(runs[r][field], reference[field]) for r in REFINEMENTS]
        assert errors[0] > errors[1] > errors[2], (field, errors)
        assert errors[0] / errors[2] > _MIN_TOTAL_RATIO, (field, errors)


@pytest.mark.parametrize(
    "span_name, loading",
    [
        pytest.param(span, loading, id=f"{span}-{loading}")
        for span in SPANS
        for loading in ("free", "slow", "fast")
    ],
)
def test_implicit_strain_converges_in_time(span_name, loading):
    """With the implicit strain, refining the step converges on every case."""
    _check_convergence(span_name, loading, implicit_strain=True)


@pytest.mark.parametrize("span_name", ["long", "medium"])
def test_explicit_strain_converges_below_the_cfl_limit(span_name):
    """The explicit strain converges where the CFL number stays below 1.

    The fast loading of the long and medium spans keeps it there at every
    refinement; the run raises if it does not.
    """
    _check_convergence(span_name, "fast", implicit_strain=False)
