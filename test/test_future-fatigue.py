"""Tests for slenderpy.future.fatigue."""

import math
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from slenderpy.future import fatigue, simulation
from slenderpy.future.beam.dynamic import solve_dynamic
from slenderpy.future.boundary_condition import clamped
from slenderpy.future.cable import dynamic as cable_dynamic
from slenderpy.future.components import Conductor, Span
from slenderpy.future.force.core import Gravity, PointExcitation
from slenderpy.future.force.wind import ConstantWind, WindDrag
from slenderpy.future.simulation import Results

# ASTM E1049-85, rainflow counting example
ASTM = np.array([-2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0])
ASTM_COUNTS = {3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}


def _count(signal):
    return fatigue._rainflow(signal[fatigue._compress(signal)])


def _by_range(rows):
    grouped = defaultdict(float)
    for lrange, _, count in rows:
        grouped[lrange] += count
    return dict(grouped)


# --- counting helpers ------------------------------------------------------


def test_astm_reference_history():
    assert _by_range(_count(ASTM)) == ASTM_COUNTS


def test_matches_the_legacy_counter():
    from slenderpy import fatigue as legacy

    signal = np.random.default_rng(3).normal(size=2000).cumsum()
    legacy_rows = legacy._rainflow(signal[legacy._compress(signal)])[[0, 1, 3]].T
    assert np.array(_count(signal)) == pytest.approx(legacy_rows, rel=1e-14)


def test_compression_keeps_the_turning_points_and_both_ends():
    signal = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 0.5, 3.0, 2.0])
    assert fatigue._compress(signal).tolist() == [0, 2, 4, 6, 7]
    # counting the compressed signal again changes nothing
    compressed = signal[fatigue._compress(signal)]
    assert _count(compressed) == _count(signal)


def test_constant_signal_has_no_cycle():
    assert _count(np.full(50, 2.5)) == []


def test_plateaus_give_no_zero_range():
    signal = np.array([0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, 2.0, 2.0, 0.0])
    rows = _count(signal)
    assert all(lrange > 0.0 for lrange, _, _ in rows)
    # repeated samples must not hide cycles (the legacy counter loses them all)
    assert rows == _count(np.array([0.0, 1.0, -1.0, 2.0, 0.0]))
    assert rows


@pytest.mark.parametrize("signal", [np.array([1.0]), np.array([1.0, 1.0])])
def test_short_signals_give_an_empty_frame(signal):
    assert _count(signal) == []


# --- count_cycles ----------------------------------------------------------

SPAN = Span(length=100.0, tension=1.0e4)
LOS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _results(los=LOS):
    """Column j holds (j + 1) times the ASTM history; n_iter is a scalar."""
    lot = [float(k) for k in range(ASTM.size)]
    res = Results(lot=lot, lov=["z", "n_iter"], lov_dims=[2, 1], los=los)
    res.data["z"][:] = np.outer(ASTM, np.arange(1.0, len(los) + 1.0))
    res.data["n_iter"][:] = 1.0
    return res


def _max_range(frame):
    return frame["range"].max()


def test_output_layout():
    frame = fatigue.count_cycles(_results(), "z", SPAN, 25.0)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["range", "mean", "count"]
    assert _by_range(frame.itertuples(index=False)) == {
        2.0 * k: v for k, v in ASTM_COUNTS.items()
    }


@pytest.mark.parametrize("support, column", [("left", 1), ("right", 3)])
def test_support_selects_the_position(support, column):
    frame = fatigue.count_cycles(_results(), "z", SPAN, 25.0, support=support)
    assert _max_range(frame) == pytest.approx(9.0 * (column + 1))


def test_position_between_stored_ones_is_interpolated():
    frame = fatigue.count_cycles(_results(), "z", SPAN, 37.5)
    # halfway between columns 1 and 2, i.e. 2.5 times the history
    assert _max_range(frame) == pytest.approx(9.0 * 2.5)


def test_distance_zero_counts_the_support():
    assert _max_range(fatigue.count_cycles(_results(), "z", SPAN, 0.0)) == 9.0
    right = fatigue.count_cycles(_results(), "z", SPAN, 0.0, support="right")
    assert _max_range(right) == pytest.approx(45.0)


def test_stored_position_is_counted_exactly():
    fraction = 1.0 - 0.089 / SPAN.length
    res = _results(los=[0.0, 0.5, fraction, 1.0])
    frame = fatigue.count_cycles(res, "z", SPAN, 0.089, support="right")
    assert _max_range(frame) == 9.0 * 3.0


@pytest.mark.parametrize("los_head", [[0.5], []])
def test_edge_position_matched_up_to_round_off(los_head):
    """The near-support point computed by the user as the last stored one.

    (L - 0.089) / L and 1 - 0.089 / L differ by about 1e-16.
    """
    span = Span(length=4.9016908, tension=80.0)
    near = (span.length - 0.089) / span.length
    res = _results(los=los_head + [near])
    frame = fatigue.count_cycles(res, "z", span, 0.089, support="right")
    assert _max_range(frame) == 9.0 * (len(los_head) + 1)


def test_single_stored_position():
    res = _results(los=[0.5])
    assert _max_range(fatigue.count_cycles(res, "z", SPAN, 50.0)) == 9.0
    with pytest.raises(ValueError):
        fatigue.count_cycles(res, "z", SPAN, 40.0)


def test_constant_signal_gives_an_empty_frame():
    res = _results()
    res.data["z"][:] = 1.0
    frame = fatigue.count_cycles(res, "z", SPAN, 25.0)
    assert frame.empty
    assert list(frame.columns) == ["range", "mean", "count"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variable": "x"},  # not stored
        {"variable": "n_iter"},  # scalar
        {"support": "middle"},
        {"distance": -1.0},
        {"distance": 101.0},
        {"distance": math.nan},
        {"distance": math.inf},
    ],
)
def test_invalid_inputs_raise(kwargs):
    arguments = {"variable": "z", "distance": 25.0, "support": "left"} | kwargs
    with pytest.raises(ValueError):
        fatigue.count_cycles(_results(), span=SPAN, **arguments)


def test_position_outside_the_stored_ones_raises():
    with pytest.raises(ValueError, match="stored"):
        fatigue.count_cycles(_results(los=[0.25, 0.5]), "z", SPAN, 10.0)


def test_nan_in_the_signal_raises():
    res = _results()
    res.data["z"][-1, :] = np.nan
    with pytest.raises(ValueError, match="drop"):
        fatigue.count_cycles(res, "z", SPAN, 25.0)


# --- integration with the solvers ------------------------------------------

XB = 0.089


def _non_empty_and_finite(frame):
    assert not frame.empty
    assert np.all(np.isfinite(frame.to_numpy()))


def test_counts_a_beam_run_at_the_clamp():
    conductor = Conductor(mass=1.57, diameter=31.1e-3, ei_min=28.28, ei_max=2155.07)
    span = Span(length=4.9016908, tension=80.0, boundary_conditions=clamped())
    near = XB / span.length
    parameters = simulation.Parameters(
        ns=101, t0=0.0, tf=1.0, dt=0.002, dr=0.002, los=[0.0, near, 0.5, 1.0]
    )
    force = Gravity(conductor.mass) + PointExcitation(
        frequency=10.0, amplitude=20.0, position=0.5 * span.length
    )
    res = solve_dynamic(conductor, span, parameters, force=force)
    for variable in ("z", "moment"):
        _non_empty_and_finite(fatigue.count_cycles(res, variable, span, XB))


def test_counts_a_cable_run_near_support_2():
    conductor = Conductor(mass=1.571, diameter=0.0313, axial_stiffness=3.76e07)
    span = Span(length=400.0, tension=3.7e04)
    near = 1.0 - XB / span.length
    parameters = simulation.Parameters(
        ns=101, t0=0.0, tf=4.0, dt=2.0e-03, dr=1.0e-02, los=[0.5, near, 1.0]
    )
    # a vertical excitation alone leaves y at zero
    force = PointExcitation(frequency=1.0, amplitude=500.0, position=200.0) + WindDrag(
        diameter=conductor.diameter, wind=ConstantWind(10.0)
    )
    res = cable_dynamic.solve(conductor, span, parameters, force=force)
    for variable in ("y", "z"):
        frame = fatigue.count_cycles(res, variable, span, XB, support="right")
        _non_empty_and_finite(frame)
