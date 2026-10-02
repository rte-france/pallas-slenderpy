"""Tests for slenderpy.cable.static.shape."""

import numpy as np
import pytest

from slenderpy._constant import _GRAVITY
from slenderpy.cable import dynamic
from slenderpy.cable.static import shape
from slenderpy.components import Conductor, Span
from slenderpy.simulation import Parameters

MASS, DIAMETER, AXS = 1.571, 0.0313, 3.76e07
LSPAN, TENSION = 400.0, 3.7e04
WEIGHT = MASS * _GRAVITY
NS = 201
MID = NS // 2


def _conductor(axs=AXS):
    return Conductor(mass=MASS, diameter=DIAMETER, axial_stiffness=axs)


def _span(sld=0.0):
    return Span(length=LSPAN, tension=TENSION, sld=sld)


def _catenary(sld=0.0):
    return np.stack(dynamic.equilibrium(_conductor(), _span(sld), NS))


@pytest.mark.parametrize("sld", [0.0, 30.0])
def test_no_load_is_the_catenary(sld):
    position = shape.solve(_conductor(), _span(sld), 0.0, 0.0, NS)
    assert np.abs(position - _catenary(sld)).max() < 1e-12 * LSPAN


def test_lifting_half_the_weight_rises():
    position = shape.solve(_conductor(), _span(), 0.0, 0.5 * WEIGHT, NS)
    assert np.all(np.isfinite(position))
    assert position[2, MID] > _catenary()[2, MID] + 1.0


def test_downward_load_lowers():
    position = shape.solve(_conductor(), _span(), 0.0, -0.5 * WEIGHT, NS)
    assert position[2, MID] < _catenary()[2, MID] - 0.5


def test_small_horizontal_load_matches_the_taut_string():
    q = 1.0e-3 * WEIGHT
    position = shape.solve(_conductor(), _span(), q, 0.0, NS)
    expected = q * LSPAN**2 / (8.0 * TENSION)
    assert position[1, MID] == pytest.approx(expected, rel=1e-2)


def test_sloped_span_under_a_load():
    position = shape.solve(_conductor(), _span(30.0), 2.0, -0.2 * WEIGHT, NS)
    assert np.all(np.isfinite(position))
    assert position[:, 0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert position[:, -1] == pytest.approx([LSPAN, 0.0, 30.0], abs=1e-9)


def test_rhs_shapes():
    scalar = shape.solve(_conductor(), _span(), 1.0, 0.0, NS)
    array = shape.solve(_conductor(), _span(), np.ones(NS), np.zeros(NS), NS)
    assert scalar == pytest.approx(array, abs=1e-12)
    with pytest.raises(ValueError):
        shape.solve(_conductor(), _span(), np.ones(NS - 1), 0.0, NS)


def test_weightless_cable_gives_a_tensioned_shape():
    # lifting the whole weight: physically slack, but the small-displacement
    # model cannot see it and keeps a small tension (documented behaviour)
    position = shape.solve(_conductor(), _span(), 0.0, WEIGHT, NS)
    assert np.all(np.isfinite(position))
    assert position[2, MID] > _catenary()[2, MID]


def test_twice_the_weight_upwards_inverts_the_catenary():
    position = shape.solve(_conductor(), _span(), 0.0, 2.0 * WEIGHT, NS)
    assert position[2, MID] == pytest.approx(-_catenary()[2, MID], rel=2e-2)


@pytest.mark.parametrize("factor", [0.6, 0.7, 0.8, 0.85, 1.2, 1.5, 2.0])
def test_upward_loads_converge(factor):
    position = shape.solve(_conductor(), _span(), 0.0, factor * WEIGHT, NS)
    assert np.all(np.isfinite(position))


@pytest.mark.parametrize("lspan", [100.0, 400.0, 1000.0])
@pytest.mark.parametrize("sag_ratio", [0.01, 0.05, 0.12])
@pytest.mark.parametrize("ns", [101, 401, 2001])
def test_lateral_load_converges_over_spans_sags_and_grids(lspan, sag_ratio, ns):
    tension = WEIGHT * lspan / (8.0 * sag_ratio)
    span = Span(length=lspan, tension=tension)
    position = shape.solve(_conductor(), span, WEIGHT, 0.0, ns)
    assert np.all(np.isfinite(position))
    assert position[1, ns // 2] > 0.0


def test_non_finite_load_returns_nan():
    rhs = np.zeros(NS)
    rhs[MID] = np.nan
    assert np.all(np.isnan(shape.solve(_conductor(), _span(), 0.0, rhs, NS)))


@pytest.mark.parametrize(
    "conductor, ns",
    [(Conductor(mass=MASS, diameter=DIAMETER), NS), (_conductor(), 2)],
)
def test_invalid_inputs_raise(conductor, ns):
    with pytest.raises(ValueError):
        shape.solve(conductor, _span(), 0.0, 0.0, ns)


@pytest.mark.parametrize("fy, fz", [(0.0, -5.0), (3.0, 0.0), (3.0, -5.0)])
def test_static_shape_is_a_dynamic_fixed_point(fy, fz):
    """Started from the static shape under a constant force, nothing moves."""
    cd, sp = _conductor(), _span()
    ns = 101
    start = shape.solve(cd, sp, fy, fz, ns)
    pm = Parameters(ns=ns, t0=0.0, tf=2.0, dt=2.0e-03, dr=1.0e-02, los=[0.25, 0.5])
    res = dynamic.solve(
        cd,
        sp,
        pm,
        force=lambda x, t, y, z, vy, vz: (fy, fz),
        initial_position=start,
    )
    scale = np.abs(start - np.stack(dynamic.equilibrium(cd, sp, ns))).max()
    for name in ("x", "y", "z"):
        values = res[name].values
        assert np.abs(values - values[0]).max() < 1e-8 * scale, name
