from dataclasses import FrozenInstanceError

import pytest

from slenderpy.future.beam.fd_utils import BoundaryCondition
from slenderpy.future.components import Conductor, Span


def test_conductor_minimal():
    c = Conductor(mass=1.571)
    assert c.mass == 1.571
    assert c.diameter is None
    assert c.axial_stiffness is None
    assert c.ei_min is None
    assert c.ei_max is None
    assert c.chi0 is None
    assert c.thermal_expansion is None


def test_conductor_full():
    c = Conductor(
        mass=1.571,
        diameter=0.0234,
        axial_stiffness=3.653e07,
        ei_min=10.0,
        ei_max=2155.0,
        chi0=0.03,
        thermal_expansion=2.3e-05,
    )
    assert c.diameter == 0.0234
    assert c.axial_stiffness == 3.653e07
    assert c.ei_min == 10.0
    assert c.ei_max == 2155.0
    assert c.chi0 == 0.03
    assert c.thermal_expansion == 2.3e-05


def test_conductor_is_frozen():
    c = Conductor(mass=1.571)
    with pytest.raises(FrozenInstanceError):
        c.mass = 2.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass": 0.0},
        {"mass": -1.0},
        {"mass": 1.0, "diameter": 0.0},
        {"mass": 1.0, "axial_stiffness": -1.0},
        {"mass": 1.0, "ei_min": 0.0},
        {"mass": 1.0, "ei_max": -5.0},
        {"mass": 1.0, "chi0": 0.0},
        {"mass": float("nan")},
        {"mass": float("inf")},
    ],
)
def test_conductor_rejects_non_positive(kwargs):
    with pytest.raises(ValueError):
        Conductor(**kwargs)


def test_conductor_rejects_ei_max_below_ei_min():
    with pytest.raises(ValueError):
        Conductor(mass=1.0, ei_min=100.0, ei_max=50.0)


def test_conductor_allows_ei_max_equal_ei_min():
    c = Conductor(mass=1.0, ei_min=100.0, ei_max=100.0)
    assert c.ei_min == c.ei_max == 100.0


def test_conductor_allows_negative_thermal_expansion():
    c = Conductor(mass=1.0, thermal_expansion=-1e-06)
    assert c.thermal_expansion == -1e-06


def test_span_minimal():
    s = Span(length=400.0, tension=30000.0)
    assert s.length == 400.0
    assert s.tension == 30000.0
    assert s.sld == 0.0
    assert s.boundary_conditions is None


def test_span_negative_sld_allowed():
    s = Span(length=400.0, tension=30000.0, sld=-5.0)
    assert s.sld == -5.0


def test_span_with_boundary_conditions():
    bc = BoundaryCondition(order=2)
    s = Span(length=400.0, tension=30000.0, boundary_conditions=bc)
    assert s.boundary_conditions is bc


def test_span_is_frozen():
    s = Span(length=400.0, tension=30000.0)
    with pytest.raises(FrozenInstanceError):
        s.length = 500.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length": 0.0, "tension": 30000.0},
        {"length": -1.0, "tension": 30000.0},
        {"length": 400.0, "tension": 0.0},
        {"length": 400.0, "tension": -1.0},
        {"length": 400.0, "tension": 30000.0, "sld": float("inf")},
        {"length": 400.0, "tension": 30000.0, "sld": float("nan")},
        {"length": float("nan"), "tension": 30000.0},
        {"length": float("inf"), "tension": 30000.0},
    ],
)
def test_span_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        Span(**kwargs)


def test_reexported_from_package():
    from slenderpy.future import Conductor as PkgConductor
    from slenderpy.future import Span as PkgSpan

    assert PkgConductor is Conductor
    assert PkgSpan is Span
