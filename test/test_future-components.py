from dataclasses import FrozenInstanceError

import pytest

from slenderpy.future.components import Conductor


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
