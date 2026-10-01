"""Tests for slenderpy.future.force.air, checked against the legacy formulas."""

import numpy as np
import pytest

from slenderpy import wind as legacy
from slenderpy.future.force.air import Air, cylinder_drag


@pytest.mark.parametrize("temperature", [263.15, 293.15, 313.15])
@pytest.mark.parametrize("pressure", [9.0e04, 1.013e05])
@pytest.mark.parametrize("humidity", [0.0, 0.5, 1.0])
def test_air_matches_the_legacy_formulas(temperature, pressure, humidity):
    air = Air(temperature=temperature, pressure=pressure, humidity=humidity)
    assert air.density == pytest.approx(
        legacy.air_volumic_mass(temperature, pressure, humidity), rel=1e-14
    )
    assert air.kinematic_viscosity == pytest.approx(
        legacy.air_kinematic_viscosity(temperature, pressure, humidity), rel=1e-14
    )


def test_default_air_is_about_1_2_kg_per_m3():
    assert Air().density == pytest.approx(1.2, rel=0.01)


def test_cylinder_drag_matches_the_legacy_formula():
    reynolds = np.logspace(0.0, 5.0, 50)
    assert cylinder_drag(reynolds) == pytest.approx(
        legacy.cylinder_drag(reynolds), rel=1e-14
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.0},
        {"temperature": -1.0},
        {"pressure": 0.0},
        {"humidity": -0.1},
        {"humidity": 1.1},
        {"humidity": float("nan")},
    ],
)
def test_invalid_air_raises(kwargs):
    with pytest.raises(ValueError):
        Air(**kwargs)
