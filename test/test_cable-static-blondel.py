"""Tests for slenderpy.cable.static.blondel."""

import warnings

import numpy as np
import pytest

from slenderpy.cable.static import blondel

WEIGHT = 1.571 * 9.81 * 400.0
AXS = 3.653e07
ALPHA = 2.3e-05


@pytest.mark.parametrize("tension_f", [5.0e03, 2.0e04, 3.7e04, 8.0e04])
def test_tension_inverts_temperature(tension_f):
    """The cubic root returned is the tension that produced the temperature."""
    tension_i = 3.7e04
    temperature_f = blondel.temperature(WEIGHT, tension_i, tension_f, 288.0, AXS, ALPHA)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tension = blondel.tension(WEIGHT, tension_i, 288.0, temperature_f, AXS, ALPHA)
    assert tension == pytest.approx(tension_f, rel=1e-09)


def test_tension_is_the_positive_root_of_the_cubic():
    """Also when the cubic has three real roots."""
    # a hot cable makes the quadratic coefficient large enough
    tension_i, temperature_f = 3.7e04, 400.0
    tension = blondel.tension(WEIGHT, tension_i, 288.0, temperature_f, AXS, ALPHA)
    b = (
        ALPHA * (temperature_f - 288.0)
        - tension_i / AXS
        + (WEIGHT / tension_i) ** 2 / 24
    )
    roots = np.roots([1.0 / AXS, b, 0.0, -(WEIGHT**2) / 24])
    real = roots[np.isreal(roots)].real
    assert real.size == 3
    assert tension == pytest.approx(real[real > 0].item(), rel=1e-09)


def test_tension_array():
    temperature_f = np.array([258.0, 288.0, 338.0])
    tension = blondel.tension(WEIGHT, 3.7e04, 288.0, temperature_f, AXS, ALPHA)
    assert tension.shape == (3,)
    assert np.all(np.diff(tension) < 0.0)
    assert tension[1] == pytest.approx(3.7e04, rel=1e-09)
