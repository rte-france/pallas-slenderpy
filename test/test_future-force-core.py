"""Tests for slenderpy.future.force.core."""

import math

import numpy as np
import pytest

from slenderpy.future._constant import _GRAVITY
from slenderpy.future.force.core import ForceSum, Gravity, PointExcitation

X = np.linspace(0.0, 10.0, 11)


def _call(force, x=X, t=0.0):
    zeros = np.zeros_like(x)
    return force(x, t, zeros, zeros, zeros, zeros)


def test_gravity_points_down():
    fy, fz = _call(Gravity(2.0))
    assert fy == 0.0
    assert fz == pytest.approx(-_GRAVITY * 2.0)


def test_sum_flattens_its_terms():
    a, b, c, d = Gravity(1.0), Gravity(2.0), Gravity(3.0), Gravity(4.0)
    assert (a + b + c).terms == (a, b, c)
    assert len(((a + b) + (c + d)).terms) == 4
    assert isinstance(a + b, ForceSum)


def test_sum_adds_the_terms_outputs():
    excitation = PointExcitation(frequency=1.0, amplitude=3.0, position=5.0)
    total = Gravity(2.0) + excitation
    t = 0.2
    fy, fz = _call(total, t=t)
    expected = _call(Gravity(2.0), t=t)[1] + _call(excitation, t=t)[1]
    assert np.all(fy == 0.0)
    assert fz == pytest.approx(expected)


def test_adding_a_plain_function_raises():
    def plain(x, t, y, z, vy, vz):
        return 0.0, 0.0

    with pytest.raises(TypeError):
        Gravity(1.0) + plain


@pytest.mark.parametrize(
    "x", [np.linspace(0.0, 10.0, 11), np.array([0.0, 0.5, 2.0, 4.5, 5.2, 7.0, 10.0])]
)
def test_point_excitation_integrates_to_its_amplitude(x):
    force = PointExcitation(frequency=2.0, amplitude=3.0, position=5.0, t_start=0.1)
    t = 0.3
    _, fz = _call(force, x=x, t=t)
    i = int(np.argmin(np.abs(x - 5.0)))
    width = 0.5 * (x[i + 1] - x[i - 1])
    assert fz[i] * width == pytest.approx(3.0 * np.sin(2.0 * np.pi * 2.0 * (t - 0.1)))
    assert np.count_nonzero(fz) == 1


def test_point_excitation_is_off_outside_its_window():
    force = PointExcitation(
        frequency=1.0, amplitude=1.0, position=5.0, t_start=1.0, t_end=2.0
    )
    assert np.all(_call(force, t=0.99)[1] == 0.0)
    assert np.all(_call(force, t=2.01)[1] == 0.0)
    assert np.any(_call(force, t=1.25)[1] != 0.0)


@pytest.mark.parametrize("position, node", [(0.0, 1), (10.0, 9), (25.0, 9)])
def test_point_excitation_never_on_a_support(position, node):
    force = PointExcitation(frequency=1.0, amplitude=1.0, position=position)
    _, fz = _call(force, t=0.25)
    assert np.flatnonzero(fz).tolist() == [node]


def test_point_excitation_picks_the_nearest_node():
    x = np.array([0.0, 1.0, 3.0, 3.4, 8.0, 10.0])
    force = PointExcitation(frequency=1.0, amplitude=1.0, position=3.3)
    _, fz = _call(force, x=x, t=0.25)
    assert np.flatnonzero(fz).tolist() == [3]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Gravity(0.0),
        lambda: Gravity(-1.0),
        lambda: Gravity(math.nan),
        lambda: PointExcitation(frequency=0.0, amplitude=1.0, position=1.0),
        lambda: PointExcitation(frequency=1.0, amplitude=math.inf, position=1.0),
        lambda: PointExcitation(frequency=1.0, amplitude=1.0, position=-1.0),
        lambda: PointExcitation(
            frequency=1.0, amplitude=1.0, position=1.0, t_start=2.0, t_end=1.0
        ),
        lambda: PointExcitation(
            frequency=1.0, amplitude=1.0, position=1.0, t_start=math.nan
        ),
    ],
)
def test_invalid_inputs_raise(factory):
    with pytest.raises(ValueError):
        factory()
