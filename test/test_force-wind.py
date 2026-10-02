"""Tests for slenderpy.force.wind."""

import warnings

import numpy as np
import pytest

from slenderpy.force.air import Air, cylinder_drag
from slenderpy.force.wind import (
    ConstantWind,
    TurbulentWindField,
    UniformTurbulentWind,
    WindDrag,
)

X = np.linspace(0.0, 100.0, 11)


def _uniform(seed=1, t_start=0.0):
    return UniformTurbulentWind(
        mean=10.0,
        std=1.5,
        length_scale=150.0,
        t_start=t_start,
        t_end=t_start + 60.0,
        dt=0.05,
        seed=seed,
    )


def _field(seed=1, length=100.0, n_points=11):
    return TurbulentWindField(
        mean=10.0,
        length=length,
        n_points=n_points,
        t_start=0.0,
        t_end=60.0,
        dt=0.25,
        seed=seed,
    )


# --- ConstantWind ----------------------------------------------------------


def test_constant_wind():
    assert ConstantWind(4.0).velocity(X, 1.0) == (4.0, 0.0)
    assert ConstantWind(-4.0).velocity(X, 1.0) == (-4.0, 0.0)


# --- UniformTurbulentWind --------------------------------------------------


def test_uniform_wind_hits_its_mean_and_std():
    wind = _uniform()
    assert np.mean(wind.speed) == pytest.approx(10.0, abs=1e-12)
    assert np.std(wind.speed) == pytest.approx(1.5, abs=1e-12)


def test_uniform_wind_is_reproducible():
    assert np.array_equal(_uniform(seed=3).speed, _uniform(seed=3).speed)
    assert not np.array_equal(_uniform(seed=3).speed, _uniform(seed=4).speed)


def test_uniform_wind_matches_legacy_random_wind():
    from slenderpy.legacy.turbwind import RandomWind1D

    legacy = RandomWind1D(
        mean=10.0, std=1.5, lxu=150.0, t0=0.0, tf=60.0, dt=0.05, seed=7
    )
    wind = _uniform(seed=7)
    assert wind.times == pytest.approx(legacy.t)
    assert wind.speed == pytest.approx(legacy.u, rel=1e-12)


def test_uniform_wind_interpolates_and_holds():
    wind = _uniform()
    t = 0.5 * (wind.times[10] + wind.times[11])
    wy, wz = wind.velocity(X, t)
    assert wy == pytest.approx(0.5 * (wind.speed[10] + wind.speed[11]))
    assert wz == 0.0
    assert wind.velocity(X, -5.0)[0] == pytest.approx(wind.speed[0])
    assert wind.velocity(X, 1e3)[0] == pytest.approx(wind.speed[-1])


def test_uniform_wind_time_vector_follows_t_start():
    wind = _uniform(t_start=100.0)
    assert wind.times[0] == pytest.approx(100.0)
    assert wind.times[-1] == pytest.approx(160.0)


# --- TurbulentWindField ----------------------------------------------------


def test_field_shapes_and_reproducibility():
    field = _field(seed=2)
    assert field.u.shape == (11, field.times.size)
    assert field.w.shape == field.u.shape
    assert np.array_equal(field.u, _field(seed=2).u)
    assert not np.array_equal(field.u, _field(seed=5).u)


def test_field_hits_its_targets():
    field = _field()
    assert np.mean(field.u) == pytest.approx(10.0, abs=1e-12)
    assert np.std(field.u) == pytest.approx(0.3, abs=1e-12)
    assert np.mean(field.w) == pytest.approx(0.0, abs=1e-12)
    assert np.std(field.w) == pytest.approx(0.1, abs=1e-12)


def test_field_coherence_decays_with_distance():
    # a long record: over 60 s only a few low-frequency cycles fit and the
    # sample correlations are noise, for the legacy generator too
    field = TurbulentWindField(
        mean=10.0, length=1000.0, n_points=21, t_start=0.0, t_end=600.0, dt=1.0, seed=1
    )
    u = field.u - field.u.mean(axis=1, keepdims=True)
    near = np.corrcoef(u[0], u[1])[0, 1]
    far = np.corrcoef(u[0], u[-1])[0, 1]
    assert near > far


def test_field_velocity_interpolates():
    field = _field()
    t = 0.5 * (field.times[4] + field.times[5])
    x = np.array([0.5 * (field.grid[2] + field.grid[3])])
    wy, wz = field.velocity(x, t)
    expected = 0.25 * (field.u[2, 4] + field.u[2, 5] + field.u[3, 4] + field.u[3, 5])
    assert wy[0] == pytest.approx(expected)
    assert wz.shape == (1,)


def test_field_velocity_clamps_outside_the_grid():
    field = _field()
    wy, wz = field.velocity(np.array([-50.0, 500.0]), 1e4)
    assert wy == pytest.approx([field.u[0, -1], field.u[-1, -1]])
    assert np.all(np.isfinite(wz))
    wy, _ = field.velocity(np.array([0.0]), -10.0)
    assert wy[0] == pytest.approx(field.u[0, 0])


# --- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ConstantWind(float("nan")),
        lambda: UniformTurbulentWind(
            mean=0.0, std=1.0, length_scale=1.0, t_start=0.0, t_end=10.0, dt=0.1
        ),
        lambda: UniformTurbulentWind(
            mean=1.0, std=1.0, length_scale=1.0, t_start=0.0, t_end=0.0, dt=0.1
        ),
        lambda: UniformTurbulentWind(
            mean=1.0, std=1.0, length_scale=1.0, t_start=0.0, t_end=10.0, dt=0.0
        ),
        # 3 and 5 samples: no harmonic fits, the normalised signal would be nan
        lambda: UniformTurbulentWind(
            mean=5.0, std=1.0, length_scale=100.0, t_start=0.0, t_end=0.2, dt=0.1
        ),
        lambda: UniformTurbulentWind(
            mean=5.0, std=1.0, length_scale=100.0, t_start=0.0, t_end=0.4, dt=0.1
        ),
        lambda: TurbulentWindField(
            mean=-1.0, length=10.0, n_points=5, t_start=0.0, t_end=10.0, dt=0.1
        ),
        lambda: TurbulentWindField(
            mean=1.0, length=10.0, n_points=1, t_start=0.0, t_end=10.0, dt=0.1
        ),
        lambda: TurbulentWindField(
            mean=1.0, length=10.0, n_points=5, t_start=0.0, t_end=1.0, dt=0.5
        ),
    ],
)
def test_invalid_winds_raise(factory):
    with pytest.raises(ValueError):
        factory()


# --- WindDrag --------------------------------------------------------------

D = 0.0313


def _state(n=5, vy=0.0, vz=0.0):
    x = np.linspace(0.0, 10.0, n)
    zeros = np.zeros(n)
    return x, zeros, zeros, zeros + vy, zeros + vz


def test_wind_drag_constant_coefficient_at_rest():
    x, y, z, vy, vz = _state()
    air = Air()
    drag = WindDrag(diameter=D, wind=ConstantWind(10.0), drag_coefficient=1.2, air=air)
    fy, fz = drag(x, 0.0, y, z, vy, vz)
    assert fy == pytest.approx(0.5 * air.density * D * 1.2 * 10.0 * 10.0)
    assert fz == pytest.approx(0.0)


def test_wind_drag_opposes_motion_in_still_air():
    x, y, z, vy, vz = _state(vz=0.5)
    fy, fz = WindDrag(diameter=D, wind=ConstantWind(0.0))(x, 0.0, y, z, vy, vz)
    assert np.all(fz < 0.0)
    assert fy == pytest.approx(0.0)


def test_wind_drag_default_law_uses_the_local_reynolds_number():
    x, y, z, vy, vz = _state(vz=-0.4)
    air = Air()
    fy, fz = WindDrag(diameter=D, wind=ConstantWind(3.0), air=air)(x, 0.0, y, z, vy, vz)
    speed = np.hypot(3.0, 0.4)
    cd = cylinder_drag(speed * D / air.kinematic_viscosity)
    gamma = 0.5 * air.density * D * cd * speed
    assert fy == pytest.approx(gamma * 3.0)
    assert fz == pytest.approx(gamma * 0.4)


def test_wind_drag_zero_relative_speed_is_silent():
    x, y, z, vy, vz = _state(vy=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fy, fz = WindDrag(diameter=D, wind=ConstantWind(2.0))(x, 0.0, y, z, vy, vz)
    assert np.all(np.asarray(fy) == 0.0)
    assert np.all(np.asarray(fz) == 0.0)


def test_wind_drag_matches_legacy_auto_drag():
    from slenderpy.legacy.wind import AutoDrag

    # the formula is frame independent: legacy b is y, legacy n is z here
    x, y, z, vy, vz = _state(vy=0.7, vz=-0.3)
    fn, fb = AutoDrag(u=8.0, d=D)(x, 0.0, None, None, vz, vy)
    fy, fz = WindDrag(diameter=D, wind=ConstantWind(8.0))(x, 0.0, y, z, vy, vz)
    assert fy == pytest.approx(fb, rel=1e-12)
    assert fz == pytest.approx(fn, rel=1e-12)


def test_wind_drag_matches_legacy_turbulent_force():
    from slenderpy.legacy.turbwind import Force1D

    air = Air()
    legacy = Force1D(
        mean=10.0,
        std=1.5,
        lxu=150.0,
        t0=0.0,
        tf=60.0,
        dt=0.05,
        cd=1.1,
        rho=air.density,
        d=D,
    )
    wind = UniformTurbulentWind(
        mean=10.0, std=1.5, length_scale=150.0, t_start=0.0, t_end=60.0, dt=0.05
    )
    # use the legacy signal in the new wind, the draws are seeded differently
    object.__setattr__(wind, "speed", legacy.wnd.u)
    x, y, z, vy, vz = _state(vy=0.2, vz=0.1)
    t = 12.34
    fn, fb = legacy(x, t, None, None, vz, vy)
    drag = WindDrag(diameter=D, wind=wind, drag_coefficient=1.1, air=air)
    fy, fz = drag(x, t, y, z, vy, vz)
    assert fy == pytest.approx(fb, rel=1e-12)
    assert fz == pytest.approx(fn, rel=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diameter": 0.0},
        {"diameter": D, "drag_coefficient": 0.0},
        {"diameter": D, "drag_coefficient": float("nan")},
    ],
)
def test_invalid_wind_drag_raises(kwargs):
    with pytest.raises(ValueError):
        WindDrag(wind=ConstantWind(1.0), **kwargs)
