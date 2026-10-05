"""Tests for the cable dynamic solver."""

import numpy as np
import pytest
from scipy.optimize import brentq

from slenderpy._constant import _GRAVITY
from slenderpy.cable import dynamic
from slenderpy.cable.static import catenary
from slenderpy.components import Conductor, Span
from slenderpy.force.air import Air
from slenderpy.force.core import Gravity, PointExcitation
from slenderpy.force.wind import ConstantWind, WindDrag
from slenderpy.simulation import Parameters

G = 9.81
MASS, DIAMETER, AXS = 1.571, 0.0313, 3.76e07
LSPAN, TENSION = 400.0, 3.7e04


def _conductor():
    return Conductor(mass=MASS, diameter=DIAMETER, axial_stiffness=AXS)


def _span(sld=0.0):
    return Span(length=LSPAN, tension=TENSION, sld=sld)


# --- geometry --------------------------------------------------------------


@pytest.mark.parametrize("sld", [0.0, 30.0, -30.0])
def test_equilibrium_ends_sit_on_the_supports(sld):
    x, y, z = dynamic.equilibrium(_conductor(), _span(sld), 51)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(LSPAN)
    assert z[0] == pytest.approx(0.0)
    assert z[-1] == pytest.approx(sld)
    assert y == pytest.approx(np.zeros_like(y))


@pytest.mark.parametrize("sld", [0.0, 30.0])
def test_equilibrium_lies_on_the_catenary(sld):
    x, _, z = dynamic.equilibrium(_conductor(), _span(sld), 101)
    expected = catenary.shape(x, LSPAN, TENSION, sld, MASS, g=G)
    assert z == pytest.approx(expected, abs=1.0e-09)


@pytest.mark.parametrize("sld", [0.0, 30.0])
def test_nodes_are_evenly_spaced_along_the_arc(sld):
    """The solve grid is uniform in arc length, not in span."""
    geom = dynamic._geometry(_conductor(), _span(sld), 101)
    arc = np.cumsum(
        np.concatenate(([0.0], np.sqrt(np.diff(geom.x) ** 2 + np.diff(geom.z) ** 2)))
    )
    # chord length underestimates the arc, so compare the spacing, not the total
    step = np.diff(arc)
    assert step.std() / step.mean() < 1.0e-04
    assert np.all(np.diff(geom.x) > 0.0)


@pytest.mark.parametrize("sld", [0.0, 30.0])
def test_slope_matches_the_catenary_derivative(sld):
    geom = dynamic._geometry(_conductor(), _span(sld), 21)
    eps = 1.0e-05
    x = geom.x[1:-1]
    fd = (
        catenary.shape(x + eps, LSPAN, TENSION, sld, MASS, g=G)
        - catenary.shape(x - eps, LSPAN, TENSION, sld, MASS, g=G)
    ) / (2.0 * eps)
    assert geom.slope[1:-1] == pytest.approx(fd, rel=1.0e-06)


@pytest.mark.parametrize("sld", [0.0, 30.0])
def test_triad_is_orthonormal_and_right_handed(sld):
    geom = dynamic._geometry(_conductor(), _span(sld), 21)
    et, en, eb = geom.et, geom.en, geom.eb
    assert (et * et).sum(0) == pytest.approx(np.ones(21))
    assert (en * en).sum(0) == pytest.approx(np.ones(21))
    assert (eb * eb).sum(0) == pytest.approx(np.ones(21))
    assert (et * en).sum(0) == pytest.approx(np.zeros(21), abs=1.0e-14)
    assert (et * eb).sum(0) == pytest.approx(np.zeros(21), abs=1.0e-14)
    assert np.cross(et.T, en.T).T == pytest.approx(eb, abs=1.0e-14)


# --- coordinate round trip -------------------------------------------------


def test_round_trip_without_tangential_part():
    """A displacement spanned by (en, eb) survives local -> global -> local."""
    ns = 41
    geom = dynamic._geometry(_conductor(), _span(30.0), ns)
    s = geom.s
    un = 0.7 * np.sin(np.pi * s)
    ub = -0.3 * np.sin(2.0 * np.pi * s)
    position = geom.equilibrium() + un * geom.en + ub * geom.eb

    un_back, ub_back, _, _ = dynamic._to_local(position, None, geom)
    assert un_back == pytest.approx(un, abs=1.0e-12)
    assert ub_back == pytest.approx(ub, abs=1.0e-12)


def test_round_trip_projects_out_the_tangential_part():
    """The model derives ut, so a tangential offset is discarded on input."""
    ns = 41
    geom = dynamic._geometry(_conductor(), _span(30.0), ns)
    un = 0.7 * np.sin(np.pi * geom.s)
    base = geom.equilibrium() + un * geom.en
    tangential = base + 0.5 * np.sin(np.pi * geom.s) * geom.et

    for p in (base, tangential):
        un_back, ub_back, _, _ = dynamic._to_local(p, None, geom)
        assert un_back == pytest.approx(un, abs=1.0e-12)
        assert ub_back == pytest.approx(np.zeros(ns), abs=1.0e-12)


def test_to_global_reproduces_equilibrium_for_a_zero_state():
    ns = 31
    geom = dynamic._geometry(_conductor(), _span(30.0), ns)
    zero = np.zeros(ns)
    xyz = dynamic._to_global(zero, zero, zero, geom)
    assert xyz == pytest.approx(geom.equilibrium())


# --- solver ----------------------------------------------------------------


def _parameters(tf=10.0, dt=2.0e-03, dr=1.0e-02, ns=101, los=None):
    return Parameters(
        ns=ns,
        t0=0.0,
        tf=tf,
        dt=dt,
        dr=dr,
        los=[0.25, 0.5] if los is None else los,
        pp=False,
    )


def test_equilibrium_is_a_fixed_point():
    """No force, no damping, starting at rest on the catenary: nothing moves."""
    cd, sp = _conductor(), _span(30.0)
    pm = _parameters(tf=5.0)
    res = dynamic.solve(cd, sp, pm)

    x, y, z = dynamic.equilibrium(cd, sp, pm.ns)
    span_fraction = dynamic._geometry(cd, sp, pm.ns).x / sp.length
    for i, s in enumerate(pm.los):
        assert res["x"].values[:, i] == pytest.approx(
            np.interp(s, span_fraction, x), abs=1.0e-06
        )
        assert res["y"].values[:, i] == pytest.approx(
            np.interp(s, span_fraction, y), abs=1.0e-09
        )
        assert res["z"].values[:, i] == pytest.approx(
            np.interp(s, span_fraction, z), abs=1.0e-06
        )
    assert np.abs(res["dtension"].values).max() < 1.0e-06


def test_stored_height_at_rest_matches_the_catenary_at_that_span_position():
    """los is a span fraction, so z(los) is the catenary at los*Lp."""
    cd, sp = _conductor(), _span(30.0)
    pm = _parameters(tf=1.0, los=[0.25, 0.5, 0.75])
    res = dynamic.solve(cd, sp, pm)
    for i, s in enumerate(pm.los):
        expected = catenary.shape(
            s * sp.length, sp.length, sp.tension, sp.sld, cd.mass, g=G
        )
        assert res["z"].values[0, i] == pytest.approx(expected, abs=1.0e-03)


def _period(t, y):
    """Period of a dominantly single-mode signal, from its zero crossings.

    Preferred over an FFT peak here: the bin width 1/tf is coarser than the
    tolerance being asserted, so argmax would snap to the nearest bin.
    """
    y = y - y.mean()
    idx = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]
    assert len(idx) > 4, "signal does not oscillate enough to time it"
    crossings = t[idx] - y[idx] * (t[idx + 1] - t[idx]) / (y[idx + 1] - y[idx])
    return 2.0 * np.mean(np.diff(crossings))


def _free_response(offset, direction, tf=60.0, amplitude=0.02):
    """Run a free vibration from a first-mode offset along a triad direction."""
    cd, sp = _conductor(), _span(0.0)
    pm = _parameters(tf=tf, dt=2.0e-03, dr=1.0e-02, ns=101, los=[0.5])
    geom = dynamic._geometry(cd, sp, pm.ns)
    shape = amplitude * np.sin(np.pi * geom.s)
    position = geom.equilibrium() + shape * getattr(geom, direction)
    res = dynamic.solve(cd, sp, pm, initial_position=position)
    return geom, res, np.array(res.lot())


def test_out_of_plane_frequency_matches_the_taut_string():
    from slenderpy.cable import frequency

    geom, res, t = _free_response(0.02, "eb")
    expected = frequency.natural(LSPAN, TENSION, 0.0, MASS, method="catenary")
    measured = 1.0 / _period(t, res["y"].values[:, 0])
    assert measured == pytest.approx(expected, rel=0.02)


def test_in_plane_frequency_matches_the_irvine_root():
    """The linearised in-plane operator is Irvine's, with lambda^2 = vl2/vt2^3."""
    cd, sp = _conductor(), _span(0.0)
    geom = dynamic._geometry(cd, sp, 101)
    vt2 = sp.tension / (cd.mass * G * geom.length)
    vl2 = cd.axial_stiffness / (cd.mass * G * geom.length)
    lm2 = vl2 / vt2**3

    def fun(x):
        return np.tan(0.5 * x) - 0.5 * x + 0.5 * x**3 / lm2

    eps = 1.0e-09
    k = brentq(fun, np.pi + eps, 3.0 * np.pi - eps, xtol=1.0e-12)
    f0 = np.sqrt(vt2) / (2.0 * np.sqrt(geom.length / G))
    expected = f0 * k / np.pi

    _, res, t = _free_response(0.02, "en")
    measured = 1.0 / _period(t, res["z"].values[:, 0])
    assert measured == pytest.approx(expected, rel=0.02)


def test_binormal_offset_moves_only_the_out_of_plane_coordinate():
    cd, sp = _conductor(), _span(30.0)
    pm = _parameters(tf=1.0, los=[0.5])
    geom = dynamic._geometry(cd, sp, pm.ns)
    position = geom.equilibrium() + 0.05 * np.sin(np.pi * geom.s) * geom.eb
    res = dynamic.solve(cd, sp, pm, initial_position=position)

    x, y, z = dynamic.equilibrium(cd, sp, pm.ns)
    frac = geom.x / sp.length
    assert res["y"].values[0, 0] != pytest.approx(0.0, abs=1.0e-06)
    assert res["x"].values[0, 0] == pytest.approx(np.interp(0.5, frac, x), abs=1.0e-06)
    assert res["z"].values[0, 0] == pytest.approx(np.interp(0.5, frac, z), abs=1.0e-06)


def test_tangential_offset_changes_nothing():
    cd, sp = _conductor(), _span(30.0)
    pm = _parameters(tf=2.0, los=[0.5])
    geom = dynamic._geometry(cd, sp, pm.ns)
    shape = 0.05 * np.sin(np.pi * geom.s)
    plain = dynamic.solve(
        cd, sp, pm, initial_position=geom.equilibrium() + shape * geom.en
    )
    tangential = dynamic.solve(
        cd,
        sp,
        pm,
        initial_position=geom.equilibrium() + shape * geom.en + shape * geom.et,
    )
    for v in ["x", "y", "z", "dtension"]:
        assert plain[v].values == pytest.approx(tangential[v].values)


def test_damping_decays_the_free_response():
    cd, sp = _conductor(), _span(0.0)
    pm = _parameters(tf=40.0, los=[0.5])
    geom = dynamic._geometry(cd, sp, pm.ns)
    position = geom.equilibrium() + 0.05 * np.sin(np.pi * geom.s) * geom.eb

    amplitudes = []
    for zeta in [0.0, 0.02]:
        res = dynamic.solve(cd, sp, pm, initial_position=position, zeta=zeta)
        y = res["y"].values[:, 0]
        amplitudes.append(np.abs(y[-len(y) // 5 :]).max())
    assert amplitudes[1] < 0.5 * amplitudes[0]


def test_initial_velocity_is_taken_into_account():
    cd, sp = _conductor(), _span(0.0)
    pm = _parameters(tf=2.0, los=[0.5])
    geom = dynamic._geometry(cd, sp, pm.ns)
    velocity = 0.1 * np.sin(np.pi * geom.s) * geom.eb
    at_rest = dynamic.solve(cd, sp, pm)
    moving = dynamic.solve(cd, sp, pm, initial_velocity=velocity)
    assert np.abs(moving["y"].values).max() > 1.0e-03
    assert np.abs(at_rest["y"].values).max() < 1.0e-09


def test_state_allows_a_restart():
    cd, sp = _conductor(), _span(0.0)
    geom = dynamic._geometry(cd, sp, 101)
    position = geom.equilibrium() + 0.02 * np.sin(np.pi * geom.s) * geom.eb

    whole = dynamic.solve(
        cd, sp, _parameters(tf=4.0, los=[0.5]), initial_position=position
    )
    first = dynamic.solve(
        cd, sp, _parameters(tf=2.0, los=[0.5]), initial_position=position
    )
    second = dynamic.solve(
        cd,
        sp,
        Parameters(ns=101, t0=2.0, tf=4.0, dt=2.0e-03, dr=1.0e-02, los=[0.5], pp=False),
        initial_position=first.state["position"],
        initial_velocity=first.state["velocity"],
    )
    assert second["y"].values[-1, 0] == pytest.approx(
        whole["y"].values[-1, 0], abs=1.0e-06
    )
    assert second.lot() == pytest.approx((2.0 + 0.01 * np.arange(201)).tolist())


@pytest.mark.parametrize("sld", [0.0, 30.0])
def test_supports_can_be_stored(sld):
    """los may hold 0 and 1: the stored ends sit on the supports all along."""
    cd, sp = _conductor(), _span(sld)
    geom = dynamic._geometry(cd, sp, 101)
    position = geom.equilibrium() + 0.02 * np.sin(np.pi * geom.s) * geom.en
    res = dynamic.solve(
        cd, sp, _parameters(tf=1.0, los=[0.0, 0.5, 1.0]), initial_position=position
    )
    assert res["x"].values[:, 0] == pytest.approx(0.0, abs=1.0e-06)
    assert res["x"].values[:, -1] == pytest.approx(LSPAN, abs=1.0e-06)
    assert res["y"].values[:, [0, -1]] == pytest.approx(0.0, abs=1.0e-09)
    assert res["z"].values[:, 0] == pytest.approx(0.0, abs=1.0e-06)
    assert res["z"].values[:, -1] == pytest.approx(sld, abs=1.0e-06)


# --- agreement with the legacy solver --------------------------------------


@pytest.mark.parametrize("ns", [11, 101, 401])
def test_operators_match_the_legacy_ones(ns):
    """Built from slenderpy.fd_utils, they must equal legacy d1M and d2M."""
    from slenderpy.legacy import fdm_utils

    ds = np.diff(np.linspace(0.0, 1.0, ns))
    first, second = dynamic._operators(ns)
    for mine, legacy in ((first, fdm_utils.d1M(ds)), (second, fdm_utils.d2M(ds))):
        legacy = legacy.toarray()
        assert mine.shape == legacy.shape
        assert np.abs(mine.toarray() - legacy).max() < 1e-12 * np.abs(legacy).max()


def test_matches_the_legacy_solver():
    """Same model, same scheme: the local state must agree to round-off."""
    from slenderpy.legacy import cable, simtools

    cd, sp = _conductor(), _span(0.0)
    ns, tf, dt, dr = 101, 4.0, 2.0e-03, 1.0e-02
    geom = dynamic._geometry(cd, sp, ns)
    amplitude = 0.05
    un0 = amplitude * np.sin(np.pi * geom.s)
    ub0 = 0.5 * amplitude * np.sin(2.0 * np.pi * geom.s)

    los_arc = [0.25, 0.5, 0.75]
    cb = cable.SCable(
        mass=MASS, diameter=DIAMETER, EA=AXS, length=LSPAN, tension=TENSION, h=0.0
    )
    legacy = cable.solve(
        cb,
        simtools.Parameters(ns=ns, t0=0.0, tf=tf, dt=dt, dr=dr, los=los_arc, pp=False),
        un0=un0,
        ub0=ub0,
    )

    # the new solver stores by span fraction; ask for the same material points
    los_span = (geom.x / sp.length)[
        [int(round(s * (ns - 1))) for s in los_arc]
    ].tolist()
    position = geom.equilibrium() + un0 * geom.en + ub0 * geom.eb
    new = dynamic.solve(
        cd,
        sp,
        Parameters(ns=ns, t0=0.0, tf=tf, dt=dt, dr=dr, los=los_span, pp=False),
        initial_position=position,
    )

    # convert the new global output back to local at those positions
    idx = [int(round(s * (ns - 1))) for s in los_arc]
    for j, i in enumerate(idx):
        d = np.stack(
            [
                new["x"].values[:, j] - geom.x[i],
                new["y"].values[:, j],
                new["z"].values[:, j] - geom.z[i],
            ]
        )
        un = d[0] * geom.en[0, i] + d[2] * geom.en[2, i]
        ub = -d[1]
        # one absolute tolerance from the overall motion amplitude, not a
        # per-component relative one: s=0.5 is a node of the ub initial shape,
        # so ub is round-off there and a relative measure would be meaningless
        scale = max(
            np.abs(legacy["un"].values).max(), np.abs(legacy["ub"].values).max()
        )
        assert un == pytest.approx(legacy["un"].values[:, j], abs=1.0e-09 * scale)
        assert ub == pytest.approx(legacy["ub"].values[:, j], abs=1.0e-09 * scale)


# --- validation ------------------------------------------------------------


def test_axial_stiffness_is_required():
    with pytest.raises(ValueError, match="axial_stiffness"):
        dynamic.solve(Conductor(mass=MASS), _span(), _parameters(tf=1.0))


@pytest.mark.parametrize("shape", [(2, 101), (3, 50), (101,)])
def test_initial_position_shape_is_checked(shape):
    pm = _parameters(tf=1.0)
    with pytest.raises(ValueError, match="shape"):
        dynamic.solve(_conductor(), _span(), pm, initial_position=np.zeros(shape))


def test_ends_must_stay_on_the_supports():
    cd, sp = _conductor(), _span(30.0)
    pm = _parameters(tf=1.0)
    geom = dynamic._geometry(cd, sp, pm.ns)
    position = geom.equilibrium()
    position[2, 0] += 0.01
    with pytest.raises(ValueError, match="support"):
        dynamic.solve(cd, sp, pm, initial_position=position)


def test_cfl_violation_warns():
    cd, sp = _conductor(), _span()
    pm = Parameters(ns=401, t0=0.0, tf=10.0, dt=0.2, dr=0.2, los=[0.5], pp=False)
    with pytest.warns(UserWarning, match="CFL"):
        dynamic.solve(cd, sp, pm)


@pytest.mark.parametrize(
    "force",
    [
        Gravity(MASS),
        PointExcitation(frequency=1.0, amplitude=1.0, position=200.0) + Gravity(MASS),
    ],
)
def test_gravity_is_refused(force):
    with pytest.raises(ValueError, match="equilibrium"):
        dynamic.solve(_conductor(), _span(), _parameters(tf=0.1), force=force)


def test_force_receives_the_global_state():
    cd, sp = _conductor(), _span(30.0)
    seen = {}

    def force(x, t, y, z, vy, vz):
        seen.update(x=x, z=z)
        return 0.0, 0.0

    dynamic.solve(cd, sp, _parameters(tf=0.1), force=force)
    assert seen["x"][0] == pytest.approx(0.0)
    assert seen["x"][-1] == pytest.approx(LSPAN)
    assert seen["z"][-1] == pytest.approx(30.0)


def test_scalar_force_is_broadcast():
    res = dynamic.solve(
        _conductor(),
        _span(),
        _parameters(tf=1.0, los=[0.5]),
        force=lambda x, t, y, z, vy, vz: (0.0, 1.0),
    )
    assert np.all(np.isfinite(res["z"].values))


def test_vertical_force_stays_in_plane():
    cd, sp = _conductor(), _span()
    # from the catenary: the default start would be the static shape, at rest
    res = dynamic.solve(
        cd,
        sp,
        _parameters(tf=2.0, los=[0.5]),
        force=lambda x, t, y, z, vy, vz: (0.0, 2.0 * np.ones_like(x)),
        initial_position=np.stack(dynamic.equilibrium(cd, sp, 101)),
    )
    assert np.abs(res["y"].values).max() < 1e-12
    assert np.ptp(res["z"].values[:, 0]) > 1e-03


def test_horizontal_force_moves_out_of_plane():
    cd, sp = _conductor(), _span()
    pm = _parameters(tf=2.0, los=[0.5])
    rest = dynamic.solve(cd, sp, pm)
    res = dynamic.solve(
        cd, sp, pm, force=lambda x, t, y, z, vy, vz: (2.0 * np.ones_like(x), 0.0)
    )
    y = res["y"].values[:, 0]
    assert y[-1] > 1e-03
    # the in-plane response comes only through the strain, at second order
    drift = np.abs(res["z"].values[:, 0] - rest["z"].values[:, 0]).max()
    assert drift < 0.1 * np.abs(y).max()


def test_wind_drag_pushes_the_cable_downwind():
    drag = WindDrag(diameter=DIAMETER, wind=ConstantWind(15.0), drag_coefficient=1.0)
    res = dynamic.solve(
        _conductor(), _span(), _parameters(tf=2.0, los=[0.5]), force=drag
    )
    assert res["y"].values[-1, 0] > 1e-03


def test_wind_drag_matches_the_legacy_solver():
    """Legacy cable under AutoDrag against the cable under WindDrag.

    Legacy blows along +b, which is -y here. Not to round-off: legacy feeds the
    force the local vn, this solver the global vz = vn / N and projects
    back with another 1 / N, so the in-plane damping differs by N**2.
    """
    from slenderpy.legacy import cable, simtools
    from slenderpy.legacy.wind import AutoDrag

    cd, sp = _conductor(), _span(0.0)
    ns, tf, dt, dr = 101, 4.0, 2.0e-03, 1.0e-02
    cb = cable.SCable(
        mass=MASS, diameter=DIAMETER, EA=AXS, length=LSPAN, tension=TENSION, h=0.0
    )
    legacy = cable.solve(
        cb,
        simtools.Parameters(ns=ns, t0=0.0, tf=tf, dt=dt, dr=dr, los=[0.5], pp=False),
        force=AutoDrag(u=10.0, d=DIAMETER),
    )
    new = dynamic.solve(
        cd,
        sp,
        Parameters(ns=ns, t0=0.0, tf=tf, dt=dt, dr=dr, los=[0.5], pp=False),
        force=WindDrag(diameter=DIAMETER, wind=ConstantWind(-10.0), air=Air()),
        # legacy starts on the catenary, not on the static shape
        initial_position=np.stack(dynamic.equilibrium(cd, sp, ns)),
    )
    geom = dynamic._geometry(cd, sp, ns)
    offset = new.state["position"] - geom.equilibrium()
    # measured gaps: 3e-06 binormal, 1.4e-04 normal (the N**2 damping factor)
    for direction, legacy_name, rtol in [
        (geom.eb, "ub", 1e-04),
        (geom.en, "un", 1e-03),
    ]:
        mine = np.sum(offset * direction, axis=0)
        reference = legacy.state[legacy_name]
        scale = np.abs(reference).max()
        assert scale > 1e-02, legacy_name
        assert np.abs(mine - reference).max() < rtol * scale, legacy_name


def test_project_matches_the_triad():
    from slenderpy.cable import _model

    geom = _model._geometry(_conductor(), _span(30.0), 21)
    fy, fz = 2.0 * np.ones(21), -3.0 * np.ones(21)
    fn, fb = _model._project(fy, fz, geom)
    force = np.stack([np.zeros(21), fy, fz])
    assert fn == pytest.approx(np.sum(force * geom.en, axis=0))
    assert fb == pytest.approx(np.sum(force * geom.eb, axis=0))


def test_default_start_is_the_static_shape_under_the_force():
    cd, sp = _conductor(), _span()
    drag = WindDrag(diameter=DIAMETER, wind=ConstantWind(10.0), drag_coefficient=1.0)
    res = dynamic.solve(cd, sp, _parameters(tf=1.0, los=[0.5]), force=drag)
    y = res["y"].values[:, 0]
    assert y[0] > 1e-02
    assert np.abs(y - y[0]).max() < 1e-6 * y[0]


def test_default_start_without_force_is_the_catenary():
    cd, sp = _conductor(), _span(30.0)
    res = dynamic.solve(cd, sp, _parameters(tf=0.1, los=[0.5]))
    expected = np.stack(dynamic.equilibrium(cd, sp, 101))
    assert res.state["position"] == pytest.approx(expected, abs=1e-12 * LSPAN)


def test_default_start_with_a_scalar_force():
    res = dynamic.solve(
        _conductor(),
        _span(),
        _parameters(tf=0.2, los=[0.5]),
        force=lambda x, t, y, z, vy, vz: (0.0, -1.0),
    )
    assert np.all(np.isfinite(res["z"].values))


def test_default_start_without_static_shape_raises():
    # a non-finite load at t0 has no static shape
    with pytest.raises(ValueError, match="static shape"):
        dynamic.solve(
            _conductor(),
            _span(),
            _parameters(tf=0.1),
            force=lambda x, t, y, z, vy, vz: (0.0, np.nan),
        )


def test_default_start_on_a_short_slack_span():
    """The reviewer's case: 100 m at 5% sag, ns = 401, under wind drag."""
    weight = MASS * _GRAVITY
    span = Span(length=100.0, tension=weight * 100.0**2 / (8.0 * 5.0))
    drag = WindDrag(diameter=DIAMETER, wind=ConstantWind(20.0))
    pm = Parameters(ns=401, t0=0.0, tf=0.1, dt=1.0e-03, dr=1.0e-02, los=[0.5])
    res = dynamic.solve(_conductor(), span, pm, force=drag)
    assert np.all(np.isfinite(res["y"].values))
