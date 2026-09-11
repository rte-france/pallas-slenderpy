"""Tests for the future cable dynamic solver."""

import numpy as np
import pytest
from scipy.optimize import brentq

from slenderpy.future.cable import dynamic
from slenderpy.future.cable.static import catenary
from slenderpy.future.components import Conductor, Span
from slenderpy.future.simulation import Parameters

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
    from slenderpy.future.cable import frequency

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


# --- agreement with the legacy solver --------------------------------------


def test_matches_the_legacy_solver():
    """Same model, same scheme: the local state must agree to round-off."""
    from slenderpy import cable, simtools

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


def test_force_uses_the_local_convention():
    """force(s, t, un, ub, vn, vb) -> (fn, fb), s being the span fraction."""
    cd, sp = _conductor(), _span()
    pm = _parameters(tf=2.0, los=[0.5])
    seen = {}

    def force(s, t, un, ub, vn, vb):
        seen["s"] = s
        return np.zeros_like(s), 2.0 * np.ones_like(s)

    res = dynamic.solve(cd, sp, pm, force=force)
    assert seen["s"][0] == pytest.approx(0.0)
    assert seen["s"][-1] == pytest.approx(1.0)
    # a steady binormal force pushes the cable out of plane, ie towards -y
    assert res["y"].values[-1, 0] < -1.0e-03
