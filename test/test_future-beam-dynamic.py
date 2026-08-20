"""Tests for the dynamic beam solver of slenderpy.future.beam.dynamic.

Two families: manufactured solutions, which check the discretisation against a
closed form, and property tests, which check the assembly itself (fixed point,
conservation, hysteresis bounds, output layout).
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from slenderpy.future import simulation
from slenderpy.future._constant import _GRAVITY
from slenderpy.future.beam import bending
from slenderpy.future.beam.beam import BeamConst
from slenderpy.future.beam.bending import BendingModel
from slenderpy.future.beam.dynamic import solve_dynamic
from slenderpy.future.boundary_condition import BoundaryCondition, clamped, hinged
from slenderpy.future.components import Conductor, Span

CONDUCTOR = Conductor(
    mass=1.57,
    diameter=31.1e-3,
    ei_min=28.28,
    ei_max=2155.07,
    beta_flexion=6.437693e-07,
)
SPAN = Span(length=440.0, tension=39e3, boundary_conditions=hinged())
BRETELLE = Span(length=4.9016908, tension=80.0, boundary_conditions=clamped())

# the four (model, curvature) combinations the solver has to cover
CASES = [
    (BendingModel.CONSTANT, True),
    (BendingModel.CONSTANT, False),
    (BendingModel.VARYING, True),
    (BendingModel.VARYING, False),
]


def _plot_animation(x, exact, sol, ymin, ymax, nb_time, final_time):
    """Animation to plot the analytical and the numerical solution."""

    fig = plt.figure()
    (line_exact,) = plt.plot([], [], color="blue", label="Analytical solution")
    (line_approx,) = plt.plot([], [], color="orange", label="Approximate solution")
    plt.legend()
    plt.xlim(x[0], x[-1])
    plt.ylim(ymin, ymax)

    dt = final_time / nb_time

    def animate(i):
        t = i * dt
        analytical = exact(x, t)
        approx = sol[i]
        line_exact.set_data(x, analytical)
        line_approx.set_data(x, approx)
        return (
            line_exact,
            line_approx,
        )

    # Keep a reference alive so the animation is not garbage-collected.
    _ani = animation.FuncAnimation(
        fig,
        animate,
        frames=np.arange(0, nb_time + 1),
        interval=1,
        blit=True,
        repeat=True,
    )
    plt.show()


def _gravity(conductor):
    """Return a constant self-weight force per unit length."""

    def force(x, t, y, v):
        return -_GRAVITY * conductor.mass * np.ones_like(x)

    return force


def _stored_positions(parameters, length):
    """Physical abscissae the solver stores, i.e. ``parameters.los`` scaled up.

    The solve runs on the ``ns`` nodes but only ``los`` is recorded, so a
    manufactured solution has to be evaluated there and not on the node grid.
    """
    return np.asarray(parameters.los) * length


def _manufactured(exact, parameters, length):
    """The closed form sampled at the stored positions and output times."""
    stored = _stored_positions(parameters, length)
    times = parameters.time_vector_output()
    return np.array([exact(stored, t) for t in times])


# --------------------------------------------------------------------------
# manufactured solutions, one per (model, curvature) combination
# --------------------------------------------------------------------------


def test_manufactured_constant_approx_static_bc(plot=False):
    """Constant model, approximate curvature, time-independent boundary rows.

    Manufactured solution ``y = cos(t) * x**2 * (x - L)**2 + 1``, which is a
    quartic in space, so the fourth difference is exact on it.
    """
    nb_space = 300
    dt = 1e-4
    final_time = 1.0
    mass = 14.3
    tension = 125.78
    ei = 1484.75
    lspan = 3.0

    x = np.linspace(0, lspan, nb_space)

    def f(t):
        return np.cos(t)

    def exact(x, t):
        return f(t) * x**2 * (x - lspan) ** 2 + 1

    def exact_time_derivative(x, t):
        return -np.sin(t) * x**2 * (x - lspan) ** 2

    def force(x, t, y, v):
        return (
            -mass * np.cos(t) * x**2 * (x - lspan) ** 2
            + ei * 24.0 * f(t)
            - tension * f(t) * (12 * x**2 - 12 * lspan * x + 2 * lspan**2)
        )

    left = [[1, 0, 0, 1], [0, 1, 0, 0]]
    right = [[1, 0, 0, 1], [0, 1, 0, 0]]
    bc = BoundaryCondition(4, left, right)

    conductor = Conductor(mass=mass, ei_max=ei)
    span = Span(length=lspan, tension=tension, boundary_conditions=bc)
    parameters = simulation.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-3, los=nb_space
    )

    res = solve_dynamic(
        conductor,
        span,
        parameters,
        model=BendingModel.CONSTANT,
        ei=ei,
        force=force,
        approx_curvature=True,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
    )
    y = res["y"].values

    if plot:
        _plot_animation(
            _stored_positions(parameters, lspan),
            exact,
            y,
            -7,
            7,
            parameters.nr,
            final_time,
        )

    assert np.allclose(
        _manufactured(exact, parameters, lspan), y, atol=1.0e-06, rtol=1.0e-02
    )


def test_manufactured_constant_approx_dynamic_bc(plot=False):
    """Constant model, approximate curvature, time-dependent boundary rows."""
    nb_space = 400
    dt = 1e-4
    final_time = 1.2
    mass = 1.45
    tension = 12.36
    ei = 147.89
    lmin = 0.0
    lmax = 4.0
    lspan = lmax - lmin

    x = np.linspace(lmin, lmax, nb_space)

    def exact(x, t):
        return np.cosh(x - 2) * np.sin(2 * np.pi * t)

    def exact_time_space_derivative(x, t):
        return 2 * np.pi * np.sinh(x - 2) * np.cos(2 * np.pi * t)

    def exact_time_derivative(x, t):
        return 2 * np.pi * np.cosh(x - 2) * np.cos(2 * np.pi * t)

    def force(x, t, y, v):
        return (
            -4 * np.pi**2 * mass * exact(x, t)
            + ei * exact(x, t)
            - tension * exact(x, t)
        )

    left = [
        [1, 0, 0, exact_time_derivative(lmin, 0)],
        [0, 1, 0, exact_time_space_derivative(lmin, 0)],
    ]
    right = [
        [1, 0, 0, exact_time_derivative(lmax, 0)],
        [0, 1, 0, exact_time_space_derivative(lmax, 0)],
    ]
    dynamic_values = [
        exact_time_derivative,
        exact_time_space_derivative,
        exact_time_space_derivative,
        exact_time_derivative,
    ]
    bc = BoundaryCondition(4, left, right, dynamic_values)

    conductor = Conductor(mass=mass, ei_max=ei)
    span = Span(length=lspan, tension=tension, boundary_conditions=bc)
    parameters = simulation.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-3, los=nb_space
    )

    res = solve_dynamic(
        conductor,
        span,
        parameters,
        model=BendingModel.CONSTANT,
        ei=ei,
        force=force,
        approx_curvature=True,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
    )
    y = res["y"].values

    if plot:
        _plot_animation(
            _stored_positions(parameters, lspan),
            exact,
            y,
            -5,
            5,
            parameters.nr,
            final_time,
        )

    assert np.allclose(
        _manufactured(exact, parameters, lspan), y, atol=1.0e-01, rtol=1.0e-06
    )


def test_manufactured_constant_exact(plot=False):
    """Constant model, exact geometric curvature."""
    nb_space = 100
    dt = 1e-5
    final_time = 0.1
    mass = 9.8
    tension = 256.12
    ei = 2698.23
    lmin = 0.0
    lmax = 2.0
    lspan = lmax - lmin
    x = np.linspace(lmin, lmax, nb_space)

    def force(x, t, y, v):
        return (
            mass * np.cosh(x + t)
            + ei
            * (
                -2 / np.cosh(x + t) ** 2
                + 6.0 * np.sinh(x + t) ** 2 / np.cosh(x + t) ** 4
            )
            - tension * np.cosh(x + t)
        )

    def exact(x, t):
        return np.cosh(x + t)

    def exact_time_derivative(x, t):
        return np.sinh(x + t)

    def exact_time_space_derivative(x, t):
        return np.cosh(x + t)

    left = [
        [1, 0, 0, exact_time_derivative(lmin, 0)],
        [0, 1, 0, exact_time_space_derivative(lmin, 0)],
    ]
    right = [
        [1, 0, 0, exact_time_derivative(lmax, 0)],
        [0, 1, 0, exact_time_space_derivative(lmax, 0)],
    ]
    dynamic_values = [
        exact_time_derivative,
        exact_time_space_derivative,
        exact_time_space_derivative,
        exact_time_derivative,
    ]
    bc = BoundaryCondition(4, left, right, dynamic_values)

    conductor = Conductor(mass=mass, ei_max=ei)
    span = Span(length=lspan, tension=tension, boundary_conditions=bc)
    parameters = simulation.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-3, los=nb_space
    )

    res = solve_dynamic(
        conductor,
        span,
        parameters,
        model=BendingModel.CONSTANT,
        ei=ei,
        force=force,
        approx_curvature=False,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
    )
    y = res["y"].values

    if plot:
        _plot_animation(
            _stored_positions(parameters, lspan),
            exact,
            y,
            1,
            7,
            parameters.nr,
            final_time,
        )

    assert np.allclose(
        _manufactured(exact, parameters, lspan), y, atol=1.0e-06, rtol=1.0e-03
    )


def test_manufactured_varying_approx(plot=False):
    """Varying model, approximate curvature.

    The curvature stays far below ``chi0``, where the Bouc-Wen tangent is
    ``ei_max``, so the manufactured force is the one of a constant ``ei_max``
    beam.
    """
    nb_space = 100
    dt = 1e-5
    final_time = 0.1
    mass = 12.8
    tension = 132.74
    ei_max = 1789.36
    ei_min = 1258.32
    chi0 = 125.78
    lmin = 0.0
    lmax = 2.0
    lspan = lmax - lmin
    x = np.linspace(lmin, lmax, nb_space)

    def force(x, t, y, v):
        return (
            mass * np.cosh(x + t) + ei_max * np.cosh(x + t) - tension * np.cosh(x + t)
        )

    def exact(x, t):
        return np.cosh(x + t)

    def exact_time_derivative(x, t):
        return np.sinh(x + t)

    def exact_time_space_derivative(x, t):
        return np.cosh(x + t)

    left = [
        [1, 0, 0, exact_time_derivative(lmin, 0)],
        [0, 1, 0, exact_time_space_derivative(lmin, 0)],
    ]
    right = [
        [1, 0, 0, exact_time_derivative(lmax, 0)],
        [0, 1, 0, exact_time_space_derivative(lmax, 0)],
    ]
    dynamic_values = [
        exact_time_derivative,
        exact_time_space_derivative,
        exact_time_space_derivative,
        exact_time_derivative,
    ]
    bc = BoundaryCondition(4, left, right, dynamic_values)

    conductor = Conductor(
        mass=mass, ei_min=ei_min, ei_max=ei_max, beta_flexion=chi0 / tension
    )
    span = Span(length=lspan, tension=tension, boundary_conditions=bc)
    parameters = simulation.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-3, los=nb_space
    )

    res = solve_dynamic(
        conductor,
        span,
        parameters,
        model=BendingModel.VARYING,
        force=force,
        approx_curvature=True,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
    )
    y = res["y"].values

    if plot:
        _plot_animation(
            _stored_positions(parameters, lspan),
            exact,
            y,
            1,
            7,
            parameters.nr,
            final_time,
        )

    assert np.allclose(
        _manufactured(exact, parameters, lspan), y, atol=1.0e-06, rtol=1.0e-01
    )


def test_manufactured_varying_exact(plot=False):
    """Varying model, exact geometric curvature."""
    nb_space = 100
    dt = 1e-5
    final_time = 0.1
    mass = 12.8
    tension = 179.15
    ei_max = 2489.46
    ei_min = 1487.13
    chi0 = 147.12
    lmin = 0.0
    lmax = 2.0
    lspan = lmax - lmin
    x = np.linspace(lmin, lmax, nb_space)

    def force(x, t, y, v):
        return (
            mass * np.cosh(x + t)
            + ei_max
            * (
                -2 / np.cosh(x + t) ** 2
                + 6.0 * np.sinh(x + t) ** 2 / np.cosh(x + t) ** 4
            )
            - tension * np.cosh(x + t)
        )

    def exact(x, t):
        return np.cosh(x + t)

    def exact_time_derivative(x, t):
        return np.sinh(x + t)

    def exact_time_space_derivative(x, t):
        return np.cosh(x + t)

    left = [
        [1, 0, 0, exact_time_derivative(lmin, 0)],
        [0, 1, 0, exact_time_space_derivative(lmin, 0)],
    ]
    right = [
        [1, 0, 0, exact_time_derivative(lmax, 0)],
        [0, 1, 0, exact_time_space_derivative(lmax, 0)],
    ]
    dynamic_values = [
        exact_time_derivative,
        exact_time_space_derivative,
        exact_time_space_derivative,
        exact_time_derivative,
    ]
    bc = BoundaryCondition(4, left, right, dynamic_values)

    conductor = Conductor(
        mass=mass, ei_min=ei_min, ei_max=ei_max, beta_flexion=chi0 / tension
    )
    span = Span(length=lspan, tension=tension, boundary_conditions=bc)
    parameters = simulation.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-3, los=nb_space
    )

    res = solve_dynamic(
        conductor,
        span,
        parameters,
        model=BendingModel.VARYING,
        force=force,
        approx_curvature=False,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
    )
    y = res["y"].values

    if plot:
        _plot_animation(
            _stored_positions(parameters, lspan),
            exact,
            y,
            1,
            7,
            parameters.nr,
            final_time,
        )

    assert np.allclose(
        _manufactured(exact, parameters, lspan), y, atol=1.0e-06, rtol=1.0e-03
    )


# --------------------------------------------------------------------------
# properties of the assembly
# --------------------------------------------------------------------------


def test_static_state_is_a_fixed_point():
    """Started from the static shape at rest, the solver must not move.

    Sharpest check of the whole assembly: it only holds if the dynamic operators
    are the ones shape.py solves to zero.
    """
    for model, approx in CASES:
        parameters = simulation.Parameters(ns=101, t0=0.0, tf=0.5, dt=0.005, dr=0.05)
        res = solve_dynamic(
            CONDUCTOR,
            BRETELLE,
            parameters,
            model=model,
            force=_gravity(CONDUCTOR),
            approx_curvature=approx,
        )
        y = res["y"].values
        v = res["v"].values
        assert np.all(np.isfinite(y)), (model, approx)
        drift = np.abs(y - y[0, :]).max()
        assert drift < 1e-9 * np.abs(y[0, :]).max(), (model, approx, drift)
        assert np.abs(v).max() < 1e-9, (model, approx, np.abs(v).max())


def test_linear_case_needs_one_solve_per_step():
    """Constant model plus approximate curvature is linear: no iteration."""
    parameters = simulation.Parameters(ns=101, t0=0.0, tf=0.2, dt=0.002, dr=0.02)
    res = solve_dynamic(
        CONDUCTOR,
        BRETELLE,
        parameters,
        model=BendingModel.CONSTANT,
        approx_curvature=True,
        initial_velocity=np.zeros(parameters.ns),
    )
    assert np.nanmax(res["n_iter"].values[1:]) == 1


def test_linear_case_matches_slenderpy():
    """The linear case must reproduce BeamConst.solve_dynamic to round-off.

    Same Crank-Nicolson scheme on both sides, so any difference in the assembly
    of A, B, the boundary rows or the output cadence shows up here.
    """
    for span, zeta in [(SPAN, 0.0), (BRETELLE, 0.0), (SPAN, 0.03)]:
        ns = 101
        x = np.linspace(0.0, span.length, ns)
        y0 = 0.2 * np.sin(np.pi * x / span.length)
        v0 = np.zeros(ns)
        f0 = 0.5 / span.length * np.sqrt(span.tension / CONDUCTOR.mass)
        parameters = simulation.Parameters(
            ns=ns, t0=0.0, tf=2.0 / f0, dt=1.0 / (f0 * 200), dr=1.0 / (f0 * 20)
        )
        force = _gravity(CONDUCTOR)
        mine = solve_dynamic(
            CONDUCTOR,
            span,
            parameters,
            model=BendingModel.CONSTANT,
            ei=CONDUCTOR.ei_max,
            force=force,
            approx_curvature=True,
            initial_position=y0,
            initial_velocity=v0,
            zeta=zeta,
        )
        beam = BeamConst(
            length=span.length,
            boundary_conditions=span.boundary_conditions,
            tension=span.tension,
            mass=CONDUCTOR.mass,
            ei=CONDUCTOR.ei_max,
        )
        ref = beam.solve_dynamic(parameters, y0, v0, force, True, zeta=zeta)

        # slenderpy stores every node, this solver stores parameters.los
        nodes = np.linspace(0.0, 1.0, ns)
        scale = np.abs(mine["y"].values).max()
        for name, rtol in [("y", 1e-09), ("v", 1e-06)]:
            sampled = np.array(
                [np.interp(parameters.los, nodes, row) for row in ref[name].values]
            )
            assert np.abs(mine[name].values - sampled).max() < rtol * scale, name


def test_free_vibration_period_matches_theory():
    """First mode of a hinged tensioned beam, constant stiffness.

    Reference: f = f0 * sqrt(1 + ei/(tension*length**2) * pi**2), the n=1 case of
    slenderpy's natural_frequencies_hinged.
    """
    ei = CONDUCTOR.ei_max
    f0 = 0.5 / SPAN.length * np.sqrt(SPAN.tension / CONDUCTOR.mass)
    expected = f0 * np.sqrt(1.0 + ei / (SPAN.tension * SPAN.length**2) * np.pi**2)

    ns = 201
    x = np.linspace(0.0, SPAN.length, ns)
    y0 = 0.5 * np.sin(np.pi * x / SPAN.length)
    periods = 4.0
    parameters = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=periods / expected,
        dt=1.0 / (expected * 400),
        dr=1.0 / (expected * 400),
        los=[0.5],
    )
    res = solve_dynamic(
        CONDUCTOR,
        SPAN,
        parameters,
        model=BendingModel.CONSTANT,
        ei=ei,
        approx_curvature=True,
        initial_position=y0,
    )

    # period from the sign changes of the mid-span displacement
    mid = res["y"].values[:, 0]
    time = np.asarray(res.lot())
    crossings = np.nonzero(np.diff(np.sign(mid)))[0]
    zeros = time[crossings] - mid[crossings] * (
        time[crossings + 1] - time[crossings]
    ) / (mid[crossings + 1] - mid[crossings])
    measured = 1.0 / (2.0 * np.mean(np.diff(zeros)))
    assert abs(measured - expected) / expected < 2e-3, (measured, expected)


def test_undamped_run_conserves_amplitude():
    """Crank-Nicolson is non-dissipative: the amplitude must not decay."""
    ns = 201
    x = np.linspace(0.0, SPAN.length, ns)
    y0 = 0.5 * np.sin(np.pi * x / SPAN.length)
    f0 = 0.5 / SPAN.length * np.sqrt(SPAN.tension / CONDUCTOR.mass)
    parameters = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=10.0 / f0,
        dt=1.0 / (f0 * 200),
        dr=1.0 / (f0 * 200),
        los=[0.5],
    )
    res = solve_dynamic(
        CONDUCTOR,
        SPAN,
        parameters,
        model=BendingModel.CONSTANT,
        approx_curvature=True,
        initial_position=y0,
    )
    mid = np.abs(res["y"].values[:, 0])
    first = mid[: len(mid) // 5].max()
    last = mid[-len(mid) // 5 :].max()
    assert abs(last - first) / first < 5e-3, (first, last)


def test_damping_follows_the_expected_envelope():
    """With zeta > 0 the first mode decays as exp(-2*pi*zeta*f0*t)."""
    ns = 201
    zeta = 0.02
    x = np.linspace(0.0, SPAN.length, ns)
    y0 = 0.5 * np.sin(np.pi * x / SPAN.length)
    f0 = 0.5 / SPAN.length * np.sqrt(SPAN.tension / CONDUCTOR.mass)
    parameters = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=6.0 / f0,
        dt=1.0 / (f0 * 400),
        dr=1.0 / (f0 * 400),
        los=[0.5],
    )
    res = solve_dynamic(
        CONDUCTOR,
        SPAN,
        parameters,
        model=BendingModel.CONSTANT,
        approx_curvature=True,
        initial_position=y0,
        zeta=zeta,
    )
    time = np.asarray(res.lot())
    mid = res["y"].values[:, 0]

    # envelope sampled at the maxima of |mid|, fitted in log space
    peaks = (
        1
        + np.nonzero(
            (np.abs(mid[1:-1]) > np.abs(mid[:-2]))
            & (np.abs(mid[1:-1]) > np.abs(mid[2:]))
        )[0]
    )
    rate = -np.polyfit(time[peaks], np.log(np.abs(mid[peaks])), 1)[0]
    assert abs(rate - 2 * np.pi * zeta * f0) / (2 * np.pi * zeta * f0) < 0.05, rate


def test_hysteresis_loop_stays_within_the_static_envelope():
    """Cyclic loading of the varying model: eta bounded, loop inside the envelope."""
    ns = 101
    law = bending.create(CONDUCTOR, BRETELLE, BendingModel.VARYING)

    f0 = 0.5 / BRETELLE.length * np.sqrt(BRETELLE.tension / CONDUCTOR.mass)
    amplitude = 4.0 * _GRAVITY * CONDUCTOR.mass

    def force(x, t, y, v):
        return amplitude * np.sin(2.0 * np.pi * f0 * t) * np.ones_like(x)

    parameters = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=3.0 / f0,
        dt=1.0 / (f0 * 500),
        dr=1.0 / (f0 * 500),
        los=[0.25, 0.5, 0.75],
    )
    res = solve_dynamic(
        CONDUCTOR,
        BRETELLE,
        parameters,
        model=BendingModel.VARYING,
        force=force,
        approx_curvature=True,
        initial_position=np.zeros(ns),
    )
    curv = res["c"].values
    mom = res["M"].values
    eta = res["eta"].values
    assert np.all(np.isfinite(mom))
    assert np.abs(eta).max() <= 1.0 + 1e-09, np.abs(eta).max()

    # the hysteretic part of M can never exceed the saturated plateau
    hysteretic = mom - CONDUCTOR.ei_min * curv
    assert np.abs(hysteretic).max() <= law.plateau * (1.0 + 1e-09)

    # and the reached moments stay at or below the static envelope
    assert np.all(np.abs(mom) <= np.abs(law.moment(curv)) + law.plateau)


def test_non_convergence_returns_nan_and_no_state():
    """A step that cannot converge stops the run and leaves nan behind."""
    parameters = simulation.Parameters(ns=101, t0=0.0, tf=0.5, dt=0.005, dr=0.05)
    x = np.linspace(0.0, BRETELLE.length, parameters.ns)
    res = solve_dynamic(
        CONDUCTOR,
        BRETELLE,
        parameters,
        model=BendingModel.VARYING,
        force=_gravity(CONDUCTOR),
        approx_curvature=False,
        initial_position=np.zeros_like(x),
        initial_velocity=np.zeros_like(x),
        max_iter=1,
        tol=1e-30,
    )
    assert res.state is None
    assert np.isnan(res["y"].values[-1, :]).all()


def test_output_layout():
    """Stored variables, shapes and time vector."""
    parameters = simulation.Parameters(
        ns=51, t0=0.0, tf=0.4, dt=0.004, dr=0.04, los=[0.2, 0.5, 0.8]
    )
    res = solve_dynamic(CONDUCTOR, BRETELLE, parameters, force=_gravity(CONDUCTOR))
    assert set(res.lov()) == {"y", "v", "c", "M", "eta", "n_iter"}
    assert res.los() == parameters.los
    assert res["y"].values.shape == (parameters.nr + 1, len(parameters.los))
    assert res["n_iter"].values.shape == (parameters.nr + 1,)
    assert len(res.lot()) == parameters.nr + 1
    assert res.compute_time is not None

    # the final state keeps the full discretisation whatever los holds
    assert res.state is not None
    for name in ("y", "v", "c", "M", "eta"):
        assert res.state[name].shape == (parameters.ns,), name


def test_start_time_offset_is_honoured():
    """t0 != 0 must not change the time step, unlike tf/nt."""
    common = dict(ns=51, dt=0.004, dr=0.04)
    force = _gravity(CONDUCTOR)
    res0 = solve_dynamic(
        CONDUCTOR,
        BRETELLE,
        simulation.Parameters(t0=0.0, tf=0.4, **common),
        force=force,
    )
    res1 = solve_dynamic(
        CONDUCTOR,
        BRETELLE,
        simulation.Parameters(t0=10.0, tf=10.4, **common),
        force=force,
    )
    assert np.allclose(res0["y"].values, res1["y"].values, atol=1e-12)
