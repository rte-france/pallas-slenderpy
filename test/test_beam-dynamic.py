"""Tests for the dynamic beam solver of slenderpy.beam.dynamic.

Two families: manufactured solutions, which check the discretisation against a
closed form, and property tests, which check the assembly itself (fixed point,
conservation, hysteresis bounds, output layout).
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pytest

from slenderpy import simulation
from slenderpy._constant import _GRAVITY
from slenderpy.beam import bending
from slenderpy.beam.bending import BendingModel
from slenderpy.beam.dynamic import solve_dynamic
from slenderpy.beam.static import shape
from slenderpy.boundary_condition import BoundaryCondition, clamped, hinged
from slenderpy.components import Conductor, Span
from slenderpy.force.core import Gravity, PointExcitation

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
    return Gravity(conductor.mass)


def _l2_gap(first, second):
    """Root-mean-square gap between two recorded fields."""
    return float(np.sqrt(np.mean((first - second) ** 2)))


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

    def force(x, t, y, z, vy, vz):
        return 0.0, (
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
    y = res["z"].values

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

    def force(x, t, y, z, vy, vz):
        return 0.0, (
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
    y = res["z"].values

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

    def force(x, t, y, z, vy, vz):
        return 0.0, (
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
    y = res["z"].values

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

    def force(x, t, y, z, vy, vz):
        return 0.0, (
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
    y = res["z"].values

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

    def force(x, t, y, z, vy, vz):
        return 0.0, (
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
    y = res["z"].values

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
        y = res["z"].values
        v = res["vz"].values
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
    mid = res["z"].values[:, 0]
    time = np.asarray(res.lot())
    crossings = np.nonzero(np.diff(np.sign(mid)))[0]
    zeros = time[crossings] - mid[crossings] * (
        time[crossings + 1] - time[crossings]
    ) / (mid[crossings + 1] - mid[crossings])
    measured = 1.0 / (2.0 * np.mean(np.diff(zeros)))
    assert abs(measured - expected) / expected < 2e-3, (measured, expected)


def test_undamped_run_conserves_amplitude():
    """A resolved mode keeps its amplitude: the hht damping is negligible on it."""
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
    mid = np.abs(res["z"].values[:, 0])
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
    mid = res["z"].values[:, 0]

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

    def force(x, t, y, z, vy, vz):
        return 0.0, amplitude * np.sin(2.0 * np.pi * f0 * t) * np.ones_like(x)

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
    curv = res["curvature"].values
    mom = res["moment"].values
    eta = res["eta"].values
    assert np.all(np.isfinite(mom))
    assert np.abs(eta).max() <= 1.0 + 1e-09, np.abs(eta).max()

    # the hysteretic part of M can never exceed the saturated plateau
    hysteretic = mom - CONDUCTOR.ei_min * curv
    assert np.abs(hysteretic).max() <= law.plateau * (1.0 + 1e-09)

    # and the reached moments stay at or below the static envelope
    assert np.all(np.abs(mom) <= np.abs(law.moment(curv)) + law.plateau)


@pytest.mark.parametrize("perturbation", [1e-13, 3e-13])
def test_hysteresis_reversals_converge_whatever_the_round_off(perturbation):
    """A reversal of the curvature rate is a kink of the Bouc-Wen law.

    Newton straddling it used to stall, and round-off decided whether it got
    through: these two perturbations of the forcing, at the 1e-13 level, used
    to fail even with max_iter = 2000 (the CI failure of
    test_hysteresis_loop_stays_within_the_static_envelope).
    """
    ns = 101
    f0 = 0.5 / BRETELLE.length * np.sqrt(BRETELLE.tension / CONDUCTOR.mass)
    amplitude = (1.0 + perturbation) * 4.0 * _GRAVITY * CONDUCTOR.mass

    def force(x, t, y, z, vy, vz):
        return 0.0, amplitude * np.sin(2.0 * np.pi * f0 * t) * np.ones_like(x)

    parameters = simulation.Parameters(
        ns=ns, t0=0.0, tf=3.0 / f0, dt=1.0 / (f0 * 500), dr=1.0 / (f0 * 500)
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
    assert np.all(np.isfinite(res["moment"].values))
    assert np.abs(res["eta"].values).max() <= 1.0 + 1e-09


@pytest.mark.parametrize("approx_curvature", [True, False])
def test_hysteresis_lives_at_the_clamped_ends(approx_curvature):
    """Cyclic loading of the varying model: eta moves at the end nodes too.

    The end nodes are where the bending moment of a clamped span matters most,
    and the curvature increment feeding eta has to reach them. Starting from
    rest, eta is exactly zero everywhere, so a non-zero eta at the first and
    last node is proof the hysteresis was integrated there as well.
    """
    ns = 101
    f0 = 0.5 / BRETELLE.length * np.sqrt(BRETELLE.tension / CONDUCTOR.mass)
    amplitude = 4.0 * _GRAVITY * CONDUCTOR.mass

    def force(x, t, y, z, vy, vz):
        return 0.0, amplitude * np.sin(2.0 * np.pi * f0 * t) * np.ones_like(x)

    parameters = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=1.0 / f0,
        dt=1.0 / (f0 * 500),
        dr=1.0 / (f0 * 50),
        los=[0.25, 0.5, 0.75],
    )
    res = solve_dynamic(
        CONDUCTOR,
        BRETELLE,
        parameters,
        model=BendingModel.VARYING,
        force=force,
        approx_curvature=approx_curvature,
        initial_position=np.zeros(ns),
        initial_velocity=np.zeros(ns),
    )

    eta = res.state["eta"]
    assert np.all(eta[[0, -1]] != 0.0), eta[[0, -1]]
    assert np.abs(eta).max() <= 1.0 + 1e-09


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
        energy=True,
    )
    assert res.state is None
    assert np.isnan(res["z"].values[-1, :]).all()
    assert np.isnan(res["kinetic_energy"].values[-1])


def test_output_layout():
    """Stored variables, shapes and time vector."""
    parameters = simulation.Parameters(
        ns=51, t0=0.0, tf=0.4, dt=0.004, dr=0.04, los=[0.2, 0.5, 0.8]
    )
    res = solve_dynamic(CONDUCTOR, BRETELLE, parameters, force=_gravity(CONDUCTOR))
    assert set(res.lov()) == {"z", "vz", "curvature", "moment", "eta", "n_iter"}
    assert res.los() == parameters.los
    assert res["z"].values.shape == (parameters.nr + 1, len(parameters.los))
    assert res["n_iter"].values.shape == (parameters.nr + 1,)
    assert len(res.lot()) == parameters.nr + 1
    assert res.compute_time is not None

    # the final state keeps the full discretisation whatever los holds
    assert res.state is not None
    for name in ("z", "vz", "curvature", "moment", "eta"):
        assert res.state[name].shape == (parameters.ns,), name


def test_supports_can_be_stored():
    """los may hold 0 and 1: the clamped ends stay put, the moment is recorded."""
    parameters = simulation.Parameters(
        ns=51, t0=0.0, tf=0.4, dt=0.004, dr=0.04, los=[0.0, 0.5, 1.0]
    )
    x = np.linspace(0.0, BRETELLE.length, parameters.ns)
    # zero value and slope at both ends, compatible with clamped()
    y0 = 0.01 * np.sin(np.pi * x / BRETELLE.length) ** 2
    res = solve_dynamic(
        CONDUCTOR,
        BRETELLE,
        parameters,
        force=_gravity(CONDUCTOR),
        initial_position=y0,
    )
    z = res["z"].values
    assert np.abs(z[:, [0, -1]]).max() < 1e-10
    assert np.isfinite(res["moment"].values[:, [0, -1]]).all()
    # the mid-span moves, so the run is not trivially at rest
    assert np.ptp(z[:, 1]) > 0.0


def test_scalar_force_is_broadcast():
    """A force may return scalars."""

    def weight(x, t, y, z, vy, vz):
        return 0.0, -9.81 * CONDUCTOR.mass

    parameters = simulation.Parameters(ns=51, t0=0.0, tf=0.2, dt=0.004, dr=0.04)
    res = solve_dynamic(CONDUCTOR, BRETELLE, parameters, force=weight)
    assert np.all(np.isfinite(res["z"].values))
    assert res["z"].values[0, len(parameters.los) // 2] < 0.0


def test_sum_of_forces_drives_the_beam():
    parameters = simulation.Parameters(ns=51, t0=0.0, tf=0.4, dt=0.004, dr=0.04)
    gravity = Gravity(CONDUCTOR.mass)
    excitation = PointExcitation(
        frequency=5.0, amplitude=10.0, position=0.5 * BRETELLE.length
    )
    alone = solve_dynamic(CONDUCTOR, BRETELLE, parameters, force=gravity)
    both = solve_dynamic(CONDUCTOR, BRETELLE, parameters, force=gravity + excitation)
    assert np.all(np.isfinite(both["z"].values))
    assert np.abs(both["z"].values - alone["z"].values).max() > 1e-6


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
    assert np.allclose(res0["z"].values, res1["z"].values, atol=1e-12)
    assert res1.lot() == pytest.approx((10.0 + 0.04 * np.arange(11)).tolist())


# a 4x refinement has to cut the error at least in half, i.e. an observed order
# of 1/2 or better. Half order is a floor rather than the expectation: the
# constant model reaches 1.6 on the position, but the bouc-wen switching is not
# smooth and the hysteresis path only reaches 1.0 here. The worst case clears
# this by 1.6x, so the check does not turn on the blas or the python build.
_MIN_CONVERGENCE_RATIO = 2.0


def test_converging_in_time():
    """Refining the time step converges, for every (model, curvature) case.

    Three runs per case, at ``dt``, ``dt/4`` and ``dt/16``, the finest standing
    as the reference. A ratio of errors rather than an absolute error: it does
    not depend on the scale of the field, and it fails on a scheme that stops
    converging rather than on one that is merely inaccurate. The span is 4x and
    not 2x on purpose, a 2x refinement leaves the ratio too close to 1 to tell
    convergence from noise on the hysteretic cases.
    """
    force = _gravity(CONDUCTOR)
    ns = 101
    x = np.linspace(0.0, BRETELLE.length, ns)
    dt = 0.002

    for model, approx in CASES:
        ei = CONDUCTOR.ei_min if model == BendingModel.CONSTANT else None
        y0 = shape.solve(
            CONDUCTOR,
            BRETELLE,
            np.zeros_like(x) + force(x, 0.0, 0.0, 0.0, 0.0, 0.0)[1],
            ns,
            model=model,
            ei=ei,
            approx_curvature=approx,
        )

        solutions = {}
        for refinement in (1, 4, 16):
            parameters = simulation.Parameters(
                ns=ns, t0=0.0, tf=0.1, dt=dt / refinement, dr=0.02, los=ns
            )
            solutions[refinement] = solve_dynamic(
                CONDUCTOR,
                BRETELLE,
                parameters,
                model=model,
                ei=ei,
                approx_curvature=approx,
                initial_position=1.02 * y0,
                force=force,
            )

        # eta is identically zero for the constant model, which would leave the
        # ratio at 0/0
        reference = solutions[16]
        fields = ("z", "vz", "eta") if model == BendingModel.VARYING else ("z", "vz")
        for field in fields:
            coarse = _l2_gap(solutions[1][field], reference[field])
            fine = _l2_gap(solutions[4][field], reference[field])
            assert coarse / fine > _MIN_CONVERGENCE_RATIO, (
                model,
                approx,
                field,
                coarse / fine,
            )


@pytest.mark.parametrize("model", [BendingModel.CONSTANT, BendingModel.VARYING])
def test_fine_grid_converges(model):
    """1 cm steps on a 10 m span: the step residual sits at its round-off floor.

    ``dt * K @ y`` carries a round-off of about ``eps * dt * ||K|| * |y|``, with
    ``||K||`` growing like ``EI / ds**4``; the Newton threshold must not ask
    for less, or the very first step fails.
    """
    span = Span(length=10.0, tension=500.0, boundary_conditions=clamped())
    parameters = simulation.Parameters(ns=1001, t0=0.0, tf=0.02, dt=2e-3, dr=2e-3)
    res = solve_dynamic(
        CONDUCTOR,
        span,
        parameters,
        model=model,
        force=_gravity(CONDUCTOR),
        approx_curvature=False,
    )
    assert np.all(np.isfinite(res["z"].values))


def test_order_2_boundary_conditions_raise():
    conductor = Conductor(mass=1.0, ei_max=10.0)
    span = Span(length=1.0, tension=100.0, boundary_conditions=BoundaryCondition(2))
    parameters = simulation.Parameters(ns=11, tf=1.0, dt=0.01)
    with pytest.raises(ValueError, match="order-4"):
        solve_dynamic(conductor, span, parameters)


# --------------------------------------------------------------------------
# energy balance
# --------------------------------------------------------------------------

# reference span of the energy tests, a line span with no support level
# difference: length (m) and tension (N)
LINE = (440.0, 31000.0)
ENERGY = (
    "kinetic_energy",
    "tension_energy",
    "bending_energy",
    "external_work",
    "damping_work",
    "hysteresis_work",
    "curvature_work",
    "numerical_work",
    "boundary_work",
)


def _line_span(bc):
    return Span(length=LINE[0], tension=LINE[1], boundary_conditions=bc)


def _line_frequency():
    """First taut-string frequency of the line span."""
    return 0.5 / LINE[0] * np.sqrt(LINE[1] / CONDUCTOR.mass)


def _line_parameters(ns, periods=4.0, per_period=400, **kwargs):
    """Time stepping on the first taut-string period of the line span."""
    f0 = _line_frequency()
    kwargs.setdefault("dr", 1.0 / (f0 * per_period))
    return simulation.Parameters(
        ns=ns, tf=periods / f0, dt=1.0 / (f0 * per_period), **kwargs
    )


def _free_shape(length, ns, amplitude=0.5):
    """Initial deflection meeting hinged and clamped conditions alike."""
    x = np.linspace(0.0, length, ns)
    return amplitude * np.sin(np.pi * x / length) ** 3


def _balance(res):
    """The quantity the energy balance keeps constant."""
    v = {name: res[name].values for name in ENERGY}
    return (
        v["kinetic_energy"]
        + v["tension_energy"]
        + v["bending_energy"]
        + v["damping_work"]
        + v["hysteresis_work"]
        + v["curvature_work"]
        + v["numerical_work"]
        - v["external_work"]
        - v["boundary_work"]
    )


def _drift(res):
    """Largest departure of the balance from its initial value."""
    balance = _balance(res)
    return np.abs(balance - balance[0]).max()


def _energy_scale(res):
    return max(np.nanmax(np.abs(res[name].values)) for name in ENERGY)


def _free_vibration(bc, ns, parameters=None, **kwargs):
    """Free vibration of the line span from the reference shape."""
    return solve_dynamic(
        CONDUCTOR,
        _line_span(bc),
        _line_parameters(ns) if parameters is None else parameters,
        initial_position=_free_shape(LINE[0], ns),
        energy=True,
        **kwargs,
    )


def test_energy_fields_layout():
    parameters = simulation.Parameters(ns=51, t0=0.0, tf=0.4, dt=0.004, dr=0.04)
    plain = solve_dynamic(CONDUCTOR, BRETELLE, parameters, force=_gravity(CONDUCTOR))
    assert plain.lov() == ["z", "vz", "curvature", "moment", "eta", "n_iter"]

    res = solve_dynamic(
        CONDUCTOR, BRETELLE, parameters, force=_gravity(CONDUCTOR), energy=True
    )
    assert res.lov() == plain.lov() + list(ENERGY)
    for name in ENERGY:
        assert res.lov_dims[name] == 1, name
        assert res[name].values.shape == (parameters.nr + 1,), name
    for name in ENERGY[3:]:
        assert res[name].values[0] == 0.0, name


def test_free_vibration_clamped_conserves_energy():
    res = _free_vibration(clamped(), 201)
    scale = _energy_scale(res)
    assert _drift(res) <= 1e-10 * scale
    assert np.abs(res["boundary_work"].values).max() <= 1e-13 * scale


def test_hinged_boundary_work_converges_at_third_order():
    boundary = []
    for ns in (201, 401):
        res = _free_vibration(hinged(), ns)
        scale = _energy_scale(res)
        assert _drift(res) <= 1e-10 * scale, ns
        boundary.append(np.abs(res["boundary_work"].values).max() / scale)
    # measured 3.9e-05 then 4.9e-06: the term is exercised, and falls as ds**3
    assert boundary[0] > 1e-6
    assert boundary[0] / boundary[1] > 6.0


def test_curvature_work():
    ns = 201
    parameters = simulation.Parameters(ns=ns, tf=0.5, dt=1e-4, dr=1e-3)
    initial = _free_shape(BRETELLE.length, ns)
    span = Span(
        length=BRETELLE.length,
        tension=BRETELLE.tension,
        boundary_conditions=clamped(),
    )
    for approx in (True, False):
        res = solve_dynamic(
            CONDUCTOR,
            span,
            parameters,
            approx_curvature=approx,
            initial_position=initial,
            energy=True,
        )
        scale = _energy_scale(res)
        work = np.abs(res["curvature_work"].values).max()
        if approx:
            assert work <= 1e-12 * scale
        else:
            # measured 8e-02 for this large slope
            assert work > 1e-2 * scale
        assert _drift(res) <= 1e-8 * scale, approx


@pytest.mark.parametrize("bc", [hinged(), clamped()], ids=["hinged", "clamped"])
@pytest.mark.parametrize("model, approx", CASES)
def test_energy_balance_closes(model, approx, bc):
    excitation = PointExcitation(
        frequency=_line_frequency(), amplitude=100.0, position=0.3 * LINE[0]
    )
    res = solve_dynamic(
        CONDUCTOR,
        _line_span(bc),
        _line_parameters(101, periods=2.0),
        model=model,
        approx_curvature=approx,
        force=_gravity(CONDUCTOR) + excitation,
        zeta=0.02,
        tol=1e-10,
        energy=True,
    )
    scale = _energy_scale(res)
    assert _drift(res) <= 1e-8 * scale

    damping = res["damping_work"].values
    assert np.diff(damping).min() >= -1e-12 * scale
    assert damping[-1] > 0.0

    hysteresis = np.abs(res["hysteresis_work"].values).max()
    if model is BendingModel.CONSTANT:
        assert hysteresis == 0.0
    else:
        assert hysteresis > 0.0


def test_external_work_of_a_constant_force():
    ns = 101
    res = solve_dynamic(
        CONDUCTOR,
        _line_span(hinged()),
        _line_parameters(ns, periods=1.0, los=ns),
        force=_gravity(CONDUCTOR),
        initial_position=np.zeros(ns),
        energy=True,
    )
    # the work of a constant force only depends on the end points
    ds = LINE[0] / (ns - 1)
    z = res["z"].values
    weight = -_GRAVITY * CONDUCTOR.mass
    expected = ds * weight * (z[:, 2:-2] - z[0, 2:-2]).sum(axis=1)
    np.testing.assert_allclose(
        res["external_work"].values,
        expected,
        rtol=1e-12,
        atol=1e-12 * _energy_scale(res),
    )


def test_energy_does_not_depend_on_los():
    ns = 201
    on_nodes = _free_vibration(clamped(), ns, _line_parameters(ns, los=ns))
    mid_span = _free_vibration(clamped(), ns, _line_parameters(ns, los=[0.5]))
    for name in ENERGY:
        np.testing.assert_allclose(
            mid_span[name].values, on_nodes[name].values, rtol=1e-14, err_msg=name
        )


def test_energy_does_not_depend_on_output_rate():
    ns = 201
    every = _free_vibration(clamped(), ns)
    dt = every.lot()[1] - every.lot()[0]
    sparse = _free_vibration(clamped(), ns, _line_parameters(ns, dr=7 * dt))
    for name in ENERGY:
        np.testing.assert_allclose(
            sparse[name].values,
            every[name].values[::7][: len(sparse.lot())],
            rtol=1e-12,
            atol=1e-14 * _energy_scale(every),
            err_msg=name,
        )


def test_energy_balance_with_moving_supports():
    """A heaving support works on the beam; the balance still closes."""
    amplitude, pulsation = 0.01, 2.0 * np.pi * 2.0

    def heave(x, t):
        return amplitude * pulsation * np.cos(pulsation * t)

    def still(x, t):
        return 0.0

    bc = BoundaryCondition(4, dynamic_values=[heave, still, still, still])
    span = Span(
        length=BRETELLE.length, tension=BRETELLE.tension, boundary_conditions=bc
    )
    ns = 101
    res = solve_dynamic(
        CONDUCTOR,
        span,
        simulation.Parameters(ns=ns, tf=1.0, dt=1e-4, dr=1e-3),
        initial_position=np.zeros(ns),
        energy=True,
    )
    scale = _energy_scale(res)
    assert _drift(res) <= 1e-8 * scale
    assert np.abs(res["boundary_work"].values).max() > 1e-6 * scale
