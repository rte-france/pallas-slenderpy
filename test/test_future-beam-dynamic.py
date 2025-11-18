import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import slenderpy.future.beam.beam as Beam
from slenderpy.future.beam.fd_utils import BoundaryCondition
from slenderpy import simtools


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

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=np.arange(0, nb_time + 1),
        interval=1,
        blit=True,
        repeat=True,
    )
    plt.show()


def test_solve_approx_curvature_static_BC(plot=False):
    nb_space = 300
    dt = 1e-4
    final_time = 1.0
    mass = 14.3
    tension = 125.78
    ei_max = 1484.75
    lspan = 3.0

    x = np.linspace(0, lspan, nb_space)

    def f(t):
        return np.cos(t)

    def exact(x, t):
        return f(t) * x**2 * (x - lspan) ** 2 + 1

    def exact_time_derivative(x, t):
        return -np.sin(t) * x**2 * (x - lspan) ** 2

    def force(x, t):
        return (
            -mass * np.cos(t) * x**2 * (x - lspan) ** 2
            + ei_max * 24.0 * f(t)
            - tension * f(t) * (12 * x**2 - 12 * lspan * x + 2 * lspan**2)
        )

    left = [[1, 0, 0, 1], [0, 1, 0, 0]]
    right = [[1, 0, 0, 1], [0, 1, 0, 0]]
    bc = BoundaryCondition(4, left, right)

    beam = Beam.Beam(
        length=lspan, boundary_condition=bc, tension=tension, ei_max=ei_max, mass=mass
    )
    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-3, los=nb_space
    )

    sol = Beam._solve_dynamic_approx_curvature(
        beam=beam,
        parameters=parameters,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
        force=force,
    )

    y = sol.data["y"]

    if plot:
        _plot_animation(x, exact, y, -7, 7, parameters.nr, final_time)

    analitical_results = np.array(
        [exact(x, i * (final_time / parameters.nr)) for i in range(parameters.nr + 1)]
    )

    atol = 1.0e-06
    rtol = 1.0e-01

    assert np.allclose(analitical_results, y, atol=atol, rtol=rtol)


def test_solve_approx_curvature_dynamic_BC(plot=False):
    nb_space = 400
    dt = 1e-4
    final_time = 1.2
    mass = 1.45
    tension = 12.36
    ei_max = 147.89
    lmin = 0.0
    lmax = 4.0
    lspan = lmax - lmin

    x = np.linspace(lmin, lmax, nb_space)

    def exact(x, t):
        return np.cosh(x - 2) * np.sin(2 * np.pi * t)

    def exact_space_derivative(x, t):
        return np.sinh(x - 2) * np.sin(2 * np.pi * t)

    def exact_time_derivative(x, t):
        return 2 * np.pi * np.cosh(x - 2) * np.cos(2 * np.pi * t)

    def force(x, t):
        return (
            -4 * np.pi**2 * mass * exact(x, t)
            + ei_max * exact(x, t)
            - tension * exact(x, t)
        )

    left = [[1, 0, 0, exact(lmin, 0)], [0, 1, 0, exact_space_derivative(lmin, 0)]]
    right = [[1, 0, 0, exact(lmax, 0)], [0, 1, 0, exact_space_derivative(lmax, 0)]]
    dynamic_values = [exact, exact_space_derivative, exact_space_derivative, exact]
    bc = BoundaryCondition(4, left, right, dynamic_values)

    beam = Beam.Beam(
        length=lspan, boundary_condition=bc, tension=tension, ei_max=ei_max, mass=mass
    )
    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-3, los=nb_space
    )
    sol = Beam._solve_dynamic_approx_curvature(
        beam=beam,
        parameters=parameters,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
        force=force,
    )

    y = sol.data["y"]

    if plot:
        _plot_animation(x, exact, y, -5, 5, parameters.nr, final_time)

    analitical_results = np.array(
        [exact(x, i * (final_time / parameters.nr)) for i in range(parameters.nr + 1)]
    )
    atol = 1.0e-01
    rtol = 1.0e-03

    assert np.allclose(analitical_results, y, atol=atol, rtol=rtol)


def test_solve_exact_curvature(plot=False):
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

    def force(x, t):
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

    def exact_space_derivative(x, t):
        return np.sinh(x + t)

    def exact_time_derivative(x, t):
        return np.sinh(x + t)

    def curvature(x, t):
        return 1 / np.cosh(x + t) ** 2

    left = [[1, 0, 0, exact(lmin, 0)], [0, 1, 0, exact_space_derivative(lmin, 0)]]
    right = [[1, 0, 0, exact(lmax, 0)], [0, 1, 0, exact_space_derivative(lmax, 0)]]
    dynamic_values = [exact, exact_space_derivative, exact_space_derivative, exact]
    bc = BoundaryCondition(4, left, right, dynamic_values)

    beam_ei_const = Beam.Beam(
        length=lspan, boundary_condition=bc, tension=tension, ei_max=ei_max, mass=mass
    )
    beam = Beam.Beam(
        length=lspan,
        boundary_condition=bc,
        tension=tension,
        ei_max=ei_max,
        ei_min=ei_min,
        critical_curvature=chi0,
        mass=mass,
    )
    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-3, los=nb_space
    )
    sol_ei_const = Beam._solve_dynamic_exact_curvature_EI_const(
        beam=beam_ei_const,
        parameters=parameters,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
        force=force,
    )
    sol = Beam._solve_dynamic_exact_curvature(
        beam=beam,
        parameters=parameters,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
        initial_bending_moment=ei_max * curvature(x, 0),
        force=force,
    )

    y_ei_const = sol_ei_const.data["y"]
    y = sol.data["y"]

    if plot:
        _plot_animation(x, exact, y_ei_const, 1, 7, parameters.nr, final_time)
        _plot_animation(x, exact, y, 1, 7, parameters.nr, final_time)

    analitical_results = np.array(
        [exact(x, i * (final_time / parameters.nr)) for i in range(parameters.nr + 1)]
    )
    atol = 1.0e-2
    rtol = 1.0e-2

    assert np.allclose(
        analitical_results, y_ei_const, atol=atol, rtol=rtol
    ) and np.allclose(analitical_results, y, atol=atol, rtol=rtol)
