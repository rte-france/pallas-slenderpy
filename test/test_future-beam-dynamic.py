import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from slenderpy.future.beam.beam import BeamConst, BeamBW
from slenderpy.future.beam.fd_utils import BoundaryCondition
from slenderpy import simtools


def _plot_animation(x, x_border, exact, sol, ymin, ymax, nb_time, final_time):
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
        analytical = exact(x_border, t)
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


def test_solve_approx_curvature_bending_moment_constant_static_BC(plot=False):
    nb_space = 300
    dt = 1e-4
    final_time = 1.0
    mass = 14.3
    tension = 125.78
    ei_min = 1484.75
    lspan = 3.0

    x = np.linspace(0, lspan, nb_space)
    x_border = np.linspace(0, lspan, nb_space + 2)[1:-1]

    def f(t):
        return np.cos(t)

    def exact(x, t):
        return f(t) * x**2 * (x - lspan) ** 2 + 1

    def exact_time_derivative(x, t):
        return -np.sin(t) * x**2 * (x - lspan) ** 2

    def force(x, t, y, v):
        return (
            -mass * np.cos(t) * x**2 * (x - lspan) ** 2
            + ei_min * 24.0 * f(t)
            - tension * f(t) * (12 * x**2 - 12 * lspan * x + 2 * lspan**2)
        )

    left = [[1, 0, 0, 1], [0, 1, 0, 0]]
    right = [[1, 0, 0, 1], [0, 1, 0, 0]]
    bc = BoundaryCondition(4, left, right)

    beam = BeamConst(
        length=lspan, boundary_conditions=bc, tension=tension, ei=ei_min, mass=mass
    )
    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-3, los=nb_space
    )

    sol = beam.solve_dynamic(
        parameters=parameters,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
        force=force,
        approx_curvature=True,
    )

    y = sol["y"]

    if plot:
        _plot_animation(x, x_border, exact, y, -7, 7, parameters.nr, final_time)

    analitical_results = np.array(
        [
            exact(x_border, i * (final_time / parameters.nr))
            for i in range(parameters.nr + 1)
        ]
    )

    atol = 1.0e-06
    rtol = 1.0e-02

    assert np.allclose(analitical_results, y, atol=atol, rtol=rtol)


def test_solve_approx_curvature_bending_moment_constant_dynamic_BC(plot=False):
    nb_space = 400
    dt = 1e-4
    final_time = 1.2
    mass = 1.45
    tension = 12.36
    ei_min = 147.89
    lmin = 0.0
    lmax = 4.0
    lspan = lmax - lmin

    x = np.linspace(lmin, lmax, nb_space)
    x_border = np.linspace(lmin, lmax, nb_space + 2)[1:-1]

    def exact(x, t):
        return np.cosh(x - 2) * np.sin(2 * np.pi * t)

    def exact_time_space_derivative(x, t):
        return 2 * np.pi * np.sinh(x - 2) * np.cos(2 * np.pi * t)

    def exact_time_derivative(x, t):
        return 2 * np.pi * np.cosh(x - 2) * np.cos(2 * np.pi * t)

    def force(x, t, y, v):
        return (
            -4 * np.pi**2 * mass * exact(x, t)
            + ei_min * exact(x, t)
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

    beam = BeamConst(
        length=lspan, boundary_conditions=bc, tension=tension, ei=ei_min, mass=mass
    )
    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-3, los=nb_space
    )
    sol = beam.solve_dynamic(
        parameters=parameters,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
        force=force,
        approx_curvature=True,
    )

    y = sol["y"]

    if plot:
        _plot_animation(x, x_border, exact, y, -5, 5, parameters.nr, final_time)

    analitical_results = np.array(
        [
            exact(x_border, i * (final_time / parameters.nr))
            for i in range(parameters.nr + 1)
        ]
    )
    atol = 1.0e-01
    rtol = 1.0e-06

    assert np.allclose(analitical_results, y, atol=atol, rtol=rtol)


def test_solve_exact_curvature_bending_moment_constant(plot=False):
    nb_space = 100
    dt = 1e-5
    final_time = 0.1
    mass = 9.8
    tension = 256.12
    ei_min = 2698.23
    lmin = 0.0
    lmax = 2.0
    lspan = lmax - lmin
    x_border = np.linspace(lmin, lmax, nb_space + 2)[1:-1]
    x = np.linspace(lmin, lmax, nb_space)

    def force(x, t, y, v):
        return (
            mass * np.cosh(x + t)
            + ei_min
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

    beam = BeamConst(
        length=lspan, boundary_conditions=bc, tension=tension, ei=ei_min, mass=mass
    )
    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-3, los=nb_space
    )
    sol = beam.solve_dynamic(
        parameters=parameters,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
        force=force,
        approx_curvature=False,
        it_picard=10,
        tol_picard=1e-4,
    )
    y = sol["y"]

    if plot:
        _plot_animation(x, x_border, exact, y, 1, 7, parameters.nr, final_time)

    analitical_results = np.array(
        [
            exact(x_border, i * (final_time / parameters.nr))
            for i in range(parameters.nr + 1)
        ]
    )
    atol = 1.0e-6
    rtol = 1.0e-3

    assert np.allclose(analitical_results, y, atol=atol, rtol=rtol)


def test_solve_approx_curvature_bending_moment_variable(plot=False):
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
    x_border = np.linspace(lmin, lmax, nb_space + 2)[1:-1]
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

    beam = BeamBW(
        length=lspan,
        boundary_conditions=bc,
        tension=tension,
        ei_max=ei_max,
        ei_min=ei_min,
        critical_curvature=chi0,
        mass=mass,
    )
    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-3, los=nb_space
    )
    sol = beam.solve_dynamic(
        parameters=parameters,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
        force=force,
        approx_curvature=True,
        it_picard=15,
        tol_picard=1e-4,
    )

    y = sol["y"]

    if plot:
        _plot_animation(x, x_border, exact, y, 1, 7, parameters.nr, final_time)

    analitical_results = np.array(
        [
            exact(x_border, i * (final_time / parameters.nr))
            for i in range(parameters.nr + 1)
        ]
    )
    atol = 1.0e-6
    rtol = 1.0e-1

    assert np.allclose(analitical_results, y, atol=atol, rtol=rtol)


def test_solve_exact_curvature_bending_moment_variable(plot=False):
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
    x_border = np.linspace(lmin, lmax, nb_space + 2)[1:-1]
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

    beam = BeamBW(
        length=lspan,
        boundary_conditions=bc,
        tension=tension,
        ei_max=ei_max,
        ei_min=ei_min,
        critical_curvature=chi0,
        mass=mass,
    )
    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-3, los=nb_space
    )
    sol = beam.solve_dynamic(
        parameters=parameters,
        initial_position=exact(x, 0),
        initial_velocity=exact_time_derivative(x, 0),
        force=force,
        approx_curvature=False,
        it_picard=15,
        tol_picard=1e-4,
    )

    y = sol["y"]

    if plot:
        _plot_animation(x, x_border, exact, y, 1, 7, parameters.nr, final_time)

    analitical_results = np.array(
        [
            exact(x_border, i * (final_time / parameters.nr))
            for i in range(parameters.nr + 1)
        ]
    )
    atol = 1.0e-6
    rtol = 1.0e-3

    assert np.allclose(analitical_results, y, atol=atol, rtol=rtol)
