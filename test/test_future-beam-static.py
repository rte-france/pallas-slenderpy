import numpy as np
import matplotlib.pyplot as plt

import slenderpy.future.beam.beam as Beam
from slenderpy.future.beam.fd_utils import BoundaryCondition


def _plot(x, exact, sol):
    """Function to plot the analytical and the numerical solution."""
    plt.plot(x, exact, "--", color="blue", label="analytical")
    plt.plot(x, sol, color="orange", label="numerical")
    plt.legend()
    plt.show()


def test_solve_cruvature_approx_order2(plot=False):
    """Check the error between the analytic and numerical solution of:
    y"(x) = x on [2,3]
    y(2) -y(2) + 3y"(2) = 0
    -y(3) + y"(3) = 4
    """

    left_bound = 2
    right_bound = 3
    n = 10000
    x = np.linspace(left_bound, right_bound, n)

    left = [[1, -1, 3, 0]]
    right = [[-1, 0, 1, 4]]
    order = 2
    bc = BoundaryCondition(order, left, right)
    beam = Beam.Beam(length=1, boundary_condition=bc, tension=-1, ei_max=0)
    sol = Beam._solve_static_approx_curvature(n=n, beam=beam, rhs=x)

    def exact(x):
        A = -1 / 12
        B = -63 / 12
        return x**3 / 6 + A * x + B

    if plot:
        _plot(x, exact(x), sol)

    atol = 1.0e-09
    rtol = 1.0e-04

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)


def test_solve_cruvature_approx_order4(plot=False):
    """Check the error between the analytic and numerical solution of:
    y"" - y" = 0 on [0,1]
    y(0) = 0
    y"(0) = 1
    y(1) = 0
    y"(1) = 0
    """

    left_bound = 0
    right_bound = 1
    n = 10000
    x = np.linspace(left_bound, right_bound, n)

    left = [[1, 0, 0, 0], [0, 0, 1, 1]]
    right = [[1, 0, 0, 0], [0, 0, 1, 0]]
    bc = BoundaryCondition(4, left, right)
    rhs = np.zeros(n)
    beam = Beam.Beam(length=1, boundary_condition=bc, tension=1, ei_max=1)
    sol = Beam._solve_static_approx_curvature(n=n, beam=beam, rhs=rhs)

    def exact(x):
        A = -1 / (np.exp(1) ** 2 - 1)
        B = np.exp(1) ** 2 / (np.exp(1) ** 2 - 1)
        D = -B - A
        C = -D - A * np.exp(1) - B * np.exp(-1)
        return A * np.exp(x) + B * np.exp(-x) + C * x + D

    if plot:
        _plot(x, exact(x), sol)

    atol = 1.0e-05
    rtol = 1.0e-09

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)


def test_curvature(plot=False):
    """Check the error between the analytic and numerical curvature of cosh(x)."""

    n = 10000
    lmin = -1.0
    lmax = 1.0
    lspan = lmax - lmin
    x = np.linspace(lmin, lmax, n)
    ds = lspan / n

    def curvature_cosh(x):
        return 1 / np.cosh(x) ** 2

    numerical_curvature = Beam.compute_curvature(n, ds, np.cosh(x))
    exact_curvature = curvature_cosh(x)

    if plot:
        _plot(x[1:-1], exact_curvature[1:-1], numerical_curvature[1:-1])

    atol = 1.0e-09
    rtol = 1.0e-03

    assert np.allclose(
        exact_curvature[1:-1], numerical_curvature[1:-1], atol=atol, rtol=rtol
    )


def test_solve_curvature_exact(plot=False):
    """Check the error between the analytic and numerical solution of:
    8.3*(d^2/dx^2)*(y"*(1 + y'²)^(3/2)) + 5 y" = -24(1 + 4x²)^(5/2) + 480x²(1 + 4x²)^(7/2) - 2 on [-1,1]
    y(-1) = 1
    y'(-1) = -2
    y(1) = 1
    y'(1) = 2
    """

    n = 1000
    lmin = -1.0
    lmax = 1.0
    lspan = lmax - lmin
    x = np.linspace(lmin, lmax, n)

    def rhs(x):
        return 8.3 * (
            -24.0 * (1 + 4 * x**2) ** (-5.0 / 2)
            + 480 * x**2 * (1 + 4 * x**2) ** (-7.0 / 2)
        ) - 2 * (-5)

    left = [[1, 0, 0, lmin**2], [0, 1, 0, 2 * lmin]]
    right = [[1, 0, 0, lmax**2], [0, 1, 0, 2 * lmax]]
    bc = BoundaryCondition(4, left, right)
    beam = Beam.Beam(length=lspan, boundary_condition=bc, tension=-5, ei_max=8.3)
    sol = Beam._solve_static_exact_curvature(n=n, beam=beam, rhs=rhs(x))

    def exact(x):
        return x**2

    if plot:
        _plot(x, exact(x), sol)

    atol = 1.0e-03
    rtol = 1.0e-09

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)


def test_solve_curvature_approx_bending_moment_variable(plot=False):
    n = 1000
    lmin = -1.0
    lmax = 3.0
    lspan = lmax - lmin
    x = np.linspace(lmin, lmax, n)

    ei_min = 18.23
    ei_max = 589.64
    critical_curvature = 25.8
    chi_bar = (1 - ei_min / ei_max) * critical_curvature
    H = 1485.24

    def curvature(x):
        return -np.sin(x)

    def curvature_first_derivative(x):
        return -np.cos(x)

    def curvature_second_derivative(x):
        return np.sin(x)

    def rhs(x):
        C = curvature(x)
        C1 = curvature_first_derivative(x)
        C2 = curvature_second_derivative(x)
        E = np.exp(-C / chi_bar)
        return (
            ei_min * C2 * (1 - E)
            + 2 * ei_min * C1**2 * E / chi_bar
            + (ei_max * chi_bar + ei_min * C)
            * (C2 * E / chi_bar - C1**2 * E / chi_bar**2)
            - H * curvature(x)
        )

    def exact(x):
        return np.sin(x)

    left = [[1, 0, 0, exact(lmin)], [0, 1, 0, np.cos(lmin)]]
    right = [[1, 0, 0, exact(lmax)], [0, 1, 0, np.cos(lmax)]]
    bc = BoundaryCondition(4, left, right)
    beam = Beam.Beam(
        length=lspan,
        boundary_condition=bc,
        tension=H,
        ei_max=ei_max,
        ei_min=ei_min,
        critical_curvature=critical_curvature,
    )
    sol = Beam._solve_static_approx_curvature(n=n, beam=beam, rhs=rhs(x))

    if plot:
        _plot(x, exact(x), sol)

    atol = 1.0e-03
    rtol = 1.0e-09

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)


def test_solve_curvature_exact_bending_moment_variable(plot=False):
    n = 1000
    lmin = -1.0
    lmax = 3.0
    lspan = lmax - lmin
    x = np.linspace(lmin, lmax, n)

    ei_min = 253.2
    ei_max = 1234.9
    critical_curvature = 12.4
    chi_bar = (1 - ei_min / ei_max) * critical_curvature
    H = 3.2

    def curvature(x):
        return 1.0 / np.cosh(x) ** 2

    def curvature_first_derivative(x):
        return -2 * np.sinh(x) / np.cosh(x) ** 3

    def curvature_second_derivative(x):
        return -2 / np.cosh(x) ** 2 + 6.0 * np.sinh(x) ** 2 / np.cosh(x) ** 4

    def rhs(x):
        C = curvature(x)
        C1 = curvature_first_derivative(x)
        C2 = curvature_second_derivative(x)
        E = np.exp(-C / chi_bar)
        return (
            ei_min * C2 * (1 - E)
            + 2 * ei_min * C1**2 * E / chi_bar
            + (ei_max * chi_bar + ei_min * C)
            * (C2 * E / chi_bar - C1**2 * E / chi_bar**2)
            - H * np.cosh(x)
        )

    def exact(x):
        return np.cosh(x)

    left = [[1, 0, 0, np.cosh(lmin)], [0, 1, 0, np.sinh(lmin)]]
    right = [[1, 0, 0, np.cosh(lmax)], [0, 1, 0, np.sinh(lmax)]]
    bc = BoundaryCondition(4, left, right)
    beam = Beam.Beam(
        length=lspan,
        boundary_condition=bc,
        tension=H,
        ei_max=ei_max,
        ei_min=ei_min,
        critical_curvature=critical_curvature,
    )
    sol = Beam._solve_static_exact_curvature(n=n, beam=beam, rhs=rhs(x))

    if plot:
        _plot(x, exact(x), sol)

    atol = 1.0e-03
    rtol = 1.0e-03

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)
