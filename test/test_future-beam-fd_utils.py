import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import slenderpy.future.beam.fd_utils as FD


def _plot(x, exact, sol):
    """Function to plot the analytical and the numerical solution."""
    plt.plot(x, exact, "--", color="blue", label="analytical")
    plt.plot(x, sol, color="orange", label="numerical")
    plt.legend()
    plt.show()


def test_first_derivative(plot=False):
    """Check the error between the analytic and numerical solution of:
    y'(x) = sin(x) on [-1,2]
    y(-1) = 3
    """
    left_bound = -1
    right_bound = 2
    n = 10000
    ds = (right_bound - left_bound) / (n - 1)
    x = np.linspace(left_bound, right_bound, n)

    rhs = np.sin(x)
    rhs[0] = 3

    bc_matrix = sp.sparse.lil_matrix((n, n))
    bc_matrix[0, 0] = 1
    bc_matrix[-1, -1] = 1 / ds
    bc_matrix[-1, -2] = -1 / ds

    A = FD.first_derivative(n, ds)

    sol = sp.sparse.linalg.spsolve(A + bc_matrix, rhs)

    def exact(x):
        return -np.cos(x) + 3 + np.cos(-1)

    if plot:
        _plot(x, exact(x), sol)

    atol = 1.0e-06
    rtol = 1.0e-09

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)


def test_second_derivative(plot=False):
    """Check the error between the analytic and numerical solution of:
    y"(x) = 0 on [0,1]
    y(0) = 0 and y'(1) = 2
    """
    left_bound = 0
    right_bound = 1
    n = 100
    ds = (right_bound - left_bound) / (n - 1)
    x = np.linspace(left_bound, right_bound, n)

    rhs = np.zeros(n)
    rhs[0] = 1
    rhs[-1] = 2

    bc_matrix = sp.sparse.lil_matrix((n, n))
    bc_matrix[0, 0] = 1
    bc_matrix[-1, -1] = 1 / ds
    bc_matrix[-1, -2] = -1 / ds

    A = FD.second_derivative(n, ds)
    sol = sp.sparse.linalg.spsolve(A + bc_matrix, rhs)

    left = [[1, 0, 0, 1]]
    right = [[0, 1, 0, 2]]
    bc = FD.BoundaryCondition(order=2, left=left, right=right)
    BC, rhs = bc.compute(n, ds)
    sol_bc = sp.sparse.linalg.spsolve(A + BC, rhs)

    def exact(x):
        return 2 * x + 1

    if plot:
        _plot(x, exact(x), sol)
        _plot(x, exact(x), sol_bc)

    atol = 1.0e-06
    rtol = 1.0e-09

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)
    assert np.allclose(exact(x), sol_bc, atol=atol, rtol=rtol)


def test_fourth_derivative(plot=False):
    """Check the error between the analytic and numerical solution of:
    y""(x) = 0 on [0,1]
    y(0) = 0
    y'(0) = 1
    y(1) = 1
    y'(1) = 2
    """

    left_bound = 0
    right_bound = 1
    n = 10000
    ds = (right_bound - left_bound) / (n - 1)
    x = np.linspace(left_bound, right_bound, n)

    rhs = np.zeros(n)
    rhs[0] = 0
    rhs[1] = 1
    rhs[-2] = 1
    rhs[-1] = 2

    bc_matrix = sp.sparse.lil_matrix((n, n))
    bc_matrix[0, 0] = 1
    bc_matrix[1, 1] = 1 / ds
    bc_matrix[1, 0] = -1 / ds
    bc_matrix[-2, -2] = 1
    bc_matrix[-1, -1] = 1 / ds
    bc_matrix[-1, -2] = -1 / ds

    A = FD.fourth_derivative(n, ds)
    sol = sp.sparse.linalg.spsolve(A + bc_matrix, rhs)

    left = [[1, 0, 0, 0], [0, 1, 0, 1]]
    right = [[1, 0, 0, 1], [0, 1, 0, 2]]
    bc = FD.BoundaryCondition(order=4, left=left, right=right)
    BC, rhs = bc.compute(n, ds)
    sol_bc = sp.sparse.linalg.spsolve(A + BC, rhs)

    def exact(x):
        return x**3 - x**2 + x

    if plot:
        _plot(x, exact(x), sol)
        _plot(x, exact(x), sol_bc)

    atol = 1.0e-06
    rtol = 1.0e-03

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)
    assert np.allclose(exact(x), sol_bc, atol=atol, rtol=rtol)
