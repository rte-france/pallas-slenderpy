import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy as sp

import slenderpy.future.fd_utils as fdu
from slenderpy.future.boundary_condition import BoundaryCondition


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

    A = fdu.first_derivative(n, ds)

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

    A = fdu.second_derivative(n, ds)
    sol = sp.sparse.linalg.spsolve(A + bc_matrix, rhs)

    left = [[1, 0, 0, 1]]
    right = [[0, 1, 0, 2]]
    bc = BoundaryCondition(order=2, left=left, right=right)
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

    A = fdu.fourth_derivative(n, ds)
    sol = sp.sparse.linalg.spsolve(A + bc_matrix, rhs)

    left = [[1, 0, 0, 0], [0, 1, 0, 1]]
    right = [[1, 0, 0, 1], [0, 1, 0, 2]]
    bc = BoundaryCondition(order=4, left=left, right=right)
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


def test_derivative_size_guards():
    """Check sizes leaving no interior node are rejected."""
    for builder in (fdu.first_derivative, fdu.second_derivative):
        with pytest.raises(ValueError):
            builder(2, 1.0)

    with pytest.raises(ValueError):
        fdu.fourth_derivative(4, 1.0)


@pytest.mark.parametrize(
    "builder, rows",
    [
        (fdu.first_derivative, [0, -1]),
        (fdu.second_derivative, [0, -1]),
        (fdu.fourth_derivative, [0, 1, -2, -1]),
    ],
)
def test_derivative_boundary_rows_are_empty(builder, rows):
    """Check the rows reserved for the boundary conditions are left empty."""
    n = 20

    A = builder(n, 0.1).toarray()
    interior = np.setdiff1d(np.arange(n), np.array(rows) % n)

    assert np.all(A[rows, :] == 0.0)
    assert np.all(np.any(A[interior, :] != 0.0, axis=1))


@pytest.mark.parametrize("order, rows", [(2, [0, -1]), (4, [0, 1, -2, -1])])
def test_clean_matrix(order, rows):
    """Check the expected rows are erased and the others are left untouched."""
    n = 20
    A = fdu.second_derivative(n, 0.1)

    cleaned = fdu.clean_matrix(order, A).toarray()
    expected = A.toarray()
    expected[rows, :] = 0.0

    assert np.array_equal(cleaned, expected)


@pytest.mark.parametrize("order, rows", [(2, [0, -1]), (4, [0, 1, -2, -1])])
def test_clean_matrix_identity(order, rows):
    """Check an identity matrix gets its boundary rows erased.

    This is the matrix used for mass and damping in the dynamic scheme: keeping
    a non-zero coefficient on a boundary row would pollute the boundary
    condition sharing that row.
    """
    n = 20

    cleaned = fdu.clean_matrix(order, sp.sparse.identity(n)).toarray()
    expected = np.eye(n)
    expected[rows, :] = 0.0

    assert np.array_equal(cleaned, expected)


@pytest.mark.parametrize("fmt", ["dia", "csr", "csc", "lil"])
def test_clean_matrix_input_format(fmt):
    """Check any sparse format is accepted and the input is left unchanged."""
    n = 20
    A = fdu.second_derivative(n, 0.1).asformat(fmt)
    before = A.toarray()

    cleaned = fdu.clean_matrix(4, A).toarray()
    expected = before.copy()
    expected[[0, 1, -2, -1], :] = 0.0

    assert np.array_equal(cleaned, expected)
    assert np.array_equal(A.toarray(), before)


def test_clean_matrix_bad_order():
    """Check an order other than 2 or 4 is rejected."""
    with pytest.raises(ValueError):
        fdu.clean_matrix(3, fdu.second_derivative(20, 0.1))


@pytest.mark.parametrize("order, rows", [(2, [0, -1]), (4, [0, 1, -2, -1])])
def test_clean_rhs(order, rows):
    """Check the expected entries are erased and the input is left unchanged."""
    n = 20
    rhs = np.arange(1.0, n + 1.0)

    cleaned = fdu.clean_rhs(order, rhs)
    expected = np.arange(1.0, n + 1.0)
    expected[rows] = 0.0

    assert np.array_equal(cleaned, expected)
    assert np.array_equal(rhs, np.arange(1.0, n + 1.0))


def test_clean_rhs_bad_order():
    """Check an order other than 2 or 4 is rejected."""
    with pytest.raises(ValueError):
        fdu.clean_rhs(3, np.zeros(20))
