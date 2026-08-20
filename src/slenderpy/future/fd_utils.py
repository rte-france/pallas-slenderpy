"""Finite-difference operators for beam/cable schemes.

Each builder returns the centered-difference matrix of a given derivative on a
uniform grid of ``n`` nodes spaced by ``ds``. The rows carrying the boundary
conditions are left empty: two rows (first and last) for an order-2 scheme,
four rows (two first and two last) for an order-4 scheme. Those rows are meant
to be filled by a :class:`~slenderpy.future.boundary_condition.BoundaryCondition`
contribution added to the matrix.

:func:`clean_matrix` and :func:`clean_rhs` empty those same rows in any other
matrix or right-hand side taking part in the scheme, so that the boundary
conditions are the only relations enforced there.
"""

import numpy as np
import scipy as sp


def _boundary_rows(order: int) -> list[int]:
    """Indices of the rows reserved for the boundary conditions.

    Parameters
    ----------
    order : int
        Number of boundary conditions (2 or 4).

    Returns
    -------
    list[int]
        Row indices, negative values counting from the end.

    Raises
    ------
    ValueError
        If order different from 2 or 4.
    """
    if order not in (2, 4):
        raise ValueError("order must be 2 or 4")

    return [0, -1] if order == 2 else [0, 1, -2, -1]


def first_derivative(n: int, ds: float) -> sp.sparse.dia_matrix:
    """Centered scheme, the first and last line have to be completed with BC (order 2).

    Parameters
    ----------
    n : int
       Matrix size, at least 3.
    ds : float
        Space discretization step.

    Returns
    -------
    sp.sparse.dia_matrix
        Derivative matrix.

    Raises
    ------
    ValueError
        If n leaves no interior node (n < 3).
    """
    if n < 3:
        raise ValueError("n must be at least 3 for an order-2 scheme")

    dinf = -1.0 * np.ones((n - 1,)) / (2 * ds)
    dsup = +1.0 * np.ones((n - 1,)) / (2 * ds)

    dinf[-1] = 0.0
    dsup[0] = 0.0

    res = sp.sparse.diags([dinf, dsup], [-1, 1])

    return res


def second_derivative(n: int, ds: float) -> sp.sparse.dia_matrix:
    """Centered scheme, the first and last line have to be completed with BC (order 2).

    Parameters
    ----------
    n : int
        Matrix size, at least 3.
    ds : float
        Space discretization step.

    Returns
    -------
    sp.sparse.dia_matrix
        Derivative matrix.

    Raises
    ------
    ValueError
        If n leaves no interior node (n < 3).
    """
    if n < 3:
        raise ValueError("n must be at least 3 for an order-2 scheme")

    dinf = +1.0 * np.ones((n - 1,)) / ds**2
    diag = -2.0 * np.ones((n,)) / ds**2
    dsup = +1.0 * np.ones((n - 1,)) / ds**2

    dinf[-1] = 0.0
    diag[0] = 0.0
    diag[-1] = 0.0
    dsup[0] = 0.0

    res = sp.sparse.diags([dinf, diag, dsup], [-1, 0, 1])

    return res


def fourth_derivative(n: int, ds: float) -> sp.sparse.dia_matrix:
    """Centered scheme, the two first and two last lines have to be completed with BC (order 4).

    Parameters
    ----------
    n : int
        Matrix size, at least 5.
    ds : float
        Space discretization step.

    Returns
    -------
    sp.sparse.dia_matrix
        Derivative matrix.

    Raises
    ------
    ValueError
        If n leaves no interior node (n < 5).
    """
    if n < 5:
        raise ValueError("n must be at least 5 for an order-4 scheme")

    dinf2 = +1.0 * np.ones((n - 2,)) / ds**4
    dinf1 = -4.0 * np.ones((n - 1,)) / ds**4
    diag = +6.0 * np.ones((n,)) / ds**4
    dsup1 = -4.0 * np.ones((n - 1,)) / ds**4
    dsup2 = +1.0 * np.ones((n - 2,)) / ds**4

    dinf2[[-1, -2]] = 0.0
    dinf1[[-1, -2, 0]] = 0.0
    diag[[0, 1, -2, -1]] = 0.0
    dsup1[[0, 1, -1]] = 0.0
    dsup2[[0, 1]] = 0.0

    res = sp.sparse.diags([dinf2, dinf1, diag, dsup1, dsup2], [-2, -1, 0, 1, 2])

    return res


def clean_matrix(order: int, a: sp.sparse.spmatrix) -> sp.sparse.csr_matrix:
    """Erase the rows reserved for the boundary conditions in a scheme matrix.

    Parameters
    ----------
    order : int
        Number of boundary conditions (2 or 4).
    a : sp.sparse.spmatrix
        Matrix to clean, left unchanged.

    Returns
    -------
    sp.sparse.csr_matrix
        Cleaned copy of the input matrix.

    Raises
    ------
    ValueError
        If order different from 2 or 4.
    """
    rows = _boundary_rows(order)

    res = a.tolil(copy=True)
    res[rows, :] = 0.0

    return res.tocsr()


def clean_rhs(order: int, rhs: np.ndarray) -> np.ndarray:
    """Erase the entries reserved for the boundary conditions in a right-hand side.

    Parameters
    ----------
    order : int
        Number of boundary conditions (2 or 4).
    rhs : np.ndarray
        Right-hand side to clean, left unchanged.

    Returns
    -------
    np.ndarray
        Cleaned copy of the input right-hand side.

    Raises
    ------
    ValueError
        If order different from 2 or 4.
    """
    rows = _boundary_rows(order)

    res = np.copy(rhs)
    res[rows] = 0.0

    return res
