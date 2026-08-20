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

The rest of the module holds the numerical helpers the solvers share:
:func:`banded`, :func:`tridiagonal` and :func:`product_band` move a narrow
sparse matrix in and out of the LAPACK banded layout, and :func:`residual_scale`
gives a Newton iteration something to measure its residual against.
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


# a scheme matrix is pentadiagonal: every term of it (D4, D2 @ diag @ D2, the D1
# term of the exact curvature, the boundary rows) fits within two diagonals of
# the main one
BANDWIDTH = 2


def banded(a: sp.sparse.spmatrix) -> np.ndarray:
    """LAPACK banded layout of a pentadiagonal sparse matrix.

    Parameters
    ----------
    a : sp.sparse.spmatrix
        Matrix to lay out, at most pentadiagonal.

    Returns
    -------
    np.ndarray
        Array of shape ``(2 * BANDWIDTH + 1, n)`` where
        ``out[BANDWIDTH + i - j, j]`` holds ``a[i, j]``, which is what
        ``scipy.linalg.solve_banded`` expects for ``l = u = BANDWIDTH``.

    Raises
    ------
    ValueError
        If the matrix has an entry further than ``BANDWIDTH`` from the diagonal.
    """
    coo = sp.sparse.csr_matrix(a).tocoo()
    if np.abs(coo.row - coo.col).max(initial=0) > BANDWIDTH:
        raise ValueError("matrix is not pentadiagonal")

    res = np.zeros((2 * BANDWIDTH + 1, a.shape[0]))
    res[BANDWIDTH + coo.row - coo.col, coo.col] = coo.data

    return res


def tridiagonal(a: sp.sparse.spmatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-indexed diagonals of a tridiagonal sparse matrix.

    Parameters
    ----------
    a : sp.sparse.spmatrix
        Matrix to split, at most tridiagonal.

    Returns
    -------
    tuple of np.ndarray
        ``(lower, diag, upper)`` with ``lower[i] = a[i, i - 1]`` and
        ``upper[i] = a[i, i + 1]``, zero where that entry does not exist.
    """
    band = banded(a)
    n = a.shape[0]

    lower, upper = np.zeros(n), np.zeros(n)
    lower[1:] = band[BANDWIDTH + 1, : n - 1]
    upper[: n - 1] = band[BANDWIDTH - 1, 1:]

    return lower, band[BANDWIDTH], upper


def product_band(
    left: tuple[np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray],
    scale: np.ndarray,
) -> np.ndarray:
    """Banded layout of ``L @ diag(scale) @ R``, both factors tridiagonal.

    Assembling the product from its diagonals costs a handful of vector
    products, where the equivalent sparse matrix multiplications cost an order of
    magnitude more for a band this narrow.

    Parameters
    ----------
    left : tuple of np.ndarray
        The ``(lower, diag, upper)`` triple of ``L``, as :func:`tridiagonal`
        returns.
    right : tuple of np.ndarray
        The same triple for ``R``.
    scale : np.ndarray
        Diagonal inserted between the two factors.

    Returns
    -------
    np.ndarray
        The product in the layout of :func:`banded`.
    """
    lower, diag, upper = left
    r_low, r_dia, r_up = right
    n = diag.size

    # columns of the left factor carry the scale; a, b, c multiply rows i-1, i
    # and i+1 of the right factor respectively
    a, c = np.zeros(n), np.zeros(n)
    a[1:] = lower[1:] * scale[:-1]
    b = diag * scale
    c[:-1] = upper[:-1] * scale[1:]

    res = np.zeros((2 * BANDWIDTH + 1, n))
    res[4, : n - 2] = a[2:] * r_low[1:-1]
    res[3, : n - 1] = a[1:] * r_dia[:-1] + b[1:] * r_low[1:]
    res[2, :] = b * r_dia
    res[2, 1:] += a[1:] * r_up[:-1]
    res[2, : n - 1] += c[:-1] * r_low[1:]
    res[1, 1:] = b[:-1] * r_up[:-1] + c[:-1] * r_dia[1:]
    res[0, 2:] = c[:-2] * r_up[1:-1]

    return res


def residual_scale(terms) -> float:
    """Magnitude of the largest term of a residual, as a tolerance scale.

    A criterion relative to the iterate itself cannot be used in a time-domain
    solve: the velocity goes through zero twice per oscillation, where anything
    measured relative to it blows up.

    Parameters
    ----------
    terms : iterable of np.ndarray
        The terms the residual is assembled from.

    Returns
    -------
    float
        The largest magnitude among the terms, floored at 1e-30 so it can be
        divided by.
    """
    return max(max(np.abs(term).max() for term in terms), 1.0e-30)
