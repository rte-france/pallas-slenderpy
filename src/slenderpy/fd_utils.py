"""Finite-difference operators for beam/cable schemes.

Each builder returns the centered-difference matrix of a given derivative on a
uniform grid of ``n`` nodes spaced by ``ds``. The rows carrying the boundary
conditions are left empty: two rows (first and last) for an order-2 scheme,
four rows (two first and two last) for an order-4 scheme. Those rows are meant
to be filled by a [](`~slenderpy.boundary_condition.BoundaryCondition`)
contribution added to the matrix. An order-4 scheme runs on the extended grid
of the beam solvers, so its four rows are the ghost nodes and the supports.

[](`~slenderpy.fd_utils.clean_matrix`) and [](`~slenderpy.fd_utils.clean_rhs`) empty those same rows in any other
matrix or right-hand side taking part in the scheme, so that the boundary
conditions are the only relations enforced there.

The rest of the module holds the numerical helpers the solvers share:
[](`~slenderpy.fd_utils.banded`), [](`~slenderpy.fd_utils.tridiagonal`) and [](`~slenderpy.fd_utils.product_band`) move a narrow
sparse matrix in and out of the LAPACK banded layout, and [](`~slenderpy.fd_utils.residual_scale`)
gives a Newton iteration something to measure its residual against.
"""

import numpy as np
import scipy as sp
import scipy.sparse
from typing import Tuple, Optional


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


def first_derivative_with_borders(n: int, ds: float) -> sp.sparse.csr_matrix:
    """Centered scheme, with the first and last line completed (order 2).

    Parameters
    ----------
    n : int
        Matrix size, at least 3.
    ds : float
        Space discretization step.

    Returns
    -------
    sp.sparse.csr_matrix
        Derivative matrix with borders.

    Raises
    ------
    ValueError
        If n leaves no interior node (n < 3).
    """
    d1 = first_derivative(n, ds).tolil()
    d1[0, :3] = np.array([-3 / 2, 2, -1 / 2]) / ds
    d1[-1, -3:] = np.array([1 / 2, -2, 3 / 2]) / ds

    return d1.tocsr()


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


def second_derivative_with_borders(n: int, ds: float) -> sp.sparse.csr_matrix:
    """Centered scheme, with the first and last line completed (order 2).

    Parameters
    ----------
    n : int
        Matrix size, at least 4.
    ds : float
        Space discretization step.

    Returns
    -------
    sp.sparse.csr_matrix
        Derivative matrix with borders.

    Raises
    ------
    ValueError
        If n leaves no interior node (n < 4).
    """
    if n < 4:
        raise ValueError("n must be at least 4 for an order-2 scheme with borders")

    d2 = second_derivative(n, ds).tolil()
    d2[0, :4] = np.array([2.0, -5.0, 4.0, -1.0]) / ds**2
    d2[-1, -4:] = np.array([-1.0, 4.0, -5.0, 2.0]) / ds**2

    return d2.tocsr()


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
        The ``(lower, diag, upper)`` triple of ``L``, as [](`~slenderpy.fd_utils.tridiagonal`)
        returns.
    right : tuple of np.ndarray
        The same triple for ``R``.
    scale : np.ndarray
        Diagonal inserted between the two factors.

    Returns
    -------
    np.ndarray
        The product in the layout of [](`~slenderpy.fd_utils.banded`).
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


# multiple of the estimate below which a residual is taken as round-off
ROUND_OFF_FACTOR = 16.0


def inf_norm(a: sp.sparse.spmatrix) -> float:
    """Infinity norm of a sparse matrix, the largest absolute row sum."""
    return float(abs(a).sum(axis=1).max())


def round_off_floor(pairs) -> float:
    """Round-off level of a residual built from operator-vector products.

    A product ``A @ x`` cannot be evaluated better than about
    ``eps * ||A|| * max|x|``. On a fine grid the ``D4`` operator grows like
    ``EI / ds**4``, so this level can exceed a tolerance taken relative to the
    load, and a solver must not ask for less than it.

    Parameters
    ----------
    pairs : iterable of (float, np.ndarray)
        ``(||A||, x)`` for each product of the residual, see [](`~slenderpy.fd_utils.inf_norm`).

    Returns
    -------
    float
        ``ROUND_OFF_FACTOR * eps * sum(||A|| * max|x|)``.
    """
    eps = np.finfo(float).eps
    return ROUND_OFF_FACTOR * eps * sum(norm * np.abs(x).max() for norm, x in pairs)







######### BUNDLE ##########


class BoundaryCondition:
    """Object to deal with boundary conditions."""

    def __init__(
        self,
        t1: Optional[Tuple[float, float, float, float]] = None,
        t2: Optional[Tuple[float, float, float, float]] = None,
        pos: Optional[str] = None,
    ) -> None:
        """Init with args.

        Input t1 and t2 are tuples with four floats each such that
        ti = (ai, bi, ci, di) for i in {1, 2} and:
            a1 * y(b) + b1 * (dy/dx)(b) * c1 * (d2y/dx2)(b) = d1
            a2 * y(b) + b2 * (dy/dx)(b) * c2 * (d2y/dx2)(b) = d2

        b is the bound defined by pos (min for left, max for right). The
        equation system defined by both tuples should have a nonzero det.

        If None values are used for t1 or t2, Dirichlet boundary conditions are
        used.

        Parameters
        ----------
        t1 : tuple, optional
            DESCRIPTION. The default is None.
        t2 : tuple, optional
            DESCRIPTION. The default is None.
        pos : str
            Must be either 'left' or right.

        Raises
        ------
        TypeError
            DESCRIPTION.
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.
        """
        if t1 is None:
            t1 = (1.0, 0.0, 0.0, 0.0)
        if t2 is None:
            t2 = (0.0, 1.0, 0.0, 0.0)
        if (not isinstance(t1, (tuple, list))) or (not isinstance(t2, (tuple, list))):
            raise TypeError("Inputs t1 and t2 must be list or tuples")
        if len(t1) != 4 or len(t2) != 4:
            raise ValueError("Inputs t1 and t2 must have 4 elements")
        if pos == "left":
            self.pp = -1.0
        elif pos == "right":
            self.pp = +1.0
        else:
            raise ValueError("Input pos must be either 'left' or 'right'")
        self.t1 = t1
        self.t2 = t2
        self.c1c = None
        self.c1q = None
        self.c2c = None
        self.c2q = None

    def set(self, ds):
        """Compute coefficients given a space discretization."""
        if not isinstance(ds, float):
            raise TypeError("Input ds must be a float")
        if ds <= 0.0:
            raise ValueError("Input ds must be positive")
        a1, b1, c1, d1 = self.t1
        a2, b2, c2, d2 = self.t2

        # order 2 (centered)
        A1 = a1 - 2.0 * c1 / ds**2
        A2 = a2 - 2.0 * c2 / ds**2
        B1 = 0.5 * self.pp * b1 / ds + c1 / ds**2
        B2 = 0.5 * self.pp * b2 / ds + c2 / ds**2
        C1 = 0.5 * self.pp * b1 / ds - c1 / ds**2
        C2 = 0.5 * self.pp * b2 / ds - c2 / ds**2

        det = np.linalg.det(np.array([[A1, B1], [A2, B2]]))
        if det == 0.0:
            raise ValueError("Matrix is singular")
        self.c1c = (B2 * d1 - B1 * d2) / det
        self.c2c = (A1 * d2 - A2 * d1) / det

        self.c1q = (B2 * C1 - B1 * C2) / det
        self.c2q = (A1 * C2 - A2 * C1) / det


def rot_free(pos, y=0.0, d2y=0.0):
    """Get boundary condition with free derivative and constrained value and curvature."""
    return BoundaryCondition(t1=(1.0, 0.0, 0.0, y), t2=(0.0, 0.0, 1.0, d2y), pos=pos)


def rot_none(pos, y=0.0, dy=0.0):
    """Get boundary condition with free curvature and constrained value and derivative."""
    return BoundaryCondition(t1=(1.0, 0.0, 0.0, y), t2=(0.0, 1.0, 0.0, dy), pos=pos)


def d2M_cst(
    n: int, ds: float, bcl: BoundaryCondition, bcr: BoundaryCondition
) -> Tuple[scipy.sparse.dia_matrix, np.ndarray]:
    """Get finite difference matrix for second-order derivative on a uniform discretization.

    Compute matrix and boundary condition vector for finite-difference space
    second derivative computation: return values are a matrix 'A' and a vector
    'a' such that d2y/dx**2 = A * y + a

    Parameters
    ----------
    n : int
        Matrix size.
    ds : float
        Discretization step.
    bcl : slenderpy.fdm_utils.BoundaryCondition
        Left boundary condition.
    bcr : slenderpy.fdm_utils.BoundaryCondition
        Right boundary condition.

    Returns
    -------
    scipy.sparse.dia_matrix
        Derivative matrix.
    numpy.ndarray
        Boundary condition vector.
    """
    bcl.set(ds)
    bcr.set(ds)

    dinf = +1.0 * np.ones((n - 1,))
    diag = -2.0 * np.ones((n - 0,))
    dsup = +1.0 * np.ones((n - 1,))

    diag[0] += bcl.c1q
    diag[-1] += bcr.c1q

    A = sp.sparse.diags([dinf, diag, dsup], [-1, 0, 1])
    a = np.concatenate(([bcl.c1c], np.zeros((n - 2)), [bcr.c1c]))

    return A / ds**2, a / ds**2


def d4M_cst(
    n: int, ds: float, bcl: BoundaryCondition, bcr: BoundaryCondition
) -> Tuple[scipy.sparse.dia_matrix, np.ndarray]:
    """Get finite difference matrix for fourth-order derivative on a uniform discretization.

    Compute matrix and boundary condition vector for finite-difference space
    fourth derivative computation: return values are a matrix 'A' and a vector
    'a' such that d4y/dx**4 = A * y + a

    Parameters
    ----------
    n : int
        Matrix size.
    ds : float
        Discretization step.
    bcl : slenderpy.fdm_utils.BoundaryCondition
        Left boundary condition.
    bcr : slenderpy.fdm_utils.BoundaryCondition
        Right boundary condition.

    Returns
    -------
    scipy.sparse.dia_matrix
        Derivative matrix.
    numpy.ndarray
        Boundary condition vector.
    """
    bcl.set(ds)
    bcr.set(ds)

    dm2 = +1.0 * np.ones((n - 2,))
    dm1 = -4.0 * np.ones((n - 1,))
    d0 = +6.0 * np.ones((n - 0,))
    dp1 = -4.0 * np.ones((n - 1,))
    dp2 = +1.0 * np.ones((n - 2,))

    d0[0] += 1.0 * bcl.c2q - 4.0 * bcl.c1q
    d0[-1] += 1.0 * bcr.c2q - 4.0 * bcr.c1q
    dm1[0] += 1.0 * bcl.c1q
    dp1[-1] += 1.0 * bcr.c1q

    D = sp.sparse.diags([dm2, dm1, d0, dp1, dp2], [-2, -1, 0, 1, 2])
    d = np.concatenate(
        (
            [1.0 * bcl.c2c - 4.0 * bcl.c1c, 1.0 * bcl.c1c],
            np.zeros((n - 4)),
            [1.0 * bcr.c1c, 1.0 * bcr.c2c - 4.0 * bcr.c1c],
        )
    )
    return D / ds**4, d / ds**4


def d2M(ds: np.ndarray) -> scipy.sparse.dia_matrix:
    """Get finite difference matrix for second-order derivative.

    Compute matrix for finite-difference space second derivative on a general
    grid with Dirichlet boundary conditions (in this case the boundary condition
    vector is zero).

    Parameters
    ----------
    ds : numpy.ndarray
        Array of grid steps. If it has a size N the output matrix will have a
        size of N-1.

    Returns
    -------
    scipy.sparse.dia_matrix
        Derivative matrix.
    """
    h1 = ds[:-1]
    h2 = ds[1:]
    dinf = +2.0 / (h1 * (h1 + h2))
    dsup = +2.0 / (h2 * (h1 + h2))
    diag = -1.0 * (dinf + dsup)
    return sp.sparse.diags([dinf[1:], diag, dsup[:-1]], [-1, 0, 1])


def d1M(ds: np.ndarray) -> scipy.sparse.dia_matrix:
    """Get finite difference matrix for first-order derivative.

    Compute matrix for finite-difference space first derivative on a general
    grid with Dirichlet boundary conditions (in this case the boundary condition
    vector is zero).

    Parameters
    ----------
    ds : numpy.ndarray
        Array of grid steps. If it has a size N the output matrix will have a
        size of N-1.

    Returns
    -------
    scipy.sparse.dia_matrix
        Derivative matrix.
    """
    h1 = ds[:-1]
    h2 = ds[1:]
    dinf = -h2 / (h1 * (h1 + h2))
    dsup = +h1 / (h2 * (h1 + h2))
    diag = -1.0 * (dinf + dsup)
    return sp.sparse.diags(
        [
            np.concatenate((dinf, [-1.0 / h2[-1]])),
            np.concatenate(([-1.0 / h1[0]], diag, [+1.0 / h2[-1]])),
            np.concatenate(([+1.0 / h1[0]], dsup)),
        ],
        [-1, 0, 1],
    )
