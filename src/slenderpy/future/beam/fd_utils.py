import numpy as np
import scipy as sp


def first_derivative(n: int, ds: float) -> sp.sparse.dia_matrix:
    """Centered scheme, the first and last line have to be completed with BC (order 2).

    Parameters
    ----------
    n : int
       Matrix size.
    ds : float
        Space discretization step.

    Returns
    -------
    sp.sparse.dia_matrix
        Derivative matrix.
    """
    dinf = -1.0 * np.ones((n - 1,)) / (2 * ds)
    dsup = +1.0 * np.ones((n - 1,)) / (2 * ds)

    dinf[-1] = 0
    dsup[0] = 0

    res = sp.sparse.diags([dinf, dsup], [-1, 1])

    return res


def second_derivative(n: int, ds: float) -> sp.sparse.dia_matrix:
    """Centered scheme, the first and last line have to be completed with BC (order 2).

    Parameters
    ----------
    n : int
        Matrix size.
    ds : float
        Space discretization step.

    Returns
    -------
    sp.sparse.dia_matrix
        Derivative matrix.
    """
    dinf = +1.0 * np.ones((n - 1,)) / ds**2
    diag = -2.0 * np.ones((n,)) / ds**2
    dsup = +1.0 * np.ones((n - 1,)) / ds**2

    dinf[-1] = 0
    diag[0] = 0
    diag[-1] = 0
    dsup[0] = 0

    res = sp.sparse.diags([dinf, diag, dsup], [-1, 0, 1])

    return res


def fourth_derivative(n: int, ds: float) -> sp.sparse.dia_matrix:
    """Centered scheme, the two first and two last line have to be completed with BC (order 4).

    Parameters
    ----------
    n : int
        Matrix size.
    ds : float
        Space discretization step.

    Returns
    -------
    sp.sparse.dia_matrix
        Derivative matrix.
    """
    dinf2 = +1.0 * np.ones((n - 2)) / ds**4
    dinf1 = -4.0 * np.ones((n - 1,)) / ds**4
    diag = +6.0 * np.ones((n,)) / ds**4
    dsup1 = -4.0 * np.ones((n - 1,)) / ds**4
    dsup2 = +1.0 * np.ones((n - 2,)) / ds**4

    dinf2[[-1, -2]] = [0, 0]
    dinf1[[-1, -2, 0]] = [0, 0, 0]
    diag[[0, 1, -2, -1]] = [0, 0, 0, 0]
    dsup1[[0, 1, -1]] = [0, 0, 0]
    dsup2[[0, 1]] = 0

    res = sp.sparse.diags([dinf2, dinf1, diag, dsup1, dsup2], [-2, -1, 0, 1, 2])

    return res


def clean_matrix(order: int, A: sp.sparse.spmatrix) -> sp.sparse.csr_matrix:
    """Earase the proper coefficients in the scheme matrix to take into account the boundary conditions.

    Parameters
    ----------
    order : int
        Number of boundary conditions (2 or 4).
    A : sp.sparse.spmatrix
        Matrix to clean.

    Returns
    -------
    sp.sparse.csr_matrix
        Cleaned matrix.

    Raises
    ------
    ValueError
        If order different than 2 or 4.
    """
    if order not in (2, 4):
        raise ValueError("order must be 2 or 4")

    if order == 4:
        A = sp.sparse.csr_matrix.copy(A)

        if A.data.shape[0] == 1:
            A.data[0, 0] = 0
            A.data[0, 1] = 0
            A.data[0, -1] = 0
            A.data[0, -2] = 0

        else:
            A.data[0, 0] = 0
            A.data[0, -3] = 0

            A.data[1, 1] = 0
            A.data[1, -2] = 0

            A.data[2, -1] = 0
            A.data[2, 2] = 0

    return A


def clean_rhs(order: int, rhs: np.ndarray[float]) -> np.ndarray[float]:
    """Earase the proper coefficients in the right-hand side to take into account the boundary conditions.

    Parameters
    ----------
    order : int
        Number of boundary conditions (2 or 4).
    rhs : np.ndarray[float]
        Right-hand side to clean.

    Returns
    -------
    np.ndarray[float]
        Cleaned right-hand side.

    Raises
    ------
    ValueError
        If order different than 2 or 4.
    """
    if order not in (2, 4):
        raise ValueError("order must be 2 or 4")

    rhs = np.copy(rhs)

    rhs[0] = 0
    rhs[-1] = 0

    if order == 4:
        rhs[1] = 0
        rhs[-2] = 0

    return rhs
