"""Finite differences utility functions."""

from typing import Tuple, Optional

import numpy as np
import scipy as sp
import scipy.sparse


class BoundaryCondition:
    """Object to deal with boundary conditions."""

    def __init__(self,
                 t1: Optional[Tuple[float, float, float, float]] = None,
                 t2: Optional[Tuple[float, float, float, float]] = None,
                 pos: Optional[str] = None) -> None:
        """Init with args.

        Input t1 and t2 are tuples with four floats each such that
        ti = (ai, bi, ci, di) for i in {1, 2} and:
            a1 * y(b) + b1 * (dy/dx)(b) * c1 * (d2y/dx2)(b) = d1
            a2 * y(b) + b2 * (dy/dx)(b) * c2 * (d2y/dx2)(b) = d

        b is the bound defined by pos (min for left, max for right). The
        equation system defined by both tuples shoud have a nonzero det.

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
            t1 = (1., 0., 0., 0.)
        if t2 is None:
            t2 = (0., 1., 0., 0.)
        if (not isinstance(t1, (tuple, list))) or (not isinstance(t2, (tuple, list))):
            raise TypeError('Inputs t1 and t2 must be list or tuples')
        if len(t1) != 4 or len(t2) != 4:
            raise ValueError('Inputs t1 and t2 must have 4 elements')
        if pos == 'left':
            self.pp = -1.
        elif pos == 'right':
            self.pp = +1.
        else:
            raise ValueError('Input pos must be either \'left\' or \'right\'')
        self.t1 = t1
        self.t2 = t2
        self.c1c = None
        self.c1q = None
        self.c2c = None
        self.c2q = None

    def set(self, ds):
        """Compute coefficients given a space discretization."""
        if not isinstance(ds, float):
            raise TypeError('Input ds must be a float')
        if ds <= 0.:
            raise ValueError('Input ds must be positive')
        a1, b1, c1, d1 = self.t1
        a2, b2, c2, d2 = self.t2

        # order 2 (centered)
        A1 = a1 - 2. * c1 / ds**2
        A2 = a2 - 2. * c2 / ds**2
        B1 = 0.5 * self.pp * b1 / ds + c1 / ds**2
        B2 = 0.5 * self.pp * b2 / ds + c2 / ds**2
        C1 = 0.5 * self.pp * b1 / ds - c1 / ds**2
        C2 = 0.5 * self.pp * b2 / ds - c2 / ds**2

        det = np.linalg.det(np.array([[A1, B1], [A2, B2]]))
        if det == 0.:
            raise ValueError('Matrix is singular')
        self.c1c = (B2 * d1 - B1 * d2) / det
        self.c1q = (B2 * C1 - B1 * C2) / det
        self.c2c = (A1 * d2 - A2 * d1) / det
        self.c2q = (A1 * C2 - A2 * C1) / det


def rot_free(pos, y=0., d2y=0.):
    """Get boundary condition with free derivative and constrained value and curvature."""
    return BoundaryCondition(t1=(1., 0., 0., y), t2=(0., 0., 1., d2y), pos=pos)


def rot_none(pos, y=0., dy=0.):
    """Get boundary condition with free curvature and constrained value and derivative."""
    return BoundaryCondition(t1=(1., 0., 0., y), t2=(0., 1., 0., dy), pos=pos)


def d2M_cst(n: int,
            ds: float,
            bcl: BoundaryCondition,
            bcr: BoundaryCondition) \
        -> Tuple[scipy.sparse.dia.dia_matrix, np.ndarray]:
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
    bcl : structvib.fdm_utils.BoundaryCondition
        Left boundary condition.
    bcr : structvib.fdm_utils.BoundaryCondition
        Right boundary condition.

    Returns
    -------
    scipy.sparse.dia.dia_matrix
        Derivative matrix.
    numpy.ndarray
        Boundary condition vector.
    """
    bcl.set(ds)
    bcr.set(ds)

    dinf = +1. * np.ones((n - 1,))
    diag = -2. * np.ones((n - 0,))
    dsup = +1. * np.ones((n - 1,))

    diag[0] += bcl.c1q
    diag[-1] += bcr.c1q

    A = sp.sparse.diags([dinf, diag, dsup], [-1, 0, 1])
    a = np.concatenate(([bcl.c1c], np.zeros((n - 2)), [bcr.c1c]))

    return A / ds**2, a / ds**2


def d4M_cst(n: int,
            ds: float,
            bcl: BoundaryCondition,
            bcr: BoundaryCondition) \
        -> Tuple[scipy.sparse.dia.dia_matrix, np.ndarray]:
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
    bcl : structvib.fdm_utils.BoundaryCondition
        Left boundary condition.
    bcr : structvib.fdm_utils.BoundaryCondition
        Right boundary condition.

    Returns
    -------
    scipy.sparse.dia.dia_matrix
        Derivative matrix.
    numpy.ndarray
        Boundary condition vector.
    """
    bcl.set(ds)
    bcr.set(ds)

    dm2 = +1. * np.ones((n - 2,))
    dm1 = -4. * np.ones((n - 1,))
    d0 = +6. * np.ones((n - 0,))
    dp1 = -4. * np.ones((n - 1,))
    dp2 = +1. * np.ones((n - 2,))

    d0[0] += (1. * bcl.c2q - 4. * bcl.c1q)
    d0[-1] += (1. * bcr.c2q - 4. * bcr.c1q)
    dm1[0] += (1. * bcl.c1q)
    dp1[-1] += (1. * bcr.c1q)

    D = sp.sparse.diags([dm2, dm1, d0, dp1, dp2], [-2, -1, 0, 1, 2])
    d = np.concatenate(([1. * bcl.c2c - 4. * bcl.c1c,
                         1. * bcl.c1c],
                        np.zeros((n - 4)),
                        [1. * bcr.c1c,
                         1. * bcr.c2c - 4. * bcr.c1c]))
    return D / ds**4, d / ds**4


def d2M(ds: np.ndarray) -> scipy.sparse.dia.dia_matrix:
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
    scipy.sparse.dia.dia_matrix
        Derivative matrix.
    """
    h1 = ds[:-1]
    h2 = ds[1:]
    dinf = +2. / (h1 * (h1 + h2))
    dsup = +2. / (h2 * (h1 + h2))
    diag = -1. * (dinf + dsup)
    return sp.sparse.diags([dinf[1:], diag, dsup[:-1]], [-1, 0, 1])


def d1M(ds: np.ndarray) -> scipy.sparse.dia.dia_matrix:
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
    scipy.sparse.dia.dia_matrix
        Derivative matrix.
    """
    h1 = ds[:-1]
    h2 = ds[1:]
    dinf = -h2 / (h1 * (h1 + h2))
    dsup = +h1 / (h2 * (h1 + h2))
    diag = -1. * (dinf + dsup)
    return sp.sparse.diags([np.concatenate((dinf, [-1. / h2[-1]])),
                            np.concatenate(([-1. / h1[0]], diag, [+1. / h2[-1]])),
                            np.concatenate(([+1. / h1[0]], dsup))
                            ], [-1, 0, 1])
