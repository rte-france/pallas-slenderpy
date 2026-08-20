"""Curvature of a beam deflection field, as an operator with its jacobian.

Two models share the same interface: the approximate (small-slope) curvature
``D2 @ y``, and the exact geometric curvature
``(D2 @ y) / (1 + (D1 @ y)**2)**(3/2)``. Both expose :meth:`Curvature.value`
and :meth:`Curvature.jacobian`, the latter being the sparse
``d(curvature)/dy`` used by the Newton iterations of the static and dynamic
solvers. Use :func:`create` to pick one from the ``approx_curvature`` flag the
solvers carry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy as sp

import slenderpy.future.fd_utils as fdu


class Curvature(ABC):
    """Curvature operator on a uniform grid, and its jacobian.

    Subclasses implement :meth:`value` and :meth:`jacobian` for one curvature
    model. The finite-difference matrices are built once, at construction.

    Parameters
    ----------
    n : int
        Number of nodes.
    ds : float
        Space step.

    Attributes
    ----------
    d2 : sp.sparse.dia_matrix
        Second-derivative matrix. Its first and last rows are empty, so the
        curvature vanishes at the end nodes.
    """

    def __init__(self, n: int, ds: float) -> None:
        self.n = n
        self.ds = ds
        self.d2 = fdu.second_derivative(n, ds)

    @abstractmethod
    def value(self, y: np.ndarray) -> np.ndarray:
        """Curvature of the deflection field ``y``.

        Parameters
        ----------
        y : np.ndarray
            Deflection at the ``n`` nodes.

        Returns
        -------
        np.ndarray
            Curvature at the ``n`` nodes.
        """

    @abstractmethod
    def jacobian(self, y: np.ndarray) -> sp.sparse.spmatrix:
        """Derivative of :meth:`value` with respect to ``y``.

        Parameters
        ----------
        y : np.ndarray
            Deflection at the ``n`` nodes.

        Returns
        -------
        sp.sparse.spmatrix
            The ``n`` by ``n`` matrix ``d(curvature)/dy`` at ``y``.
        """


class ApproximateCurvature(Curvature):
    """Small-slope curvature ``D2 @ y``, linear in ``y``."""

    def value(self, y: np.ndarray) -> np.ndarray:
        """Curvature of the deflection field ``y``."""
        return self.d2 @ y

    def jacobian(self, y: np.ndarray) -> sp.sparse.spmatrix:
        """Derivative of :meth:`value` with respect to ``y``, constant here."""
        return self.d2


class ExactCurvature(Curvature):
    """Exact geometric curvature ``(D2 @ y) / (1 + (D1 @ y)**2)**(3/2)``.

    Parameters
    ----------
    n : int
        Number of nodes.
    ds : float
        Space step.

    Attributes
    ----------
    d1 : sp.sparse.dia_matrix
        First-derivative matrix, used for the slope.
    """

    def __init__(self, n: int, ds: float) -> None:
        super().__init__(n, ds)
        self.d1 = fdu.first_derivative(n, ds)

    def value(self, y: np.ndarray) -> np.ndarray:
        """Curvature of the deflection field ``y``."""
        metric = 1.0 + (self.d1 @ y) ** 2
        # metric * sqrt(metric) rather than metric**1.5: a float exponent goes
        # through pow(), several times slower than a square root
        return self.d2 @ y / (metric * np.sqrt(metric))

    def jacobian(self, y: np.ndarray) -> sp.sparse.spmatrix:
        """Derivative of :meth:`value` with respect to ``y``."""
        slope = self.d1 @ y
        metric = 1.0 + slope**2
        # see value() on the square root; the second factor reuses the first
        inv_metric_15 = 1.0 / (metric * np.sqrt(metric))
        inv_metric_25 = inv_metric_15 / metric
        return (
            sp.sparse.diags(inv_metric_15) @ self.d2
            - 3.0 * sp.sparse.diags(slope * (self.d2 @ y) * inv_metric_25) @ self.d1
        )


def create(n: int, ds: float, approx_curvature: bool) -> Curvature:
    """Build the curvature operator selected by ``approx_curvature``.

    Parameters
    ----------
    n : int
        Number of nodes.
    ds : float
        Space step.
    approx_curvature : bool
        ``True`` for :class:`ApproximateCurvature`, ``False`` for
        :class:`ExactCurvature`.

    Returns
    -------
    Curvature
        The selected operator.
    """
    return ApproximateCurvature(n, ds) if approx_curvature else ExactCurvature(n, ds)
