"""Static shape (deflection) of a beam under a custom right-hand-side force.

The solver is a free function taking the ``Conductor`` and ``Span`` dataclasses
from :mod:`slenderpy.future.components`. The two constitutive ingredients come
from their own modules and are chosen independently:

    - the bending law, constant or Bouc-Wen, from
      :mod:`slenderpy.future.beam.bending`;
    - the curvature model, approximate or exact, from
      :mod:`slenderpy.future.beam.curvature`.

The nonlinear system is solved with a damped Newton iteration on the analytic
Jacobian assembled from ``law.tangent`` and ``chi.jacobian``. A
finite-difference Jacobian stalls on low-tension spans, where the Bouc-Wen
moment varies over a curvature scale ``chi_bar`` orders of magnitude below the
curvature itself. A solve that does not reach the requested tolerance returns
``nan`` rather than its last iterate, so a failed solve cannot be mistaken for
a solution downstream.
"""

from __future__ import annotations

import numpy as np
import scipy as sp

import slenderpy.future.beam.bending as bending
import slenderpy.future.beam.curvature as curvature
import slenderpy.future.fd_utils as fdu
from slenderpy.future.beam.bending import BendingModel
from slenderpy.future.components import Conductor, Span

__all__ = ["BendingModel", "solve"]

# smallest Newton relaxation factor tried before declaring the step useless
_MIN_RELAXATION = 1.0e-06


def _solve(
    length: float,
    tension: float,
    bc,
    law: bending.Bending,
    rhs: np.ndarray,
    n: int,
    approx_curvature: bool,
    tol: float = 1.0e-06,
    max_iter: int = 64,
) -> np.ndarray:
    """Shared static-solve core: finite-difference assembly then damped Newton.

    Solves ``(d^2/dx^2) M - tension * (d^2/dx^2) y = rhs``, where ``M`` is the
    bending moment ``law.moment(chi.value(y))``. The Newton Jacobian is
    assembled analytically as ``D2 @ diag(law.tangent) @ chi.jacobian + linear
    part``, and each step is relaxed until it decreases the residual norm.

    Returns the displacement, or an array of ``nan`` when ``max|residual|`` does
    not reach ``tol`` relative to the load level within ``max_iter`` iterations.
    """
    ds = length / (n - 1)
    order = bc.order
    D2 = fdu.clean_matrix(order, fdu.second_derivative(n, ds))
    D4 = fdu.fourth_derivative(n, ds)
    BC, rhs_bc = bc.compute(n, ds)
    rhs_tot = fdu.clean_rhs(order, rhs) + rhs_bc

    # linear part of the equation, and constant-stiffness solution as first guess
    linear = -tension * D2 + BC
    y = sp.sparse.linalg.spsolve(law.ei_linear * D4 + linear, rhs_tot)

    chi = curvature.create(n, ds, approx_curvature)

    def residual(y):
        return D2 @ law.moment(chi.value(y)) + linear @ y - rhs_tot

    threshold = tol * np.abs(rhs_tot).max()
    res = residual(y)

    for _ in range(max_iter):
        if np.abs(res).max() <= threshold:
            return y

        tangent = sp.sparse.diags(law.tangent(chi.value(y)))
        jacobian = sp.sparse.csr_matrix(D2 @ tangent @ chi.jacobian(y) + linear)
        step = sp.sparse.linalg.spsolve(jacobian, -res)

        # backtrack until the step actually decreases the residual norm
        relaxation = 1.0
        res_new = residual(y + step)
        while (
            np.linalg.norm(res_new) >= np.linalg.norm(res)
            and relaxation > _MIN_RELAXATION
        ):
            relaxation *= 0.5
            res_new = residual(y + relaxation * step)

        if np.linalg.norm(res_new) >= np.linalg.norm(res):
            break

        y = y + relaxation * step
        res = res_new

    print(
        f"static solve did not converge: max|residual| = {np.abs(res).max():.3e} "
        f"for a target of {threshold:.3e}"
    )
    return np.full(n, np.nan)


def solve(
    conductor: Conductor,
    span: Span,
    rhs: np.ndarray,
    n: int,
    model: BendingModel = BendingModel.CONSTANT,
    ei: float | None = None,
    approx_curvature: bool = True,
    tol: float = 1.0e-06,
    max_iter: int = 64,
) -> np.ndarray:
    """Compute the static displacement (shape) of a beam under a nodal force.

    Parameters
    ----------
    conductor : Conductor
        Conductor properties. The bending-stiffness fields are read according to
        ``model``; ``mass`` is not used by the static solve.
    span : Span
        Span geometry and loading. ``length``, ``tension`` and
        ``boundary_conditions`` are used; ``boundary_conditions`` must be set.
    rhs : np.ndarray
        Nodal force values, a 1-D array of length ``n``.
    n : int
        Number of nodes.
    model : BendingModel, optional
        ``CONSTANT`` or ``VARYING``; accepts the enum member or its string value.
        Default ``CONSTANT``.
    ei : float or None, optional
        Constant model only: overrides the bending stiffness. When ``None``
        (default), ``conductor.ei_max`` is used.
    approx_curvature : bool, optional
        ``True`` (default) uses the approximate curvature ``D2 @ y``; ``False``
        uses the exact geometric curvature.
    tol : float, optional
        Convergence threshold on ``max|residual|``, relative to the load level.
        Default 1e-06.
    max_iter : int, optional
        Maximum number of Newton iterations. Default 64.

    Returns
    -------
    np.ndarray
        Displacement at the ``n`` nodes, or an array of ``nan`` if the solve did
        not converge.

    Raises
    ------
    ValueError
        If boundary conditions are missing, ``rhs`` has the wrong length, or the
        bending-stiffness parameters required by ``model`` are not set.
    """
    if span.boundary_conditions is None:
        raise ValueError("span.boundary_conditions is required for a beam solve")

    rhs = np.asarray(rhs)
    if rhs.shape != (n,):
        raise ValueError(f"rhs must have length {n}, got shape {rhs.shape}")

    law = bending.create(conductor, span, model, ei)

    return _solve(
        span.length,
        span.tension,
        span.boundary_conditions,
        law,
        rhs,
        n,
        approx_curvature,
        tol=tol,
        max_iter=max_iter,
    )
