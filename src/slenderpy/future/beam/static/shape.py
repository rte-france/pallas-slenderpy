"""Static shape (deflection) of a beam under a custom right-hand-side force.

Two bending models are supported: a constant bending stiffness and a varying
(Bouc-Wen) bending stiffness. The solver is a free function taking the
``Conductor`` and ``Span`` dataclasses from :mod:`slenderpy.future.components`.
The constitutive laws are exposed as standalone factories so the dynamic solver
can reuse them.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD
from slenderpy.future.components import Conductor, Span


class BendingModel(str, Enum):
    """Bending-stiffness model used by the static solver."""

    CONSTANT = "constant"
    VARYING = "varying"


def _bending_moment_constant(ei):
    """Return the constant-stiffness law ``M(curvature) = ei * curvature``."""

    def bending_moment(curvature):
        return ei * curvature

    return bending_moment


def _bending_moment_varying(ei_min, ei_max, chi0):
    """Return the Bouc-Wen static bending-moment law as a function of curvature."""
    chi_bar = (1 - ei_min / ei_max) * chi0

    def bending_moment(curvature):
        c = np.abs(curvature)
        return (
            (ei_max * chi_bar + ei_min * c)
            * (1 - np.exp(-c / chi_bar))
            * np.sign(curvature)
        )

    return bending_moment


def _solve(length, tension, bc, ei_linear, bending_moment, rhs, n, approx_curvature):
    """Shared static-solve core: finite-difference assembly then nonlinear root find.

    Solves ``(d^2/dx^2) M - tension * (d^2/dx^2) y = rhs``, where ``M`` is the
    bending moment produced by ``bending_moment(curvature(y))``.
    """
    ds = length / (n - 1)
    order = bc.order
    D2_border = FD.second_derivative(n, ds)
    D2 = FD.clean_matrix(order, D2_border)
    BC, rhs_bc = bc.compute(n, ds)
    D4 = FD.fourth_derivative(n, ds)
    K = ei_linear * D4 - tension * D2
    A = K + BC
    rhs = FD.clean_rhs(order, rhs)
    rhs_tot = rhs + rhs_bc

    sol = sp.sparse.linalg.spsolve(A, rhs_tot)

    if approx_curvature:

        def curvature(y):
            return D2_border @ y

    else:
        D1 = FD.first_derivative(n, ds)

        def curvature(y):
            return D2_border @ y / np.sqrt((1 + (D1 @ y) ** 2) ** 3)

    def equation(y):
        moment = bending_moment(curvature(y))
        return D2 @ moment - tension * D2 @ y + BC @ y - rhs_tot

    result = sp.optimize.root(equation, sol)

    if not result.success:
        print(result.message)

    return result.x


def solve(
    conductor: Conductor,
    span: Span,
    rhs: np.ndarray,
    n: int,
    model: BendingModel = BendingModel.CONSTANT,
    ei: float | None = None,
    approx_curvature: bool = True,
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
        Constant model only: overrides the constant bending stiffness. When
        ``None`` (default), ``conductor.ei_max`` is used.
    approx_curvature : bool, optional
        ``True`` (default) uses the approximate curvature ``D2 @ y``; ``False``
        uses the exact geometric curvature.

    Returns
    -------
    np.ndarray
        Displacement at the ``n`` nodes.

    Raises
    ------
    ValueError
        If boundary conditions are missing, ``rhs`` has the wrong length, or the
        bending-stiffness parameters required by ``model`` are not set.
    """
    model = BendingModel(model)

    if span.boundary_conditions is None:
        raise ValueError("span.boundary_conditions is required for a beam solve")

    rhs = np.asarray(rhs)
    if rhs.shape != (n,):
        raise ValueError(f"rhs must have length {n}, got shape {rhs.shape}")

    if model is BendingModel.CONSTANT:
        ei_used = ei if ei is not None else conductor.ei_max
        if ei_used is None:
            raise ValueError(
                "constant model requires `ei` or `conductor.ei_max` to be set"
            )
        ei_linear = ei_used
        bending_moment = _bending_moment_constant(ei_used)
    else:
        if (
            conductor.ei_min is None
            or conductor.ei_max is None
            or conductor.beta_flexion is None
        ):
            raise ValueError(
                "varying model requires conductor.ei_min, ei_max and "
                "beta_flexion to be set"
            )
        ei_linear = conductor.ei_min
        # The critical curvature depends on the span tension.
        chi0 = conductor.beta_flexion * span.tension
        bending_moment = _bending_moment_varying(
            conductor.ei_min, conductor.ei_max, chi0
        )

    return _solve(
        span.length,
        span.tension,
        span.boundary_conditions,
        ei_linear,
        bending_moment,
        rhs,
        n,
        approx_curvature,
    )
