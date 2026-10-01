"""Static shape of a suspended cable under a load.

Solves the equations of :mod:`slenderpy.future.cable.dynamic` ([Lee1992]: the
displacement from the catenary of :mod:`slenderpy.future.cable.static.catenary`
in its local triad) without the time derivatives, on the interior nodes of the
same grid, uniform in arc length::

    0 = b(e) D2 un + (vl2 / vt2) e + fn          b(e) = vt2 + vl2 e
    0 = b(e) D2 ub + fb

with ``e`` the uniform strain, the integral over the cable of
``-un / vt2 + (1/2)(un_s^2 + ub_s^2)``. For a fixed ``e`` both equations are
linear, so the problem reduces to the scalar equation ``e = E(un(e), ub(e))``,
solved by secant.

The load comes on top of the weight, which is part of the model, as for the
``force`` of :func:`slenderpy.future.cable.dynamic.solve`: no load gives the
catenary.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import root_scalar

from slenderpy.future._constant import _GRAVITY
from slenderpy.future.cable import _model
from slenderpy.future.components import Conductor, Span


def solve(
    conductor: Conductor,
    span: Span,
    rhs_y: np.ndarray | float,
    rhs_z: np.ndarray | float,
    ns: int,
    tol: float = 1.0e-12,
    max_iter: int = 64,
) -> np.ndarray:
    """Static position of a cable under a load on top of its weight.

    Parameters
    ----------
    conductor : Conductor
        Conductor properties. ``mass`` and ``axial_stiffness`` are required.
    span : Span
        Span geometry and loading. ``boundary_conditions`` is ignored.
    rhs_y, rhs_z : numpy.ndarray or float
        Load per unit length (N/m) along global ``y`` and ``z`` at the nodes,
        on top of the weight; broadcast to ``(ns,)``. Zero gives the
        catenary. The weight must not be included: it is part of the model,
        so ``rhs_z = -conductor.mass * g`` (e.g. a ``Gravity`` force value)
        doubles it. The end values are irrelevant, the ends being pinned.
    ns : int
        Number of nodes, evenly spaced along the cable, at least 3.
    tol : float, optional
        Convergence threshold on the strain residual, relative to
        ``max(|e|, vt2 / vl2)``. Default 1e-12.
    max_iter : int, optional
        Maximum number of secant iterations. Default 64.

    Returns
    -------
    numpy.ndarray
        Global position of the nodes, shape ``(3, ns)``, rows ``x``, ``y``,
        ``z``, ready to pass as ``initial_position`` to
        :func:`slenderpy.future.cable.dynamic.solve`. All nan when the solve
        fails: no convergence, a non-finite state, or a compressed cable
        (``b(e) <= 0`` at the root, e.g. an upward load equal to the weight).

    Raises
    ------
    ValueError
        If ``axial_stiffness`` is missing, ``ns < 3``, or a load does not
        broadcast to ``(ns,)``.
    """
    if conductor.axial_stiffness is None:
        raise ValueError("conductor.axial_stiffness is required for a cable solve")
    if ns < 3:
        raise ValueError(f"ns must be at least 3, got {ns}")
    try:
        fy = np.broadcast_to(np.asarray(rhs_y, dtype=float), (ns,))
        fz = np.broadcast_to(np.asarray(rhs_z, dtype=float), (ns,))
    except ValueError as error:
        raise ValueError(f"rhs_y and rhs_z must broadcast to ({ns},)") from error

    failed = np.full((3, ns), np.nan)
    geom = _model._geometry(conductor, span, ns)
    length = geom.length
    weight = conductor.mass * _GRAVITY
    vt2 = span.tension / (weight * length)
    vl2 = conductor.axial_stiffness / (weight * length)
    scale = vt2 / vl2

    fn, fb = _model._project(fy, fz, geom)
    fn, fb = fn / weight, fb / weight
    first, second = _model._operators(ns)
    upper, diagonal, lower = (second.diagonal(k) for k in (1, 0, -1))
    banded = np.zeros((3, ns - 2))
    un = np.zeros(ns)
    ub = np.zeros(ns)

    def residual(strain):
        """Strain of the state reached at a given strain, minus that strain."""
        wave = vt2 + vl2 * strain
        banded[0, 1:] = wave * upper
        banded[1, :] = wave * diagonal
        banded[2, :-1] = wave * lower
        rhs = np.column_stack((-(vl2 / vt2 * strain + fn[1:-1]), -fb[1:-1]))
        solution = solve_banded((1, 1), banded, rhs)
        un[1:-1], ub[1:-1] = solution[:, 0], solution[:, 1]
        h = -un / vt2 + 0.5 * ((first @ un) ** 2 + (first @ ub) ** 2)
        return 0.5 * np.sum((h[:-1] + h[1:]) * geom.ds) - strain

    try:
        root = root_scalar(
            residual,
            x0=0.0,
            x1=1.0e-06 * scale,
            method="secant",
            xtol=tol * scale,
            rtol=tol,
            maxiter=max_iter,
        )
        strain = root.root
        error = abs(residual(strain))  # also leaves (un, ub) at the root
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        return failed

    if (
        not root.converged
        or not np.isfinite(strain)
        or error > tol * max(abs(strain), scale)
        or vt2 + vl2 * strain <= 0.0
        or not (np.all(np.isfinite(un)) and np.all(np.isfinite(ub)))
    ):
        return failed

    ut, _ = _model._stretching(un, ub, first, geom.ds, vt2)
    return _model._to_global(ut * length, un * length, ub * length, geom)
