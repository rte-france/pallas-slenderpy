"""Dynamic (time-domain) response of a suspended cable under a custom force.

The model is the one of [Lee1992]: the state is the displacement from the
catenary equilibrium, expressed in the local triad of that equilibrium and
non-dimensionalised by the cable length, on a grid uniform in arc length. The
tangential displacement is not a degree of freedom; it follows from the
quasi-static stretching condition.

Ported from :func:`slenderpy.cable.solve`, with the interface moved to global
coordinates: the initial conditions and the results are absolute positions in a
fixed frame, and positions of interest are normalised span positions, as in
:mod:`slenderpy.beam.dynamic`. The frame is the one described at
:func:`slenderpy.cable.tnb2xyz`: ``ex`` horizontal and ``ez`` vertical in the
plane of the two supports and the cable at rest, ``ey`` completing a
right-handed set.

[Lee1992] Lee and Perkins, Nonlinear oscillations of suspended cables containing
a two-to-one internal resonance, Nonlinear Dynamics 3, 1992.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import solve_banded

import slenderpy.simulation as simulation
from slenderpy import _progress_bar as spb
from slenderpy._constant import _GRAVITY
from slenderpy.cable._model import (
    _Geometry,
    _geometry,
    _operators,
    _project,
    _stretching,
    _to_global,
    _to_local,
)
from slenderpy.cable.static import shape
from slenderpy.components import Conductor, Span
from slenderpy.force.core import ForceSum, Gravity

# an end further than this (relative to the span) from its support is an error
_END_TOLERANCE = 1.0e-09


def equilibrium(
    conductor: Conductor, span: Span, ns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global position of the cable nodes at rest under gravity.

    The nodes are the material points the solver tracks: they are evenly spaced
    along the cable, not along the span. Use this to build an initial condition
    without knowing the arc parametrisation::

        x, y, z = equilibrium(conductor, span, parameters.ns)
        initial_position = np.stack([x, y, z + perturbation])

    Parameters
    ----------
    conductor : Conductor
        Conductor properties; only ``mass`` is used.
    span : Span
        Span geometry and loading.
    ns : int
        Number of nodes.

    Returns
    -------
    tuple of numpy.ndarray
        The ``x``, ``y`` and ``z`` coordinates (m) of the ``ns`` nodes. ``y`` is
        zero: the equilibrium lies in the vertical plane of the supports.
    """
    geom = _geometry(conductor, span, ns)
    return geom.x, np.zeros_like(geom.x), geom.z


def _check_position(position: np.ndarray, geom: _Geometry, span: Span) -> np.ndarray:
    """Validate a (3, ns) global array and return it as floats."""
    ns = len(geom.s)
    array = np.asarray(position, dtype=float)
    if array.shape != (3, ns):
        raise ValueError(
            f"initial position and velocity must have shape (3, {ns}), got {array.shape}"
        )
    return array


def _holds_gravity(force) -> bool:
    """Whether a force is, or sums, a Gravity."""
    terms = force.terms if isinstance(force, ForceSum) else (force,)
    return any(isinstance(term, Gravity) for term in terms)


def solve(
    conductor: Conductor,
    span: Span,
    parameters: simulation.Parameters,
    force: callable | None = None,
    zeta: float = 0.0,
    initial_position: np.ndarray | None = None,
    initial_velocity: np.ndarray | None = None,
) -> simulation.Results:
    """Dynamic solver for a suspended cable under an external force.

    The equations of [Lee1992] are solved for the normal and binormal
    displacement from the catenary, non-dimensionalised by the cable length
    ``L``, with times scaled by ``sqrt(L/g)``::

        u_tt = (vt2 + vl2 e) u_ss + vl2 e / vt2 + f     (normal)
        u_tt = (vt2 + vl2 e) u_ss + f                   (binormal)

    with ``vt2 = tension/(mass g L)``, ``vl2 = axial_stiffness/(mass g L)`` and
    ``e`` the spatially uniform dynamic strain, the integral over the cable of
    ``-un/vt2 + (1/2)(un_s^2 + ub_s^2)``. Time integration is Crank-Nicolson on
    the first-order system, solved for the velocity on a tridiagonal operator,
    then ``u(n+1) = u(n) + dt/2 (v(n) + v(n+1))``.

    The grid is uniform in arc length and both ends are fixed at their support.
    The interface, however, is global: the initial conditions and the results are
    absolute positions, and ``parameters.los`` holds normalised span positions.

    Parameters
    ----------
    conductor : Conductor
        Conductor properties. ``mass`` and ``axial_stiffness`` are required.
    span : Span
        Span geometry and loading. ``boundary_conditions`` is ignored; the ends
        are always pinned.
    parameters : simulation.Parameters
        Simulation parameters. ``ns`` sets the space discretisation, ``t0``,
        ``tf`` and the derived ``nt`` the time stepping, ``nr``/``rr`` the output
        rate, ``los`` the normalised span positions to store (supports 0 and 1
        allowed) and ``pp`` the progress bar. The solve always runs on the
        ``ns`` nodes; ``los`` only selects what is stored, by interpolation in
        the span coordinate.
    force : callable, optional
        ``force(x, t, y, z, vy, vz) -> (fy, fz)``, the interface of
        :mod:`slenderpy.force.core`: ``x`` the horizontal position of
        each node (m), ``y``, ``z``, ``vy``, ``vz`` its global position (m)
        and velocity (m/s), and ``(fy, fz)`` the force per unit length (N/m)
        along global ``y`` and ``z`` (scalars allowed). The force is projected
        on the local normal and binormal; its tangential part is dropped.
        The weight must not be included: it is already in the catenary
        equilibrium, so a :class:`~slenderpy.force.core.Gravity`, or
        a sum holding one, raises ``ValueError`` (a plain function cannot be
        checked). Default a null force. It is evaluated at ``t`` and
        ``t+dt`` but both at the state of the previous step, so a
        state-dependent force lags one step. The static solver
        :func:`slenderpy.cable.static.shape.solve` takes its load with
        the same convention.
    zeta : float, optional
        Damping ratio, by default 0. The damping coefficient is ``2 w0 zeta``
        with ``w0`` the first taut-string mode.
    initial_position : numpy.ndarray, optional
        Global ``(3, ns)`` position of the nodes at ``t0``. Default the
        static shape under ``force`` at ``t0``, evaluated at rest, from
        :func:`slenderpy.cable.static.shape.solve`; without force it is
        the catenary equilibrium of :func:`equilibrium`. Both ends must sit on their
        support. The tangential component of the displacement is discarded: the
        model derives it from the stretching condition.
    initial_velocity : numpy.ndarray, optional
        Global ``(3, ns)`` velocity of the nodes at ``t0``. Default at rest.

    Returns
    -------
    simulation.Results
        Global coordinates ``x``, ``y``, ``z`` (m) and ``dtension``, the dynamic
        increment of axial force (N), at the positions of ``parameters.los``.
        ``dtension`` is zero at equilibrium; the total axial force, under the
        model's assumption of a uniform static tension, is
        ``span.tension + dtension``. The final state, recorded with
        :meth:`simulation.Results.set_state`, holds the global ``position`` and
        ``velocity`` at full ``ns`` resolution, so a restart is
        ``initial_position=state["position"]``.

    Warns
    -----
    UserWarning
        If the CFL number exceeds 1, which makes the explicit part of the scheme
        unreliable.
    """
    if conductor.axial_stiffness is None:
        raise ValueError("conductor.axial_stiffness is required for a cable solve")
    if force is not None and _holds_gravity(force):
        raise ValueError(
            "a cable force must not include Gravity: the weight is already in "
            "the catenary equilibrium the model is written around"
        )

    ns = parameters.ns
    geom = _geometry(conductor, span, ns)
    span_fraction = geom.x / span.length

    # scales: lengths by the cable length, times by sqrt(L/g)
    length = geom.length
    time_scale = np.sqrt(length / _GRAVITY)
    speed_scale = length / time_scale
    weight = conductor.mass * _GRAVITY
    vt2 = span.tension / (weight * length)
    vl2 = conductor.axial_stiffness / (weight * length)

    if force is None:

        def force(x, t, y, z, vy, vz):
            return 0.0, 0.0

    if initial_position is None:
        # static shape under the force at t0, evaluated at rest; both solvers
        # take the load on top of the weight
        rest = geom.equilibrium()
        zeros = np.zeros(ns)
        fy0, fz0 = force(geom.x, parameters.t0, rest[1], rest[2], zeros, zeros)
        initial_position = shape.solve(conductor, span, fy0, fz0, ns)
        if not np.all(np.isfinite(initial_position)):
            raise ValueError(
                "no static shape under the force at t0 (non-finite load, or "
                "the static solve did not converge): pass initial_position, "
                "e.g. from cable.static.shape.solve with another tol or max_iter"
            )
    else:
        initial_position = _check_position(initial_position, geom, span)
        ends = initial_position[:, [0, -1]] - geom.equilibrium()[:, [0, -1]]
        if np.abs(ends).max() > _END_TOLERANCE * span.length:
            raise ValueError(
                "initial position must leave both ends on their support "
                f"(largest offset {np.abs(ends).max():.3e} m)"
            )
    if initial_velocity is not None:
        initial_velocity = _check_position(initial_velocity, geom, span)

    un, ub, vn, vb = _to_local(initial_position, initial_velocity, geom)
    un, ub = un / length, ub / length
    vn, vb = vn / speed_scale, vb / speed_scale
    # the ends are pinned; the second derivative acts on the interior only
    un[0], un[-1], ub[0], ub[-1] = 0.0, 0.0, 0.0, 0.0
    vn[0], vn[-1], vb[0], vb[-1] = 0.0, 0.0, 0.0, 0.0

    ds = geom.ds
    first, second = _operators(ns)
    ut, dtension = _stretching(un, ub, first, ds, vt2)

    dt = parameters.dt / time_scale
    ht = 0.5 * dt
    damping = -2.0 * np.pi * np.sqrt(vt2) * zeta * ht

    def local_force(time, position, velocity):
        """Force at the nodes, projected on (en, eb), over the weight."""
        fy, fz = force(geom.x, time, position[1], position[2], velocity[1], velocity[2])
        fn, fb = _project(fy, fz, geom)
        return fn / weight, fb / weight

    lov = ["x", "y", "z", "dtension"]
    res = simulation.Results(
        lot=parameters.time_vector_output().tolist(),
        lov=lov,
        lov_dims=[2, 2, 2, 2],
        los=parameters.los,
    )

    def snapshot(index, ut, un, ub, dtension):
        position = _to_global(ut * length, un * length, ub * length, geom)
        res.update(
            index,
            span_fraction,
            lov,
            [
                position[0],
                position[1],
                position[2],
                dtension * conductor.axial_stiffness,
            ],
        )

    res.start_timer()
    snapshot(0, ut, un, ub, dtension)

    banded = np.zeros((3, ns - 2))
    t = parameters.t0 / time_scale
    warned = False

    pb = spb.generate(parameters.pp, parameters.nt, desc=__name__)
    for step in range(parameters.nt):
        h = -un / vt2 + 0.5 * ((first * un) ** 2 + (first * ub) ** 2)
        strain = 0.5 * np.sum((h[:-1] + h[1:]) * ds)
        wave = vt2 + vl2 * strain

        cfl = np.sqrt(wave) * dt / np.min(ds)
        if cfl > 1.0 and not warned:
            warnings.warn(
                f"CFL number is {cfl:.3e} > 1 at step {step + 1}: increase the "
                "number of time steps or decrease parameters.ns",
                stacklevel=2,
            )
            warned = True

        # global state of the previous step, shared by both evaluations
        position = _to_global(ut * length, un * length, ub * length, geom)
        velocity = (vn * geom.en + vb * geom.eb) * speed_scale
        fn1, fb1 = local_force(t * time_scale, position, velocity)
        fn2, fb2 = local_force((t + dt) * time_scale, position, velocity)

        rhs_n = (
            (dt * wave) * second * (un[1:-1] + 0.5 * ht * vn[1:-1])
            + (1.0 + damping) * vn[1:-1]
            + dt * (0.5 * (fn1[1:-1] + fn2[1:-1]) + vl2 / vt2 * strain)
        )
        rhs_b = (
            (dt * wave) * second * (ub[1:-1] + 0.5 * ht * vb[1:-1])
            + (1.0 + damping) * vb[1:-1]
            + ht * (fb1[1:-1] + fb2[1:-1])
        )

        tau = -(ht**2) * wave
        banded[0, 1:] = tau * second.diagonal(k=1)
        banded[1, :] = 1.0 - damping + tau * second.diagonal(k=0)
        banded[2, :-1] = tau * second.diagonal(k=-1)

        speed = solve_banded((1, 1), banded, np.column_stack((rhs_n, rhs_b)))

        un[1:-1] += ht * (vn[1:-1] + speed[:, 0])
        ub[1:-1] += ht * (vb[1:-1] + speed[:, 1])
        vn[1:-1] = speed[:, 0]
        vb[1:-1] = speed[:, 1]
        t += dt

        ut, dtension = _stretching(un, ub, first, ds, vt2)

        if (step + 1) % parameters.rr == 0:
            snapshot((step + 1) // parameters.rr, ut, un, ub, dtension)
            pb.update(parameters.rr)

    pb.close()
    res.stop_timer()
    res.set_state(
        {
            "position": _to_global(ut * length, un * length, ub * length, geom),
            "velocity": (vn * geom.en + vb * geom.eb) * speed_scale,
        }
    )

    return res
