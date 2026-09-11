"""Dynamic (time-domain) response of a suspended cable under a custom force.

The model is the one of [Lee1992]: the state is the displacement from the
catenary equilibrium, expressed in the local triad of that equilibrium and
non-dimensionalised by the cable length, on a grid uniform in arc length. The
tangential displacement is not a degree of freedom; it follows from the
quasi-static stretching condition.

Ported from :func:`slenderpy.cable.solve`, with the interface moved to global
coordinates: the initial conditions and the results are absolute positions in a
fixed frame, and positions of interest are normalised span positions, as in
:mod:`slenderpy.future.beam.dynamic`. The frame is the one described at
:func:`slenderpy.cable.tnb2xyz`: ``ex`` horizontal and ``ez`` vertical in the
plane of the two supports and the cable at rest, ``ey`` completing a
right-handed set.

[Lee1992] Lee and Perkins, Nonlinear oscillations of suspended cables containing
a two-to-one internal resonance, Nonlinear Dynamics 3, 1992.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

import slenderpy.future.simulation as simulation
from slenderpy import _progress_bar as spb
from slenderpy import fdm_utils as fdmu
from slenderpy.future._constant import _GRAVITY
from slenderpy.future.cable.static import catenary
from slenderpy.future.components import Conductor, Span

# an end further than this (relative to the span) from its support is an error
_END_TOLERANCE = 1.0e-09


@dataclass(frozen=True)
class _Geometry:
    """Catenary equilibrium sampled on a grid uniform in arc length.

    Attributes
    ----------
    s : arc fraction of each node, uniform in [0, 1].
    ds : spacing of ``s``.
    x, z : global horizontal and vertical coordinates of the nodes (m).
    slope : dz/dx of the catenary at each node.
    et, en, eb : tangent, normal and binormal unit vectors, shape (3, ns).
    length : catenary length (m).
    """

    s: np.ndarray
    ds: np.ndarray
    x: np.ndarray
    z: np.ndarray
    slope: np.ndarray
    et: np.ndarray
    en: np.ndarray
    eb: np.ndarray
    length: float

    def equilibrium(self) -> np.ndarray:
        """Global position of the nodes at rest, shape (3, ns)."""
        return np.stack([self.x, np.zeros_like(self.x), self.z])


def _geometry(conductor: Conductor, span: Span, ns: int) -> _Geometry:
    """Build the equilibrium geometry and its local triad."""
    a = catenary._mechparam(span.tension, conductor.mass, g=_GRAVITY)
    length = catenary.length(
        span.length, span.tension, span.sld, conductor.mass, g=_GRAVITY
    )
    # xm is half the catenary offset, ie the abscissa shift of the low point
    xm = 0.5 * (a * catenary._qfactor(length, span.sld) - span.length)

    s = np.linspace(0.0, 1.0, ns)
    x = a * np.arcsinh(length * s / a + np.sinh(xm / a)) - xm
    # the two ends are exact by construction; keep them free of round-off
    x[0], x[-1] = 0.0, span.length
    # catenary.shape is declared floatArrayLike; it is an array for an array x
    z = np.asarray(
        catenary.shape(
            x, span.length, span.tension, span.sld, conductor.mass, g=_GRAVITY
        ),
        dtype=float,
    )

    slope = np.sinh((x + xm) / a)
    norm = np.sqrt(1.0 + slope**2)
    zero = np.zeros_like(x)
    et = np.stack([np.ones_like(x), zero, slope]) / norm
    en = np.stack([-slope, zero, np.ones_like(x)]) / norm
    eb = np.stack([zero, -np.ones_like(x), zero])

    return _Geometry(
        s=s, ds=np.diff(s), x=x, z=z, slope=slope, et=et, en=en, eb=eb, length=length
    )


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


def _to_local(
    position: np.ndarray | None, velocity: np.ndarray | None, geom: _Geometry
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project a global position and velocity onto the local triad.

    The tangential component of the displacement is dropped: the model derives
    the tangential displacement from the stretching condition, so it cannot be
    imposed. Returns dimensional ``(un, ub, vn, vb)``.
    """
    ns = len(geom.s)
    if position is None:
        un = np.zeros(ns)
        ub = np.zeros(ns)
    else:
        offset = position - geom.equilibrium()
        un = (offset * geom.en).sum(axis=0)
        ub = (offset * geom.eb).sum(axis=0)

    if velocity is None:
        vn = np.zeros(ns)
        vb = np.zeros(ns)
    else:
        vn = (velocity * geom.en).sum(axis=0)
        vb = (velocity * geom.eb).sum(axis=0)

    return un, ub, vn, vb


def _to_global(
    ut: np.ndarray, un: np.ndarray, ub: np.ndarray, geom: _Geometry
) -> np.ndarray:
    """Assemble the global position from a dimensional local state, shape (3, ns)."""
    return geom.equilibrium() + ut * geom.et + un * geom.en + ub * geom.eb


def _stretching(un, ub, first, ds, vt2):
    """Tangential offset and axial strain from the quasi-static condition."""
    h = -un / vt2 + 0.5 * ((first * un) ** 2 + (first * ub) ** 2)
    segment = 0.5 * (h[:-1] + h[1:]) * ds
    ut = np.sum(segment) * np.linspace(0.0, 1.0, len(un)) - np.cumsum(
        np.concatenate(([0.0], segment))
    )
    strain = (first * ut) + 0.5 * (
        (first * ut) ** 2 + (first * un) ** 2 + (first * ub) ** 2
    )
    return ut, np.log(np.sqrt(1.0 + 2.0 * strain))


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
        rate, ``los`` the normalised span positions to store and ``pp`` the
        progress bar. The solve always runs on the ``ns`` nodes; ``los`` only
        selects what is stored, by interpolation in the span coordinate.
    force : callable, optional
        ``force(s, t, un, ub, vn, vb) -> (fn, fb)``, the local convention of
        :mod:`slenderpy.wind` and :mod:`slenderpy.force`, so those classes work
        unchanged. ``s`` is the span fraction of each node, ``un``, ``ub``,
        ``vn`` and ``vb`` are dimensional local displacements (m) and velocities
        (m/s), and the two returned arrays are forces per unit length (N/m)
        along the normal and the binormal. Default a null force. It is evaluated
        at ``t`` and ``t+dt`` but both at the state of the previous step, so a
        state-dependent force lags one step.
    zeta : float, optional
        Damping ratio, by default 0. The damping coefficient is ``2 w0 zeta``
        with ``w0`` the first taut-string mode.
    initial_position : numpy.ndarray, optional
        Global ``(3, ns)`` position of the nodes at ``t0``. Default the
        equilibrium, from :func:`equilibrium`. Both ends must sit on their
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

    if initial_position is not None:
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
    # the ends are pinned; d2M assumes zero there
    un[0], un[-1], ub[0], ub[-1] = 0.0, 0.0, 0.0, 0.0
    vn[0], vn[-1], vb[0], vb[-1] = 0.0, 0.0, 0.0, 0.0

    ds = geom.ds
    first = fdmu.d1M(ds)
    second = fdmu.d2M(ds)
    ut, dtension = _stretching(un, ub, first, ds, vt2)

    dt = (parameters.tf - parameters.t0) / parameters.nt / time_scale
    ht = 0.5 * dt
    damping = -2.0 * np.pi * np.sqrt(vt2) * zeta * ht

    if force is None:

        def force(s, t, un, ub, vn, vb):
            return np.zeros_like(s), np.zeros_like(s)

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

        # dimensional state for the force, then back to non-dimensional
        fn1, fb1 = force(
            span_fraction,
            t * time_scale,
            un * length,
            ub * length,
            vn * speed_scale,
            vb * speed_scale,
        )
        fn2, fb2 = force(
            span_fraction,
            (t + dt) * time_scale,
            un * length,
            ub * length,
            vn * speed_scale,
            vb * speed_scale,
        )
        fn1, fb1 = fn1 / weight, fb1 / weight
        fn2, fb2 = fn2 / weight, fb2 / weight

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
