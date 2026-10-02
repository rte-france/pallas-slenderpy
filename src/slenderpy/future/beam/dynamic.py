"""Dynamic (time-domain) response of a beam under a custom force.

Companion of :mod:`slenderpy.future.beam.static.shape`: same ``Conductor`` and
``Span`` inputs, the same bending law from
:mod:`slenderpy.future.beam.bending` and the same curvature model from
:mod:`slenderpy.future.beam.curvature`, with one solver covering the four
(model, curvature) combinations. The banded-storage helpers the Newton tangent
is assembled with live in :mod:`slenderpy.future.fd_utils`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy as sp

import slenderpy.future.beam.bending as bending
import slenderpy.future.beam.curvature as curvature
import slenderpy.future.fd_utils as fdu
import slenderpy.future.simulation as simulation
from slenderpy import _progress_bar as spb
from slenderpy.future.beam.bending import BendingModel
from slenderpy.future.beam.static import shape
from slenderpy.future.components import Conductor, Span

# smallest newton relaxation factor tried before declaring the step useless
_MIN_RELAXATION = 1.0e-06


def solve_dynamic(
    conductor: Conductor,
    span: Span,
    parameters: simulation.Parameters,
    model: BendingModel = BendingModel.CONSTANT,
    ei: float | None = None,
    force: callable | None = None,
    approx_curvature: bool = True,
    initial_position: np.ndarray[float] | None = None,
    initial_velocity: np.ndarray[float] | None = None,
    initial_bending_moment: Optional[np.ndarray[float]] = None,
    zeta: Optional[float] = 0,
    tol: float = 1.0e-06,
    max_iter: int = 64,
) -> simulation.Results:
    """Dynamic solver for beam equations (either 1-2-3 or 1-2-3bis-4) :
    (1) mass*(d^2/dt^2)*y + 2*mass*w0*zeta*(d/dt)*y + (d^2/dx^2)*M - tension*(d^2/dx^2)*y = force
    (2) chi = curvature(y) -- can be either approximate, (d^2/dx^2)*y or exact

    (3) M = ei*chi if the bending moment model is constant

    (3bis) M = ei_min*chi + (ei_max-ei_min)*chi0*eta
    (4) chi0 * (d/dt)*eta = (d/dt)*chi - 1/2 * ((d/dt)*chi * abs(eta) + abs((d/dt)*chi) * eta)

    Time integration is Crank-Nicolson on the first-order system, solved for the
    velocity:

        A @ v(n+1) = B @ v(n) - dt*K @ y(n) - dt/2*(G(n) + G(n+1)) + dt/2*(f(n) + f(n+1))
        y(n+1) = y(n) + dt/2*(v(n) + v(n+1))

    with ``K = ei_linear*D4 - tension*D2`` the linear part of the equation,
    ``A = mass*I + dt/2*damp*I + (dt/2)**2*K + BC``, ``B`` the same with the two
    last signs flipped, and ``G = D2 @ M - ei_linear*D4 @ y`` the nonlinear
    remainder (hysteresis and, for the exact curvature, geometry). The split
    between ``K`` and ``G`` is purely algebraic: the step residual is independent
    of ``ei_linear``.

    ``G`` vanishes identically for a constant model with the approximate
    curvature. That case is linear, and since ``A`` does not depend on the state
    it is factorised once and each step costs a single triangular solve. The
    three other cases are solved with a Newton iteration on the step residual,
    stopped on ``max|residual| <= tol * scale`` with ``scale`` the largest term of
    that residual, using the tangent

        d(residual)/dv = A + (dt/2)**2 * (D2 @ diag(dM/dchi) @ dchi/dy
                                          - ei_linear*D4)

    where ``dM/dchi`` is ``ei`` for the constant model and, for the varying one,
    the tangent of (3bis)-(4) over the step -- ``ei_max`` on the stiff branch,
    down to ``ei_min`` once the hysteresis saturates -- with its hysteretic part
    counted twice, since the ``eta`` path reaches ``v`` through ``dt`` where the
    curvature path reaches it through ``dt/2``. Iterating instead on a
    fixed ``A`` (Picard, as slenderpy does) does not converge here: with
    ``ei_linear = ei_min`` in ``A`` and a true tangent of ``ei_max``, the fixed
    point has a gain of ``ei_max/ei_min``, about 76 for an ASTER 570.

    Three deliberate differences with the slenderpy solvers:

    - the iteration is Newton on the exact tangent rather than a fixed-point
      iteration, for the reason above;
    - ``d(chi)/dt`` in (4) is evaluated as ``d(chi)/dy @ v(n+1)`` in both
      curvature options, i.e. fully implicit like the ``eta`` update it feeds.
    - a step whose Newton iteration does not converge within ``max_iter`` is
      re-solved once with the Bouc-Wen branch of every node lagged from the
      previous step, a smooth problem, then ``eta`` is recomputed on the law's
      own branch so it stays bounded by 1. This only happens at a few reversal
      steps, where Newton straddling the kink of the law can stall depending on
      round-off; the error there is first order. If the fallback fails too, the
      run stops: the snapshots already computed are kept, the remaining ones are
      left at nan and no final state is recorded. A non-finite state counts as a
      failure, so nan never passes for a converged step.

    Parameters
    ----------
    conductor : Conductor
        Conductor properties. ``mass`` and the bending-stiffness fields required
        by ``model`` must be set.
    span : Span
        Span geometry and loading; ``boundary_conditions`` must be set. For the
        varying model the critical curvature is ``beta_flexion * tension``.
        ``span.sld`` is ignored: the end heights are the boundary-condition
        values.
    parameters : simulation.Parameters
        Simulation parameters. ``ns`` sets the space discretisation, ``t0``,
        ``tf`` and the derived ``nt`` the time stepping, ``nr``/``rr`` the output
        rate, ``los`` the positions of interest and ``pp`` the progress bar. The
        solve itself always runs on the ``ns`` nodes; ``los`` only selects what
        is stored, by interpolation of the nodal fields, so a field read back
        from the result is a linear interpolation and no longer satisfies (2) to
        (4) exactly unless its position falls on a node. ``los`` may hold the
        supports 0 and 1, which are nodes.
    model : BendingModel, optional
        ``CONSTANT`` or ``VARYING``; accepts the enum member or its string value.
        Default ``CONSTANT``.
    ei : float or None, optional
        Constant model only: overrides the constant bending stiffness. When
        ``None`` (default), ``conductor.ei_max`` is used.
    force : callable, optional
        ``force(x, t, y, z, vy, vz) -> (fy, fz)``, the interface of
        :mod:`slenderpy.future.force.core`, e.g.
        ``Gravity(conductor.mass) + PointExcitation(...)``. The beam is
        planar: it is called with ``y = vy = 0`` and only ``fz`` (N/m, may be
        a scalar) is used. Default a null force. It is evaluated at the state
        of the previous step, so a state-dependent force is lagged by one
        step.
    approx_curvature : bool, optional
        ``True`` (default) uses the approximate curvature ``D2 @ y``; ``False``
        uses the exact geometric curvature.
    initial_position : np.ndarray, optional
        Initial position. Default the static shape under ``force`` at ``t0``,
        from :func:`shape.solve` with the same model and curvature option.
    initial_velocity : np.ndarray, optional
        Initial velocity. Default at rest.
    initial_bending_moment : np.ndarray, optional
        Initial bending moment, used by the varying model to initialise the
        hysteresis variable. Default the static law at the initial curvature.
    zeta : float, optional
        Damping ratio, by default 0. The damping coefficient is
        ``2*mass*(2*pi*f0)*zeta`` with ``f0`` the first taut-string frequency.
    tol : float, optional
        Convergence threshold on ``max|step residual|``, relative to the largest
        term of that residual. Default 1e-06.
    max_iter : int, optional
        Maximum number of Newton iterations per step, and per fallback
        solve. Default 64.

    Returns
    -------
    simulation.Results
        Vertical position ``z``, vertical velocity ``vz``, ``curvature``,
        bending ``moment`` and hysteresis variable ``eta`` (zero for the
        constant model) at the positions of ``parameters.los``, and the
        per-step iteration count ``n_iter``; names follow the output contract
        of :mod:`slenderpy.future.simulation`. Snapshots after a failed step are
        left at nan. The final state, recorded with
        :meth:`simulation.Results.set_state`, keeps the same five fields, under
        the same names, at full ``ns`` resolution.
    """
    model = BendingModel(model)

    if span.boundary_conditions is None:
        raise ValueError("span.boundary_conditions is required for a beam solve")

    law = bending.create(conductor, span, model, ei)

    # discretisation
    ns = parameters.ns
    ds = span.length / (ns - 1)
    dt = parameters.dt
    dt2 = 0.5 * dt
    x = np.linspace(0.0, span.length, ns)
    bc = span.boundary_conditions
    order = bc.order

    # operators; fourth_derivative already zeroes its border rows
    D2 = fdu.clean_matrix(order, fdu.second_derivative(ns, ds))
    D4 = fdu.fourth_derivative(ns, ds)
    BC, _ = bc.compute(ns, ds)
    identity = fdu.clean_matrix(order, sp.sparse.identity(ns))
    chi_operator = curvature.create(ns, ds, approx_curvature)

    # crank-nicolson matrices; A is state-independent, factorise it once
    f0 = 0.5 / span.length * np.sqrt(span.tension / conductor.mass)
    damp = 2.0 * conductor.mass * 2.0 * np.pi * f0 * zeta
    ei_D4 = law.ei_linear * D4
    stiffness = ei_D4 - span.tension * D2
    damped_mass = (conductor.mass + dt2 * damp) * identity
    A = damped_mass + dt2**2 * stiffness + BC
    B = 2.0 * conductor.mass * identity - damped_mass - dt2**2 * stiffness
    lu = sp.sparse.linalg.splu(sp.sparse.csc_matrix(A))

    # round-off floor of the step residual: A @ v and dt * K @ y cannot be
    # evaluated better than eps * ||operator|| * |state|, which grows like
    # EI / ds**4 on fine grids; the stiffest tangent of the law bounds K
    stiffest = max(law.ei_linear, float(np.max(law.tangent(np.zeros(1)))))
    stiff = stiffest * D4 - span.tension * D2
    step_norm = fdu.inf_norm(damped_mass + dt2**2 * stiff + BC)
    elastic_norm = dt * fdu.inf_norm(stiff)

    # newton tangent, assembled and solved in banded storage. Its constant part
    # and the left factor of its bending term never change; for the approximate
    # curvature the right factor is the constant D2 as well, so only the tangent
    # stiffness varies from one iteration to the next
    jacobian_base = fdu.banded(A - dt2**2 * ei_D4)
    left_rows = fdu.tridiagonal(D2)
    if approx_curvature:
        constant_rows = fdu.tridiagonal(chi_operator.jacobian(np.zeros(ns)))

        def right_rows(y):
            return constant_rows

    else:

        def right_rows(y):
            return fdu.tridiagonal(chi_operator.jacobian(y))

    # the remainder G is identically zero for a law with no hysteresis
    # taken with the approximate curvature: that case is linear
    linear = not law.hysteretic and approx_curvature

    if force is None:

        def force(x, t, y, z, vy, vz):
            return 0.0, 0.0

    # the beam is planar: y and vy are zero, only fz is used
    zeros = np.zeros(ns)

    def vertical_force(t, z, vz):
        return zeros + force(x, t, zeros, z, zeros, vz)[1]

    # initial state
    if initial_velocity is None:
        initial_velocity = np.zeros(ns)
    if initial_position is None:
        initial_position = shape.solve(
            conductor,
            span,
            vertical_force(parameters.t0, zeros, zeros),
            ns,
            model=model,
            ei=ei,
            approx_curvature=approx_curvature,
            tol=tol,
            max_iter=max_iter,
        )

    y_old = np.asarray(initial_position, dtype=float)
    v_old = np.asarray(initial_velocity, dtype=float)
    if y_old.shape != (ns,) or v_old.shape != (ns,):
        raise ValueError(f"initial position and velocity must have length {ns}")

    chi_old = chi_operator.value(y_old)
    if initial_bending_moment is None:
        initial_bending_moment = law.moment(chi_old)
    eta_old = law.initial_eta(initial_bending_moment, chi_old)

    # output
    lov = ["z", "vz", "curvature", "moment", "eta", "n_iter"]
    res = simulation.Results(
        lot=parameters.time_vector_output().tolist(),
        lov=lov,
        lov_dims=[2, 2, 2, 2, 2, 1],
        los=parameters.los,
    )
    res.start_timer()
    res.update(
        0,
        x / span.length,
        lov,
        [
            y_old,
            v_old,
            chi_old,
            law.dynamic_moment(chi_old, eta_old),
            eta_old,
            0,
        ],
    )

    # time loop
    pb = spb.generate(parameters.pp, parameters.nt, desc=__name__)
    t_old = parameters.t0
    converged = True
    # curvature increment of the previous step, for the lagged branch
    dchi_old = np.zeros(ns)

    for step in range(parameters.nt):
        t_new = t_old + dt
        load = vertical_force(t_old, y_old, v_old) + vertical_force(t_new, y_old, v_old)
        rhs_bc = (
            np.zeros(ns) if bc.dynamic_values is None else bc.update_rhs(ns, x, t_new)
        )
        inertia = B @ v_old
        elastic = dt * stiffness @ y_old
        external = dt2 * fdu.clean_rhs(order, load)
        rhs = inertia - elastic + external + rhs_bc
        threshold = max(
            tol * fdu.residual_scale((inertia, elastic, external, rhs_bc)),
            fdu.round_off_floor([(step_norm, v_old), (elastic_norm, y_old)]),
        )

        if linear:
            v_new = lu.solve(rhs)
            y_new = y_old + dt2 * (v_old + v_new)
            chi_new = chi_operator.value(y_new)
            eta_new = eta_old
            n_iter = 1
        else:
            remainder_old = D2 @ law.dynamic_moment(chi_old, eta_old) - ei_D4 @ y_old

            def step_state(v, branch=None):
                """State and step residual reached by a candidate velocity.

                The law is evaluated on ``branch``, or on its own branch when
                ``branch`` is None.
                """
                y = y_old + dt2 * (v_old + v)
                chi = chi_operator.value(y)
                # curvature increment driving the hysteresis, as the curvature
                # rate at the end of the step times the step: the same fully
                # implicit discretisation the eta update of the law is built on
                dchi = dt * chi_operator.rate(y, v)
                eta = law.update_eta(eta_old, dchi, branch)
                remainder = D2 @ law.dynamic_moment(chi, eta) - ei_D4 @ y
                residual = A @ v - rhs + dt2 * (remainder_old + remainder)
                return y, chi, dchi, eta, residual

            def newton(branch=None):
                """Newton on the step residual, on ``branch`` (default the law's own)."""
                v = v_old
                state = step_state(v, branch)
                error = np.abs(state[-1]).max()
                n_iter = 0
                while n_iter < max_iter and error > threshold:
                    y, _, dchi, eta, step_residual = state
                    # tangent bending stiffness of the law over the step. The
                    # moment reaches the velocity twice: once through the
                    # curvature, whose derivative carries dt2, and once through
                    # eta, whose increment carries dt, i.e. twice as much. Only
                    # the hysteretic part of dynamic_tangent takes the second
                    # path, so it is the only one weighted twice
                    tangent = (
                        2.0 * law.dynamic_tangent(eta, dchi, branch) - law.ei_linear
                    )

                    jacobian = jacobian_base + dt2**2 * fdu.product_band(
                        left_rows, right_rows(y), tangent
                    )
                    try:
                        increment = sp.linalg.solve_banded(
                            (fdu.BANDWIDTH, fdu.BANDWIDTH), jacobian, -step_residual
                        )
                    except np.linalg.LinAlgError:
                        break

                    # backtrack while the increment increases the residual,
                    # because the full newton step can overshoot where the
                    # bouc-wen law is not differentiable, at a sign change of
                    # dchi or eta. Such a sign change also needs a step that
                    # momentarily increases the residual, so a failed backtrack
                    # takes the full step instead of giving up: only max_iter
                    # ends the iteration.
                    relaxation = 1.0
                    trial = step_state(v + increment, branch)
                    while (
                        np.abs(trial[-1]).max() >= error
                        and relaxation > _MIN_RELAXATION
                    ):
                        relaxation *= 0.5
                        trial = step_state(v + relaxation * increment, branch)

                    if np.abs(trial[-1]).max() >= error:
                        relaxation = 1.0
                        trial = step_state(v + increment, branch)

                    v = v + relaxation * increment
                    state = trial
                    error = np.abs(state[-1]).max()
                    n_iter += 1
                return v, state, error, n_iter

            v_new, state, error, n_iter = newton()
            if not (error <= threshold and np.all(np.isfinite(state[0]))):
                # fallback. The law is kinked where a node changes branch (a
                # reversal of dchi, a zero of eta), its tangent jumping by up
                # to ei_max/ei_min, and newton straddling such a kink can
                # stall: round-off decides whether it gets through. The step
                # is then re-solved with the branch lagged from the previous
                # step, a smooth problem, and eta is recomputed on the law's
                # own branch from the converged curvature increment, which
                # keeps it bounded by 1. The error is first order, on these
                # steps only
                v_lag, state_lag, error_lag, n_lag = newton(
                    law.branch(eta_old, dchi_old)
                )
                n_iter += n_lag
                if error_lag <= threshold and np.all(np.isfinite(state_lag[0])):
                    y, chi, dchi, _, step_residual = state_lag
                    state = (y, chi, dchi, law.update_eta(eta_old, dchi), step_residual)
                    v_new, error = v_lag, error_lag
            y_new, chi_new, dchi_new, eta_new, step_residual = state

            if not (error <= threshold and np.all(np.isfinite(y_new))):
                print(
                    f"dynamic solve did not converge at step {step + 1}"
                    f"/{parameters.nt} (t = {t_new:.6g} s): max|residual| = "
                    f"{error:.3e} for a target of {threshold:.3e}"
                )
                converged = False
                break

        if (step + 1) % parameters.rr == 0:
            res.update(
                (step + 1) // parameters.rr,
                x / span.length,
                lov,
                [
                    y_new,
                    v_new,
                    chi_new,
                    law.dynamic_moment(chi_new, eta_new),
                    eta_new,
                    n_iter,
                ],
            )
            pb.update(parameters.rr)

        t_old = t_new
        y_old, v_old, chi_old, eta_old = y_new, v_new, chi_new, eta_new
        if not linear:
            dchi_old = dchi_new

    pb.close()
    res.stop_timer()

    if converged:
        res.set_state(
            {
                "z": y_old,
                "vz": v_old,
                "curvature": chi_old,
                "moment": law.dynamic_moment(chi_old, eta_old),
                "eta": eta_old,
            }
        )

    return res
