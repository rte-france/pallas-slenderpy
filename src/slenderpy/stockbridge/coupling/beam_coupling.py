from __future__ import annotations

from typing import Optional

import numpy as np
import scipy as sp

import slenderpy.beam.bending as bending
import slenderpy.beam.curvature as curvature
import slenderpy.fd_utils as fdu
from slenderpy import _progress_bar as spb
from slenderpy import simulation
from slenderpy.beam.bending import BendingModel
from slenderpy.beam.static import shape
from slenderpy.components import Conductor, Span
from slenderpy.stockbridge.core.stockbridge import Result

# smallest newton relaxation factor tried before declaring the step useless
_MIN_RELAXATION = 1.0e-06


def solve_dynamic_with_sb(
    stockbridges_dict: dict,
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
) -> tuple[simulation.Results, dict[str, Result]]:
    """Solver for the dynamic coupling of a beam with stockbridge dampers.

    Parameters
    ----------
    stockbridges_dict : dict
        One entry per damper, keyed by its name. Each value is a dictionary
        with the keys ``"stockbridge"`` (the
        [](`~slenderpy.stockbridge.core.stockbridge.Stockbridge`)),
        ``"position"`` (its distance from support 1, m), ``"initial condition
        right"`` and ``"initial condition left"`` (the initial states of its two
        masses).
    conductor : Conductor
        Conductor properties, as for
        [](`~slenderpy.beam.dynamic.solve_dynamic`).
    span : Span
        Span geometry and loading; ``boundary_conditions`` must be set.
    parameters : simulation.Parameters
        Simulation parameters (space discretisation, time stepping, outputs).
    model : BendingModel, optional
        ``CONSTANT`` or ``VARYING``. Default ``CONSTANT``.
    ei : float or None, optional
        Constant model only: overrides ``conductor.ei_max``.
    force : callable, optional
        ``force(x, t, y, z, vy, vz) -> (fy, fz)``, the interface of
        [](`slenderpy.force.core`), as for
        [](`~slenderpy.beam.dynamic.solve_dynamic`): the beam is
        planar, it is called with ``y = vy = 0`` and only ``fz`` (N/m, may be
        a scalar) is used. Default a null force.
    approx_curvature : bool, optional
        ``True`` (default) for the approximate curvature, ``False`` for the
        exact one.
    initial_position : numpy.ndarray, optional
        Initial vertical position of the beam nodes. Default the static shape
        under ``force`` at ``t0``.
    initial_velocity : numpy.ndarray, optional
        Initial vertical velocity of the beam nodes. Default at rest.
    initial_bending_moment : numpy.ndarray, optional
        Initial bending moment, for the varying model. Default the static law
        at the initial curvature.
    zeta : float, optional
        Damping ratio. Default 0.
    tol : float, optional
        Newton convergence threshold, relative to the largest term of the step
        residual. Default 1e-06.
    max_iter : int, optional
        Maximum number of Newton iterations per step. Default 64.

    Returns
    -------
    tuple[simulation.Results, dict[str, Result]]
        The beam results, with the variables of
        [](`~slenderpy.beam.dynamic.solve_dynamic`) (``z``, ``vz``,
        ``curvature``, ``moment``, ``eta``, ``n_iter``), and a dictionary of
        Result objects for each stockbridge damper.
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
    lov = ["z", "vz", "curvature", "moment", "eta", "n_iter"]  # TODO ajouter energies
    lot = parameters.time_vector_output().tolist()
    res_cable = simulation.Results(
        lot=lot,
        lov=lov,
        lov_dims=[2, 2, 2, 2, 2, 1],
        los=parameters.los,
    )
    res_cable.start_timer()
    res_cable.update(
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

    acc_clamp_old = [0 for _ in stockbridges_dict.values()]
    acc_ang_clamp_old = [0 for _ in stockbridges_dict.values()]
    force_clamp = np.array([0 for _ in stockbridges_dict.values()])

    # Initialize stockbridge state and result containers for each damper.
    u1_old = [
        value.get("initial condition right") for value in stockbridges_dict.values()
    ]
    u1_new = u1_old.copy()
    u2_old = [
        value.get("initial condition left") for value in stockbridges_dict.values()
    ]
    u2_new = u2_old.copy()
    sb_results_dict = {}
    for idx, key in enumerate(stockbridges_dict.keys()):
        sb = stockbridges_dict[key]["stockbridge"]
        sb_results_dict[key] = Result(sb, lot)
        sb_results_dict[key].update(
            0, u1_old[idx], u2_old[idx], acc_clamp_old[idx], acc_ang_clamp_old[idx]
        )

    old_curvature_derivative1 = [
        np.zeros(value.get("stockbridge").mass_right.nb_space_points)
        for value in stockbridges_dict.values()
    ]
    old_curvature_derivative2 = [
        np.zeros(value.get("stockbridge").mass_left.nb_space_points)
        for value in stockbridges_dict.values()
    ]
    all_pos = np.array([value.get("position") for value in stockbridges_dict.values()])
    id_pos_stockbridge = np.maximum(
        1, np.minimum(ns - 2, np.round(all_pos / span.length * (ns - 1)))
    ).astype(int)
    d = x[id_pos_stockbridge + 1] - x[id_pos_stockbridge - 1]

    # time loop
    pb = spb.generate(parameters.pp, parameters.nt, desc=__name__)
    t_old = parameters.t0
    converged = True
    # curvature increment of the previous step, for the lagged branch
    dchi_old = np.zeros(ns)

    force_sb_old = np.zeros(ns)
    # time iteration
    for step in range(parameters.nt):
        t_new = t_old + dt

        # Apply previous stockbridge forces to the beam as distributed loads.
        # The stencil weights approximate the clamp force on adjacent beam nodes.
        force_sb_new = np.zeros(ns)
        force_sb_new[id_pos_stockbridge] += (
            -0.5 * 2 * force_clamp / d
        )  # TODO tester sans distriuer sur 3 points
        force_sb_new[id_pos_stockbridge + 1] += -0.25 * 2 * force_clamp / d
        force_sb_new[id_pos_stockbridge - 1] += -0.25 * 2 * force_clamp / d

        load = (
            vertical_force(t_old, y_old, v_old)
            + force_sb_old
            + vertical_force(t_new, y_old, v_old)
            + force_sb_new
        )
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

        # Compute clamp acceleration from the updated cable state.
        # This acceleration is passed back to the stockbridge model.
        bending_moment = law.dynamic_moment(chi_new, eta_new)
        acc_clamp_new = (
            vertical_force(t_new, y_old, v_old)[id_pos_stockbridge]
            + force_sb_new[id_pos_stockbridge]
            + span.tension * (D2 @ y_new)[id_pos_stockbridge]
            - (D2 @ bending_moment)[id_pos_stockbridge]
            - damp * v_new[id_pos_stockbridge]
        ) / conductor.mass
        acc_ang_clamp_new = 0 * acc_clamp_new

        # loop for all stockbridges
        for idx, key in enumerate(stockbridges_dict.keys()):
            sb = stockbridges_dict[key]["stockbridge"]
            A1 = sb.mass_right.build_matrix_acceleration_imposed(
                old_curvature_derivative1[idx], dt
            )
            A2 = sb.mass_left.build_matrix_acceleration_imposed(
                old_curvature_derivative2[idx], dt
            )
            rhs1 = sb.mass_right.build_rhs_acceleration_imposed(
                u1_old[idx],
                sb.clamp.half_length,
                (acc_clamp_old[idx] + acc_clamp_new[idx]) / 2,
                (acc_ang_clamp_old[idx] + acc_ang_clamp_new[idx]) / 2,
                dt,
            )
            rhs2 = sb.mass_left.build_rhs_acceleration_imposed(
                u2_old[idx],
                sb.clamp.half_length,
                (acc_clamp_old[idx] + acc_clamp_new[idx]) / 2,
                (acc_ang_clamp_old[idx] + acc_ang_clamp_new[idx]) / 2,
                dt,
            )

            # Solve the stockbridge damper state for the current imposed clamp accelerations.
            u1_new[idx] = sp.sparse.linalg.spsolve(A1, rhs1)
            u2_new[idx] = sp.sparse.linalg.spsolve(A2, rhs2)

            # Compute the clamp forces produced by the updated mass states.
            force_clamp, _ = sb.clamp.compute_forces_at_clamp(
                u1_new[idx][4],
                u2_new[idx][4],
                u1_new[idx][5],
                u2_new[idx][5],
                sb.mass_right.length_to_clamp,
                sb.mass_left.length_to_clamp,
                acc_clamp_new[idx],
                acc_ang_clamp_new[idx],
            )

            old_curvature_derivative1[idx] = (
                u1_new[idx][6 : 6 + sb.mass_right.nb_space_points]
                - u1_old[idx][6 : 6 + sb.mass_right.nb_space_points]
            )
            old_curvature_derivative2[idx] = (
                u2_new[idx][6 : 6 + sb.mass_left.nb_space_points]
                - u2_old[idx][6 : 6 + sb.mass_left.nb_space_points]
            )
            u1_old[idx] = np.array(u1_new[idx])
            u2_old[idx] = np.array(u2_new[idx])

        acc_clamp_old = acc_clamp_new
        force_sb_old = force_sb_new

        if (step + 1) % parameters.rr == 0:
            res_cable.update(
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

            for idx, key in enumerate(stockbridges_dict.keys()):
                sb = stockbridges_dict[key]["stockbridge"]
                sb_results_dict[key].update(
                    (step // parameters.rr) + 1,
                    u1_old[idx],
                    u2_old[idx],
                    acc_clamp_new[idx],
                    acc_ang_clamp_new[idx],
                )

        t_old = t_new
        y_old, v_old, chi_old, eta_old = y_new, v_new, chi_new, eta_new
        if not linear:
            dchi_old = dchi_new

    pb.close()
    res_cable.stop_timer()

    if converged:
        res_cable.set_state(
            {
                "z": y_old,
                "vz": v_old,
                "curvature": chi_old,
                "moment": law.dynamic_moment(chi_old, eta_old),
                "eta": eta_old,
            }
        )

    return res_cable, sb_results_dict
