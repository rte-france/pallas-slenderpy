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
from slenderpy.beam.dynamic import newmark_parameters
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
    alpha_m: float = 0.1,
    alpha_f: float = 0.37,
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
        under ``force`` at ``t0``. Its ghost nodes are set by the boundary
        relations, as for [](`~slenderpy.beam.dynamic.solve_dynamic`).
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
    alpha_m : float, optional
        Weight of the inertia at the start of the step in the generalized-alpha
        scheme, as for [](`~slenderpy.beam.dynamic.solve_dynamic`). Default 0.1.
    alpha_f : float, optional
        Weight of the forces at the start of the step. Default 0.37.

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
    x = np.linspace(0.0, span.length, ns)
    # the scheme runs on the extended grid, a ghost node beyond each support
    # carrying the boundary conditions (see slenderpy.boundary_condition); the
    # span nodes are nodes 1 to ne - 2
    ne = ns + 2

    # generalized-alpha weights, as in beam.dynamic.solve_dynamic: y(n+1) =
    # y_base + dy_dv * v(n+1), and the step equation, multiplied by gamma*dt,
    # weights the forces at n+1 by weight_new and the forces at n by weight_old
    gamma, beta = newmark_parameters(alpha_m, alpha_f)
    dy_dv = beta * dt / gamma
    weight_new = gamma * dt * (1.0 - alpha_f)
    weight_old = gamma * dt * alpha_f
    bc = span.boundary_conditions
    order = bc.order

    # operators; fourth_derivative already zeroes its border rows
    D2 = fdu.clean_matrix(order, fdu.second_derivative(ne, ds))
    D4 = fdu.fourth_derivative(ne, ds)
    BC, _ = bc.compute(ne, ds)
    identity = fdu.clean_matrix(order, sp.sparse.identity(ne))
    chi_operator = curvature.create(ne, ds, approx_curvature)

    # scheme matrices; A is state-independent, factorise it once
    f0 = 0.5 / span.length * np.sqrt(span.tension / conductor.mass)
    damp = 2.0 * conductor.mass * 2.0 * np.pi * f0 * zeta
    ei_D4 = law.ei_linear * D4
    stiffness = ei_D4 - span.tension * D2
    mass_matrix = conductor.mass * identity
    damping_matrix = damp * identity
    damped_mass = (1.0 - alpha_m) * mass_matrix + weight_new * damping_matrix
    A = damped_mass + weight_new * dy_dv * stiffness + BC
    lu = sp.sparse.linalg.splu(sp.sparse.csc_matrix(A))

    # round-off floor of the step residual: A @ v and dt * K @ y cannot be
    # evaluated better than eps * ||operator|| * |state|, which grows like
    # EI / ds**4 on fine grids; the stiffest tangent of the law bounds K
    stiffest = max(law.ei_linear, float(np.max(law.tangent(np.zeros(1)))))
    stiff = stiffest * D4 - span.tension * D2
    step_norm = fdu.inf_norm(damped_mass + weight_new * dy_dv * stiff + BC)
    elastic_norm = dt * fdu.inf_norm(stiff)

    # newton tangent, assembled and solved in banded storage. Its constant part
    # and the left factor of its bending term never change; for the approximate
    # curvature the right factor is the constant D2 as well, so only the tangent
    # stiffness varies from one iteration to the next
    tangent_weight = weight_new * dy_dv
    jacobian_base = fdu.banded(A - tangent_weight * ei_D4)
    left_rows = fdu.tridiagonal(D2)
    if approx_curvature:
        constant_rows = fdu.tridiagonal(chi_operator.jacobian(np.zeros(ne)))

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

    # the beam is planar: y and vy are zero, only fz is used. It is evaluated
    # on the span nodes, the ghost nodes carry none
    zeros = np.zeros(ns)

    def vertical_force(t, z, vz):
        return np.pad(zeros + force(x, t, zeros, z[1:-1], zeros, vz[1:-1])[1], 1)

    def boundary_rhs(t):
        """Right-hand side of the boundary rows, on the velocity."""
        return np.zeros(ne) if bc.dynamic_values is None else bc.update_rhs(ne, x, t)

    # initial state
    if initial_velocity is None:
        initial_velocity = np.zeros(ns)
    if initial_position is None:
        initial_position = shape.solve(
            conductor,
            span,
            vertical_force(parameters.t0, np.zeros(ne), np.zeros(ne))[1:-1],
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
    y_old = bc.extend(y_old, ds)
    v_old = bc.extend(v_old, ds, boundary_rhs(parameters.t0))

    chi_old = chi_operator.value(y_old)
    if initial_bending_moment is None:
        initial_bending_moment = law.moment(chi_old)
    else:
        # the ghost values only set an eta that nothing reads
        initial_bending_moment = np.pad(initial_bending_moment, 1, mode="edge")
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
            y_old[1:-1],
            v_old[1:-1],
            chi_old[1:-1],
            law.dynamic_moment(chi_old, eta_old)[1:-1],
            eta_old[1:-1],
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
    # the same nodes on the extended grid of the scheme
    id_pos_extended = id_pos_stockbridge + 1

    # time loop
    pb = spb.generate(parameters.pp, parameters.nt, desc=__name__)
    t_old = parameters.t0
    converged = True
    # curvature increment of the previous step, for the lagged branch
    dchi_old = np.zeros(ne)
    # velocity of the step before, for the newton initial guess
    v_prev = v_old

    force_sb_old = np.zeros(ne)
    # initial acceleration, from the equation of motion on the solved rows; the
    # boundary rows prescribe the velocity, their acceleration starts at zero at
    # the supports. The ghosts take theirs from the relations, as the position
    # does: the newmark updates then keep the relations on the position, where a
    # ghost starting from zero would drift from the node it mirrors
    a_old = (
        fdu.clean_rhs(
            order,
            vertical_force(t_old, y_old, v_old)
            + force_sb_old
            - damping_matrix @ v_old
            - stiffness @ y_old
            - (D2 @ law.dynamic_moment(chi_old, eta_old) - ei_D4 @ y_old),
        )
        / conductor.mass
    )
    a_old = bc.extend(a_old[1:-1], ds, np.zeros(ne))
    # time iteration
    for step in range(parameters.nt):
        t_new = t_old + dt

        # Apply previous stockbridge forces to the beam as distributed loads.
        # The stencil weights approximate the clamp force on adjacent beam nodes.
        force_sb_new = np.zeros(ne)
        force_sb_new[id_pos_extended] += (
            -0.5 * 2 * force_clamp / d
        )  # TODO tester sans distriuer sur 3 points
        force_sb_new[id_pos_extended + 1] += -0.25 * 2 * force_clamp / d
        force_sb_new[id_pos_extended - 1] += -0.25 * 2 * force_clamp / d

        # the force is evaluated at the state of the previous step
        load_old = vertical_force(t_old, y_old, v_old) + force_sb_old
        load_new = vertical_force(t_new, y_old, v_old) + force_sb_new
        rhs_bc = boundary_rhs(t_new)
        # newmark predictors: v(n+1) = v_star + gamma*dt*a(n+1) and
        # y(n+1) = y_base + dy_dv * v(n+1)
        v_star = v_old + dt * (1.0 - gamma) * a_old
        y_base = y_old + dt * v_old + dt**2 * (0.5 - beta) * a_old - dy_dv * v_star
        inertia = mass_matrix @ (
            (1.0 - alpha_m) * v_star - gamma * dt * alpha_m * a_old
        )
        elastic = stiffness @ (
            weight_new * y_base + weight_old * y_old
        ) + weight_old * (damping_matrix @ v_old)
        external = fdu.clean_rhs(order, weight_new * load_new + weight_old * load_old)
        rhs = inertia - elastic + external + rhs_bc
        threshold = max(
            tol * fdu.residual_scale((inertia, elastic, external, rhs_bc)),
            fdu.round_off_floor([(step_norm, v_old), (elastic_norm, y_old)]),
        )

        if linear:
            v_new = lu.solve(rhs)
            y_new = y_base + dy_dv * v_new
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
                y = y_base + dy_dv * v
                chi = chi_operator.value(y)
                # the hysteresis is driven by the curvature increment of the
                # step itself, as in beam.dynamic.solve_dynamic
                dchi = chi - chi_old
                eta = law.update_eta(eta_old, dchi, branch)
                remainder = D2 @ law.dynamic_moment(chi, eta) - ei_D4 @ y
                residual = (
                    A @ v - rhs + weight_old * remainder_old + weight_new * remainder
                )
                return y, chi, dchi, eta, residual

            def newton(branch=None):
                """Newton on the step residual, on ``branch`` (default the law's own).

                It starts from the velocity extrapolated from the two previous
                steps, as in ``beam.dynamic.solve_dynamic``.
                """
                v = 2.0 * v_old - v_prev
                state = step_state(v, branch)
                error = np.abs(state[-1]).max()
                n_iter = 0
                while n_iter < max_iter and error > threshold:
                    y, _, dchi, eta, step_residual = state
                    # tangent bending stiffness of the law over the step. The
                    # moment reaches the velocity through the curvature and
                    # through eta, whose increment is the curvature increment:
                    # both carry dy_dv, so dynamic_tangent applies as is
                    tangent = law.dynamic_tangent(eta, dchi, branch)

                    jacobian = jacobian_base + tangent_weight * fdu.product_band(
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

        # acceleration of the beam at the end of the step, from the newmark
        # update; its value at the clamps is passed back to the stockbridges
        a_new = (v_new - v_star) / (gamma * dt)
        acc_clamp_new = a_new[id_pos_extended]
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
                    y_new[1:-1],
                    v_new[1:-1],
                    chi_new[1:-1],
                    law.dynamic_moment(chi_new, eta_new)[1:-1],
                    eta_new[1:-1],
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
        v_prev = v_old
        y_old, v_old, a_old = y_new, v_new, a_new
        chi_old, eta_old = chi_new, eta_new
        if not linear:
            dchi_old = dchi_new

    pb.close()
    res_cable.stop_timer()

    if converged:
        res_cable.set_state(
            {
                "z": y_old[1:-1],
                "vz": v_old[1:-1],
                "curvature": chi_old[1:-1],
                "moment": law.dynamic_moment(chi_old, eta_old)[1:-1],
                "eta": eta_old[1:-1],
            }
        )

    return res_cable, sb_results_dict
