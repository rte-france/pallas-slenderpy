from __future__ import annotations

from typing import Optional

import numpy as np
import scipy as sp

from slenderpy import _progress_bar as spb
import slenderpy.future.fd_utils as fdu
import slenderpy.future.beam.bending as bending
import slenderpy.future.beam.curvature as curvature
from slenderpy.future import simulation
from slenderpy.future.beam.bending import BendingModel
from slenderpy.future.beam.static import shape
from slenderpy.future.components import Span, Conductor 

from slenderpy.future.stockbridge.core.stockbridge import Result

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
        Dictionary of stockbridge dampers, with keys being the name of the damper and values being dictionaries with keys "stockbridge" (the Stockbridge object),
        "position" (the position of the damper on the beam) and "initial_conditions" (the initial conditions for the damper).
    beam : Any
        Beam object.
    parameters : Any
        Parameters object containing the simulation parameters.
    initial_position : np.ndarray
        Initial position of the beam.
    initial_velocity : np.ndarray
        Initial velocity of the beam.
    force : np.ndarray
        External force acting on the beam.
    approx_curvature : bool
        Whether to use an approximation for the curvature.
    initial_bending_moment : Optional[np.ndarray], optional
        Initial bending moment of the beam, by default None
    zeta : float, optional
        Damping ratio, by default 0.0
    f0 : Optional[float], optional
        Natural frequency, by default None
    it_picard : int, optional
        Number of Picard iterations, by default 1
    tol_picard : float, optional
        Tolerance for Picard iterations, by default 1e-3

    Returns
    -------
    tuple[simtools.Results, dict[str, Result]]
        The first element is a simtools.Results object containing the results for the beam, and the second element is a dictionary of Result objects for each stockbridge damper.
    """
    model = BendingModel(model)
    
    if span.boundary_conditions is None:
        raise ValueError("span.boundary_conditions is required for a beam solve")

    law = bending.create(conductor, span, model, ei)

    # discretisation
    ns = parameters.ns
    ds = span.length / (ns - 1)
    dt = (parameters.tf - parameters.t0) / parameters.nt
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

        def force(x, t, y, v):
            return np.zeros_like(x)

    # initial state
    if initial_velocity is None:
        initial_velocity = np.zeros(ns)
    if initial_position is None:
        initial_position = shape.solve(
            conductor,
            span,
            force(x, parameters.t0, np.zeros(ns), np.zeros(ns)),
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
    lov = ["y", "v", "c", "M", "eta", "n_iter"] # TODO ajouter energies 
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

    force_sb_old = np.zeros(ns)
    # time iteration
    for step in range(parameters.nt):
        t_new = t_old + dt

        # Apply previous stockbridge forces to the beam as distributed loads.
        # The stencil weights approximate the clamp force on adjacent beam nodes.
        force_sb_new = np.zeros(ns)
        force_sb_new[id_pos_stockbridge] += -0.5 * 2 * force_clamp / d # TODO tester sans distriuer sur 3 points
        force_sb_new[id_pos_stockbridge + 1] += -0.25 * 2 * force_clamp / d
        force_sb_new[id_pos_stockbridge - 1] += -0.25 * 2 * force_clamp / d

        load = force(x, t_old, y_old, v_old) + force_sb_old + force(x, t_new, y_old, v_old) + force_sb_new
        rhs_bc = (
            np.zeros(ns) if bc.dynamic_values is None else bc.update_rhs(ns, x, t_new)
        )
        inertia = B @ v_old
        elastic = dt * stiffness @ y_old
        external = dt2 * fdu.clean_rhs(order, load)
        rhs = inertia - elastic + external + rhs_bc
        threshold = tol * fdu.residual_scale((inertia, elastic, external, rhs_bc))

        if linear:
            v_new = lu.solve(rhs)
            y_new = y_old + dt2 * (v_old + v_new)
            chi_new = chi_operator.value(y_new)
            eta_new = eta_old
            n_iter = 1
        else:
            remainder_old = D2 @ law.dynamic_moment(chi_old, eta_old) - ei_D4 @ y_old

            def step_state(v):
                """State and step residual reached by a candidate velocity."""
                y = y_old + dt2 * (v_old + v)
                chi = chi_operator.value(y)
                # curvature increment driving the hysteresis, as the curvature
                # rate at the end of the step times the step: the same fully
                # implicit discretisation the eta update of the law is built on
                dchi = dt * chi_operator.rate(y, v)
                eta = law.update_eta(eta_old, dchi)
                remainder = D2 @ law.dynamic_moment(chi, eta) - ei_D4 @ y
                residual = A @ v - rhs + dt2 * (remainder_old + remainder)
                return y, chi, dchi, eta, residual

            v_new = v_old
            y_new, chi_new, dchi_new, eta_new, step_residual = step_state(v_new)
            error = np.abs(step_residual).max()
            n_iter = 0

            while n_iter < max_iter and error > threshold:
                # tangent bending stiffness of the law over the step. The
                # moment reaches the velocity twice: once through the curvature,
                # whose derivative carries dt2, and once through eta, whose
                # increment carries dt, i.e. twice as much. Only the hysteretic
                # part of dynamic_tangent takes the second path, so it is the
                # only one weighted twice
                tangent = 2.0 * law.dynamic_tangent(eta_new, dchi_new) - law.ei_linear

                jacobian = jacobian_base + dt2**2 * fdu.product_band(
                    left_rows, right_rows(y_new), tangent
                )
                try:
                    increment = sp.linalg.solve_banded(
                        (fdu.BANDWIDTH, fdu.BANDWIDTH), jacobian, -step_residual
                    )
                except np.linalg.LinAlgError:
                    break

                # backtrack while the increment increases the residual, because
                # the full newton step can overshoot where the bouc-wen law is
                # not differentiable, at a sign change of dchi or eta. Such a
                # sign change also needs a step that momentarily increases the
                # residual, so a failed backtrack takes the full step instead of
                # giving up: only max_iter ends the iteration.
                relaxation = 1.0
                trial = step_state(v_new + increment)
                while np.abs(trial[3]).max() >= error and relaxation > _MIN_RELAXATION:
                    relaxation *= 0.5
                    trial = step_state(v_new + relaxation * increment)

                if np.abs(trial[3]).max() >= error:
                    relaxation = 1.0
                    trial = step_state(v_new + increment)

                v_new = v_new + relaxation * increment
                y_new, chi_new, dchi_new, eta_new, step_residual = trial
                error = np.abs(step_residual).max()
                n_iter += 1

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
            force(x, t_new, y_old, v_old)[id_pos_stockbridge] + force_sb_new[id_pos_stockbridge]
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

    pb.close()
    res_cable.stop_timer()

    if converged:
        res_cable.set_state(
            {
                "y": y_old,
                "v": v_old,
                "c": chi_old,
                "M": law.dynamic_moment(chi_old, eta_old),
                "eta": eta_old,
            }
        )
    
    return res_cable, sb_results_dict
