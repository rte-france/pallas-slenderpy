from typing import Any, Optional

import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD
from slenderpy import _progress_bar as spb
from slenderpy import simtools
from slenderpy.future.stockbridge import Result


def solve_dynamic_with_sb(
    stockbridges_dict: dict,
    beam: Any,
    parameters: Any,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    force: np.ndarray,
    approx_curvature: bool,
    initial_bending_moment: Optional[np.ndarray] = None,
    zeta: float = 0.0,
    f0: Optional[float] = None,
    it_picard: int = 1,
    tol_picard: float = 1e-3,
) -> tuple[simtools.Results, dict[str, Result]]:
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
    # Beam setup: spatial grid, time step and derivative matrices.
    lspan = beam.length
    ns = parameters.ns
    ds = lspan / (ns - 1)
    dt = parameters.tf / parameters.nt
    x = np.linspace(0.0, lspan, ns)
    current_time = parameters.t0 + dt

    order = beam.bc.order
    D1 = FD.first_derivative(ns, ds)
    D2_border = FD.second_derivative(ns, ds)
    D2 = FD.clean_matrix(order, D2_border)
    D4 = FD.fourth_derivative(ns, ds)
    rhs_bc = np.zeros(ns)

    if approx_curvature:
        # Linear curvature approximation simplifies the beam operator.
        K = beam.get_ei() * D4 - beam.tension * D2

        def curvature(y):
            return D2 @ y
    else:
        # Nonlinear curvature uses the full geometric expression.
        K = -beam.tension * D2

        def curvature(y):
            return D2_border @ y / np.sqrt((1 + (D1 @ y) ** 2) ** 3)

    y_old = initial_position
    v_old = initial_velocity
    curvature_old = curvature(y_old)
    if initial_bending_moment is None:
        initial_bending_moment = beam._bending_moment(curvature_old)
    bending_moment_old = initial_bending_moment
    eta_old = beam._init_eta(bending_moment_old, curvature_old)

    if f0 is None:
        f0 = beam.natural_frequency()
    damp = 2 * beam.mass * 2 * np.pi * f0 * zeta

    # beam result
    toolbox = beam._build_dict(parameters, damp, K, D2)
    powers_name = ["p_kin", "p_bend", "p_tens", "p_ext", "p_dissip"]
    energies_name = ["e_kin", "e_bend", "e_tens", "e_ext", "e_dissip"]
    picard = ["it_picard"]
    lov = ["y", "v", "c", "M"]
    all_lov = lov + powers_name + energies_name + picard
    time_vector = parameters.time_vector_output().tolist()
    res_cable = simtools.Results(
        lot=time_vector,
        lov=all_lov,
        lov_dims=[
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        los=np.linspace(0, 1, len(parameters.los)),
    )
    res_cable.update(
        0, x / lspan, lov, [y_old, v_old, curvature_old, bending_moment_old]
    )
    pb = spb.generate(parameters.pp, parameters.nt, desc=__name__)

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
        sb_results_dict[key] = Result(sb, time_vector)
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
        1, np.minimum(ns - 2, np.round(all_pos / lspan * (ns - 1)))
    ).astype(int)
    d = x[id_pos_stockbridge + 1] - x[id_pos_stockbridge - 1]

    # time iteration
    for k in range(1, parameters.nt):
        if beam.bc.dynamic_values is not None:
            rhs_bc = beam.bc.update_rhs(ns, x, k)

        # Apply previous stockbridge forces to the beam as distributed loads.
        # The stencil weights approximate the clamp force on adjacent beam nodes.
        force[id_pos_stockbridge, k] += -0.5 * 2 * force_clamp / d
        force[id_pos_stockbridge + 1, k] += -0.25 * 2 * force_clamp / d
        force[id_pos_stockbridge - 1, k] += -0.25 * 2 * force_clamp / d

        force_previous = FD.clean_rhs(order, force[:, k - 1])
        force_current = FD.clean_rhs(order, force[:, k])

        v_new, y_new, eta_new, curvature_new, bending_moment_new, it = (
            beam.picard_process(
                toolbox,
                v_old,
                y_old,
                eta_old,
                rhs_bc,
                curvature,
                approx_curvature,
                force_current + force_previous,
                it_picard,
                tol_picard,
            )
        )

        # Compute clamp acceleration from the updated cable state.
        # This acceleration is passed back to the stockbridge model.
        acc_clamp_new = (
            force[id_pos_stockbridge, k]
            + beam.tension * (D2 @ y_new)[id_pos_stockbridge]
            - (D2 @ bending_moment_new)[id_pos_stockbridge]
            - damp * v_new[id_pos_stockbridge]
        ) / beam.mass
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

        if (k + 1) % parameters.rr == 0:
            values = (
                [y_new, v_new, curvature_new, bending_moment_new]
                + list(
                    beam.compute_power(
                        D2,
                        curvature_new,
                        v_old,
                        v_new,
                        y_new,
                        eta_new,
                        force_current,
                        dt,
                        x,
                    ).values()
                )
                + [it]
            )
            res_cable.update(
                (k // parameters.rr) + 1,
                x / lspan,
                lov + powers_name + picard,
                values,
            )
            pb.update(parameters.rr)
            for idx, key in enumerate(stockbridges_dict.keys()):
                sb = stockbridges_dict[key]["stockbridge"]
                sb_results_dict[key].update(
                    (k // parameters.rr) + 1,
                    u1_old[idx],
                    u2_old[idx],
                    acc_clamp_new[idx],
                    acc_ang_clamp_new[idx],
                )

        current_time += dt
        v_old = v_new
        y_old = y_new
        eta_old = eta_new
        curvature_old = curvature_new
        bending_moment_old = bending_moment_new

    beam.update_energies(
        res_cable,
        powers_name,
        energies_name,
        parameters.tf / parameters.nr,
        parameters.nr,
    )
    pb.close()
    res_cable.set_state(
        {"y": y_new, "v": v_new, "c": curvature_new, "M": bending_moment_new}
    )
    return res_cable, sb_results_dict
