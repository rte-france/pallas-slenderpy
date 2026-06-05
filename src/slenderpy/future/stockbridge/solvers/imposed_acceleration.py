"""Time integration with the clamp acceleration imposed.

Each mass is integrated independently in a sparse linear system of size
``6 + 2*n_i`` per side. The clamp force and moment are reconstructed
post-hoc from Newton's second law on the clamp.
"""

import numpy as np
import scipy as sp

from ..core.stockbridge import Stockbridge, Result


def solve_imposed_acceleration(
    sb: Stockbridge,
    tf: float,
    initial_conditions_right: np.ndarray,
    initial_conditions_left: np.ndarray,
    vertical_acceleration: float,
    rot_acceleration: float,
    dt: float,
) -> Result:
    """Solver for the imposed-acceleration case. 

    Parameters
    ----------
    sb : Stockbridge
        Stockbridge object containing the simulation data.
    tf : float
        Final time of the simulation.
    initial_conditions_right : np.ndarray
        Initial state vector of the right mass of size ``6 + 2*n1``.
    initial_conditions_left : np.ndarray
        Initial state vector of the left mass of size ``6 + 2*n2``.
    vertical_acceleration : float
        Imposed vertical clamp acceleration (m/s^2).
    rot_acceleration : float
        Imposed rotational clamp acceleration (rad/s^2).
    dt : float
        Time step for the simulation.

    Returns
    -------
    Result
        _description_
    """
    n1 = sb.mass_right.nb_space_points
    n2 = sb.mass_left.nb_space_points
    dt = dt

    t = np.arange(0, tf, dt)
    res = Result(sb, t)
    res.update(0, initial_conditions_right, initial_conditions_left, vertical_acceleration[0], rot_acceleration[0])

    u1_old = initial_conditions_right.copy()
    u2_old = initial_conditions_left.copy()
    old_curvature_derivative1 = np.zeros(n1)
    old_curvature_derivative2 = np.zeros(n2)

    for k in range(1, len(t)):
        # Each side is independent once the clamp acceleration is given,
        # so we solve two decoupled sparse systems per step.
        A1 = sb.mass_right.build_matrix_acceleration_imposed(
            old_curvature_derivative1, dt
        )
        A2 = sb.mass_left.build_matrix_acceleration_imposed(
            old_curvature_derivative2, dt
        )
        rhs1 = sb.mass_right.build_rhs_acceleration_imposed(
            u1_old,
            sb.clamp.half_length,
            (vertical_acceleration[k]+vertical_acceleration[k-1])/2,
            (rot_acceleration[k]+rot_acceleration[k-1])/2,
            dt,
        )
        rhs2 = sb.mass_left.build_rhs_acceleration_imposed(
            u2_old,
            sb.clamp.half_length,
            (vertical_acceleration[k]+vertical_acceleration[k-1])/2,
            (rot_acceleration[k]+rot_acceleration[k-1])/2,
            dt,
        )

        u1 = sp.sparse.linalg.spsolve(A1, rhs1)
        u2 = sp.sparse.linalg.spsolve(A2, rhs2)

        # Curvature increment to freeze the Bouc-Wen term at the next step.
        old_curvature_derivative1 = u1[6 : 6 + n1] - u1_old[6 : 6 + n1]
        old_curvature_derivative2 = u2[6 : 6 + n2] - u2_old[6 : 6 + n2]

        res.update(k, u1, u2, vertical_acceleration[k], rot_acceleration[k]) 
        u1_old = np.array(u1)
        u2_old = np.array(u2)

    return res
