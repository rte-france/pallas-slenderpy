"""Linearised 5-DOF integration with imposed clamp force.

Reduces the coupled stockbridge system to a constant 5x5 linear system
solved at each time step. The five unknowns are:

- ``0..1``: right-mass position ``[x, phi]``,
- ``2..3``: right-mass velocity ``[v, omega]``,
- ``4``: clamp vertical acceleration.

The state is symmetric (right and left masses share the same trajectory),
so only one mass is integrated; the other entries of the :class:`Result`
container are filled with NaNs.
"""

import numpy as np

from ..core.stockbridge import Stockbridge, Result 

#: Size of the reduced linearised state vector.
LINEARIZED_SIZE = 5


def _build_A_linearized(sb: Stockbridge, dt) -> np.ndarray:
    """Build the constant 5x5 system matrix.

    Parameters
    ----------
    sb : Stockbridge
        Stockbridge object, which must have been constructed with stiffness and damping matrices.
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Dense matrix of shape ``(5, 5)``.

    Raises
    ------
    ValueError
        If the stockbridge does not carry a linearised stiffness/damping pair.
    """
    if sb.K is None or sb.C is None:
        raise ValueError(
            "Stockbridge must be constructed with K and C to use the linearized solver"
        )
    A = np.zeros((LINEARIZED_SIZE, LINEARIZED_SIZE))
    invM = sb.mass_right.mass_matrix_inv
    a, _ = sb.ab
    a = a[:2]  # only the right-mass coupling vector (the system is symmetric)
    ht = 0.5 * dt

    # Kinematics: X_{n+1} - dt/2 * dotX_{n+1} = ...
    A[0:2, 0:2] = np.eye(2)
    A[0:2, 2:4] = -ht * np.eye(2)

    # Newton on the right mass, projected through M^{-1}.
    A[2:4, 0:2] = ht * (invM @ sb.K)
    A[2:4, 2:4] = np.eye(2) + ht * (invM @ sb.C)
    A[2:4, 4] = ht * (invM @ a)

    # Newton on the clamp (vertical only); the factor 2 picks up the two
    # symmetric reactions transmitted through K[0,:] and C[0,:].
    A[4, 4] = sb.clamp.mass
    A[4, 0:2] = -2 * sb.K[0, :]
    A[4, 2:4] = -2 * sb.C[0, :]

    return A


def build_rhs_linearized(
    sb: Stockbridge, time_step: int, old_unknowns: np.ndarray, f_ext: np.ndarray, dt: float
) -> np.ndarray:
    """Build the right-hand side at a given time step.

    Parameters
    ----------
    sb : Stockbridge
        Stockbridge object. 
    time_step : int
        Time step index.
    old_unknowns : np.ndarray
        Unknowns at the previous time step, shape ``(5,)``.
    f_ext : np.ndarray
        External forces, shape ``(5, nb_time_steps)``.
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Right-hand side vector of shape ``(5,)``.
    """
    rhs = np.zeros(LINEARIZED_SIZE)
    invM = sb.mass_right.mass_matrix_inv
    a, _ = sb.ab
    a = a[:2]
    ht = 0.5 * dt

    X_n = old_unknowns[0:2]
    dotX_n = old_unknowns[2:4]
    ddotWc_n = old_unknowns[4]

    # Crank-Nicolson explicit half of the kinematic relation.
    rhs[0:2] = X_n + ht * dotX_n

    # Newton on the mass: explicit half + averaged external force.
    f_sum = f_ext[0:2, time_step] + f_ext[0:2, time_step - 1]
    rhs[2:4] = (
        dotX_n
        - ht
        * (
            (invM @ sb.K) @ X_n
            + (invM @ sb.C) @ dotX_n
            + (invM @ a) * ddotWc_n
        )
        + ht * (invM @ f_sum)
    )

    # Imposed clamp force (last row of f_ext) drives Newton on the clamp.
    rhs[4] = f_ext[-1, time_step]
    return rhs


def solve_linearized_imposed_force(
    sb: Stockbridge,
    tf: float,
    initial_conditions: np.ndarray,
    f_ext: np.ndarray,
    dt: float,
) -> Result:
    """Solve the stockbridge dynamics with an imposed clamp force, using a linearised model and Crank-Nicolson time integration.

    Parameters
    ----------
    sb : Stockbridge
        Stockbridge object.
    tf : float
        Final time.
    initial_conditions : np.ndarray
        Initial conditions, shape ``(5,)``.
    f_ext : np.ndarray
        External forces, shape ``(5, nb_time_steps)``.
    dt : float
        Time step.

    Returns
    -------
    Result
        Result object containing the solution.
    """
    
    dt = dt
    n1 = sb.mass_right.nb_space_points
    n2 = sb.mass_left.nb_space_points

    t = np.arange(0, tf, dt)
    res = Result(sb, t)
    indexr = [0, 1, 4, 5, 10, 11] + list(range(14, 14+2*n1))
    indexl = [2, 3, 6, 7, 12, 13] + list(range(14+2*n1, 14+2*n1+2*n2))
    res.update(0, initial_conditions[indexr], initial_conditions[indexl], initial_conditions[8], initial_conditions[9], f_ext[-1][0], np.nan)

    # Reduced state: pick out [xR, phiR, vR, omegaR, ddot_wc] from the full
    # 14+2*n state vector at indices [0, 1, 4, 5, 8].
    u_old = np.hstack(
        [
            initial_conditions[:2],
            initial_conditions[4:6],
            initial_conditions[8],
        ]
    )
    # System matrix is constant along time, so build it once.
    A = _build_A_linearized(sb, dt)
    # Pad with NaN for components the linearised model does not compute,
    # so the Out container clearly flags them.
    u_zero = np.full(len(initial_conditions) - 5, np.nan)

    for k in range(1, len(t)):
        rhs = build_rhs_linearized(sb, k, u_old, f_ext, dt)
        u = np.linalg.solve(A, rhs)

        # Re-expand the 5 unknowns onto the full state vector layout used
        # by Out.fill_imposed_force (positions 0..1, velocities 4..5,
        # clamp acceleration 8); leave everything else as NaN.
        u_tot = np.hstack(
            [u[:2], u_zero[:2], u[2:4], u_zero[2:4], u[4], u_zero[4:]]
        )
        res.update(k, u_tot[indexr], u_tot[indexl], u_tot[8], u_tot[9], f_ext[-1][k], np.nan)
        u_old = u

    return res
