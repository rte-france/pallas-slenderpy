"""Time integration with the clamp force and moment imposed.

The unknowns of each time step are stacked into a single state vector of
size ``14 + 2*n1 + 2*n2`` and are ordered as follows:

- ``0..3``: positions ``[xR, phiR, xL, phiL]``,
- ``4..7``: velocities ``[vR, omegaR, vL, omegaL]``,
- ``8``: clamp vertical acceleration,
- ``9``: clamp angular acceleration,
- ``10..13``: contact forces and moments at the masses
  ``[FR, MR, FL, ML]``,
- ``14..14+n1``: right-mass curvature field,
- ``14+n1..14+2*n1``: right-mass hysteresis field,
- ``14+2*n1..14+2*n1+n2``: left-mass curvature field (the layout uses
  ``j*n[j]`` offsets so the loops work for both sides),
- the next ``n2`` entries: left-mass hysteresis field.
"""

import numpy as np

from ..core.stockbridge import Result, Stockbridge


def _build_A_base(sb: Stockbridge, dt: float) -> np.ndarray:
    """Build the time-step-independent part of the system matrix.

    Parameters
    ----------
    sb : Stockbridge
        Stockbridge model containing the system parameters.
    dt : float
        Time step for the simulation.

    Returns
    -------
    np.ndarray
        Dense matrix of shape ``(14 + 2*n1 + 2*n2, 14 + 2*n1 + 2*n2)``.
    """
    n1 = sb.mass_right.nb_space_points
    n2 = sb.mass_left.nb_space_points
    n = {0: n1, 2: n2}
    size = 14 + 2 * n1 + 2 * n2
    M = sb.mass_matrix_inv
    A = np.zeros((size, size))
    a, b = sb.ab
    length = sb.mass_right.length_to_clamp
    x1 = np.linspace(0, length, n1)
    x2 = np.linspace(0, length, n2)
    x = {0: x1, 2: x2}
    ei_min = {0: sb.mass_right.ei_min, 2: sb.mass_left.ei_min}
    ei_max = {0: sb.mass_right.ei_max, 2: sb.mass_left.ei_max}
    chi0 = {0: sb.mass_right.chi0, 2: sb.mass_left.chi0}

    ht = 0.5 * dt

    # Kinematic relation: x_{n+1} - dt/2 * v_{n+1} = ... (Crank-Nicolson)
    for k in range(4):
        A[k, k] = 1
        A[k, k + 4] = -ht

    # Newton's law on each mass, projected through M^{-1}, coupled to the
    # clamp accelerations (cols 8, 9) and the contact reactions (cols 10..13).
    for k in range(4, 8):
        A[k, k] = 1
        A[k, 8] = M[k - 4, :] @ a * ht
        A[k, 9] = M[k - 4, :] @ b * ht
        A[k, 10:14] = M[k - 4, :] * ht

    # Newton's law on the clamp: row 8 vertical, row 9 angular.
    A[8, 8] = sb.clamp.mass
    A[8, 10] = -1
    A[8, 12] = -1
    A[9, 9] = sb.clamp.moment_of_inertia
    A[9, 11] = -1
    A[9, 13] = 1
    A[9, 10] = -(sb.clamp.half_length + sb.mass_right.length_to_clamp)
    A[9, 12] = +(sb.clamp.half_length + sb.mass_left.length_to_clamp)

    # Compatibility: link mass position/rotation to the integral of the
    # cable curvature field (trapezoidal weights). j=0 is the right mass,
    # j=2 the left one.
    for j in [0, 2]:
        A[10 + j, 0 + j] = 1
        A[11 + j, 1 + j] = 1
        A[10 + j, 14 + j * n[j]] = -(x[j][1] - x[j][0]) / 2 * (length - x[j][0])
        A[11 + j, 14 + j * n[j]] = -(x[j][1] - x[j][0]) / 2
        for k in range(1, n[j] - 1):
            A[10 + j, 14 + j * n[j] + k] = (
                -(x[j][k + 1] - x[j][k - 1]) / 2 * (length - x[j][k])
            )
            A[11 + j, 14 + j * n[j] + k] = -(x[j][k + 1] - x[j][k - 1]) / 2
        A[10 + j, 14 + j * n[j] + n[j] - 1] = (
            -(x[j][-1] - x[j][-2]) / 2 * (length - x[j][-1])
        )
        A[11 + j, 14 + j * n[j] + n[j] - 1] = -(x[j][-1] - x[j][-2]) / 2

    # Hysteretic moment-curvature constitutive law along the cable.
    # The eta-eta diagonal block (the only term depending on the previous
    # step) is left at zero here and refreshed by build_matrix_force_imposed.
    for j in [0, 2]:
        A[14 + j * n[j] : 14 + j * n[j] + n[j], 10 + j] = length - x[j]
        A[14 + j * n[j] : 14 + j * n[j] + n[j], 11 + j] = 1
        for k in range(n[j]):
            A[14 + j * n[j] + k, 14 + j * n[j] + k] = -ei_min[j][k]
            A[14 + j * n[j] + k, 14 + j * n[j] + n[j] + k] = (
                -(ei_max[j][k] - ei_min[j][k]) * chi0[j][k]
            )
            A[14 + j * n[j] + n[j] + k, 14 + j * n[j] + k] = -1

    return A


def build_matrix_force_imposed(
    sb: Stockbridge, A_base: np.ndarray, old_curvature_derivative: np.ndarray
) -> np.ndarray:
    """Refresh the hysteresis diagonal entries of the system matrix.

    Parameters
    ----------
    sb : Stockbridge
        Stockbridge model containing the system parameters.
    A_base : np.ndarray
        Base matrix produced by :func:`_build_A_base`. Mutated in place and returned.
    old_curvature_derivative : np.ndarray
        Concatenation of the curvature increments of the right and left masses, shape ``(n1 + n2,)``.

    Returns
    -------
    np.ndarray
        The mutated ``A_base`` matrix.
    """
    n1 = sb.mass_right.nb_space_points
    n2 = sb.mass_left.nb_space_points
    n = {0: n1, 2: n2}
    chi0 = sb.mass_right.chi0
    A = A_base
    # Bouc-Wen-like linearisation: the |dchi/dt| term is frozen at the
    # previous step. j=0 -> right mass, j=2 -> left mass.
    for j in [0, 2]:
        for k in range(n[j]):
            A[14 + j * n[j] + n[j] + k, 14 + j * n[j] + n[j] + k] = chi0[k] + np.abs(
                old_curvature_derivative[k + n[j] * (j // 2)]
            )
    return A


def build_rhs_force_imposed(
    sb: Stockbridge,
    time_step: int,
    old_unknowns: np.ndarray,
    fc: np.ndarray,
    mc: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Build the right-hand side at a given time step.

    Parameters
    ----------
    sb : Stockbridge
        Stockbridge model containing the system parameters.
    time_step : int
        Index of the current time step.
    old_unknowns : np.ndarray
        Solver state vector at the previous step.
    fc : np.ndarray
        Imposed clamp force time series (N), shape ``(nb_time_steps,)``.
    mc : np.ndarray
        Imposed clamp moment time series (N.m), shape ``(nb_time_steps,)``.
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        1D array of size ``14 + 2*n1 + 2*n2``.
    """
    n1 = sb.mass_right.nb_space_points
    n2 = sb.mass_left.nb_space_points
    n = {0: n1, 2: n2}
    size = 14 + 2 * n1 + 2 * n2
    rhs = np.zeros(size)
    M = sb.mass_matrix_inv
    a, b = sb.ab
    chi0 = sb.mass_right.chi0
    ht = 0.5 * dt

    # Crank-Nicolson explicit halves of kinematics (rows 0..3) and Newton
    # on the masses (rows 4..7).
    for k in range(4):
        rhs[k] = old_unknowns[k] + ht * old_unknowns[k + 4]
        rhs[k + 4] = old_unknowns[k + 4] - ht * (
            M[k, :] @ old_unknowns[10:14]
            + M[k, :] @ a * old_unknowns[8]
            + M[k, :] @ b * old_unknowns[9]
        )
    # Imposed clamp force / moment.
    rhs[8] = fc[time_step]
    rhs[9] = mc[time_step]
    # Hysteresis update: explicit half of eta_{n+1} - eta_n equation.
    for j in [0, 2]:
        rhs[14 + j * n[j] + n[j] : 14 + j * n[j] + 2 * n[j]] = (
            -old_unknowns[14 + j * n[j] : 14 + j * n[j] + n[j]]
            + chi0 * old_unknowns[14 + j * n[j] + n[j] : 14 + j * n[j] + 2 * n[j]]
        )
    return rhs


def solve_imposed_force(
    sb: Stockbridge,
    tf: float,
    initial_conditions: np.ndarray,
    fc: np.ndarray,
    mc: np.ndarray,
    dt,
) -> Result:
    """Integrate the stockbridge response under imposed clamp force/moment.

    Uses a Crank-Nicolson scheme on the coupled system. The hysteresis
    equation is linearised at each step using the previous-step curvature
    increment.

    Parameters
    ----------
    sb : Stockbridge
        Stockbridge model containing the system parameters.
    tf : float
        Final time of the simulation.
    initial_conditions : np.ndarray
        Initial state vector of size ``14 + 2*n1 + 2*n2``.
    fc : np.ndarray
        Imposed clamp force time series (N), shape ``(nb_time_steps,)``.
    mc : np.ndarray
        Imposed clamp moment time series (N.m), shape ``(nb_time_steps,)``.
    dt : float
        Time step.

    Returns
    -------
    Result
        Simulation results, with time series for the clamp and both masses.
    """
    n1 = sb.mass_right.nb_space_points
    n2 = sb.mass_left.nb_space_points
    dt = dt

    t = np.arange(0, tf, dt)
    res = Result(sb, t)
    indexr = [0, 1, 4, 5, 10, 11] + list(range(14, 14 + 2 * n1))
    indexl = [2, 3, 6, 7, 12, 13] + list(range(14 + 2 * n1, 14 + 2 * n1 + 2 * n2))
    res.update(
        0,
        initial_conditions[indexr],
        initial_conditions[indexl],
        initial_conditions[8],
        initial_conditions[9],
        fc[0],
        mc[0],
    )

    # Pre-compute the time-step-independent block of the system matrix.
    A_base = _build_A_base(sb, dt)
    u_old = initial_conditions.copy()
    old_curvature_derivative = np.zeros(n1 + n2)

    for k in range(1, len(t)):
        # Refresh the Bouc-Wen diagonal then assemble RHS and solve.
        A = build_matrix_force_imposed(sb, A_base, old_curvature_derivative)
        rhs = build_rhs_force_imposed(sb, k, u_old, fc, mc, dt)
        u = np.linalg.solve(A, rhs)

        # Cache the curvature increment for the next step's linearisation.
        old_curvature_derivative1 = u[14 : 14 + n1] - u_old[14 : 14 + n1]
        old_curvature_derivative2 = (
            u[14 + 2 * n2 : 14 + 3 * n2] - u_old[14 + 2 * n2 : 14 + 3 * n2]
        )
        old_curvature_derivative = np.concatenate(
            (old_curvature_derivative1, old_curvature_derivative2)
        )

        res.update(k, u[indexr], u[indexl], u[8], u[9], fc[k], mc[k])
        u_old = np.array(u)

    return res
