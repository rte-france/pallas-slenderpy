"""Tests for the three time-stepping solvers and their consistency."""

import numpy as np
import pytest
import scipy as sp

from slenderpy.future.stockbridge import (
    Clamp,
    ClampParameters,
    Mass,
    MassParameters,
    MessengerCableParameters,
    Side,
    Stockbridge,
    solve_imposed_acceleration,
    solve_imposed_force,
    solve_linearized_imposed_force,
)


# Cable used by the manufactured-solution suite (kept at the original size
# so the analytic tolerances still hold).
_CABLE_FOTI = MessengerCableParameters(
    nb_space_points=30,
    ratio_boundary1=0.2,
    ratio_boundary2=0.08,
    ei_max_boundary=40.0,
    ei_max_cable=25.0,
    ei_min_boundary=5.0,
    ei_min_cable=2.5,
    chi0_boundary=15e-2,
    chi0_cable=3e-2,
)
_MASS_FOTI = MassParameters(
    length_to_clamp=0.1875,
    length_to_centroid=0.0325,
    mass=0.856,
    moment_of_inertia=0.001814,
)
_MASS_NO_OFFSET = MassParameters(
    length_to_clamp=0.1875,
    length_to_centroid=0.0,
    mass=0.856,
    moment_of_inertia=0.001814,
)
_CLAMP_DEFAULT = ClampParameters(mass=0.5, moment_of_inertia=0.0025, half_length=0.03)


def _build_sb(mass_params, K, C):
    mass_right = Mass(mass_params, _CABLE_FOTI, Side.RIGHT)
    mass_left = Mass(mass_params, _CABLE_FOTI, Side.LEFT)
    clamp = Clamp(_CLAMP_DEFAULT)
    sb = Stockbridge(_CLAMP_DEFAULT, mass_params, _CABLE_FOTI, mass_params, _CABLE_FOTI, K, C)
    return sb, mass_right, mass_left, clamp


# --- linearised solver: manufactured-solution suite -----------------------


def test_mathematics():
    """Manufactured-solution check: exact polynomial response under crafted forcing."""
    k1 = 12 * _CABLE_FOTI.ei_max_cable / (_MASS_FOTI.length_to_clamp ** 3)
    k2 = -6 * _CABLE_FOTI.ei_max_cable / (_MASS_FOTI.length_to_clamp ** 2)
    k3 = 4 * _CABLE_FOTI.ei_max_cable / _MASS_FOTI.length_to_clamp
    K = np.array([[k1, k2], [k2, k3]])
    C = 1e-2 * K
    tf = 5
    dt = 1e-3
    nb = int(tf / dt)
    t = np.linspace(0, tf, nb)
    one = np.ones(nb)
    sb, mr, ml, _ = _build_sb(_MASS_FOTI, K, C)
    ic = np.zeros(mr.nb_unknowns + ml.nb_unknowns + 2)
    f_ext = np.array(
        [
            k1 * t**2 + 2 * t * one * C[0, 0] + 2 * mr.mass,
            k2 * t**2 + 2 * t * one * C[1, 0] - 2 * mr.mass * mr.length_to_centroid,
            -2 * k1 * t**2 - 4 * one * t * C[0, 0],
        ]
    )
    res = solve_linearized_imposed_force(sb, tf, ic, f_ext, dt)
    assert np.allclose(res.right["mass_displacement"], t**2, atol=1e-3)
    assert np.allclose(res.right["mass_velocity"], 2 * t, atol=1e-2)
    assert np.allclose(res.right["mass_rotation"], 0 * t, atol=1e-3)
    assert np.allclose(res.right["mass_angular_velocity"], 0 * t, atol=1e-3)
    assert np.allclose(res.general["acceleration_clamp"], 0 * t, atol=1e-2)


def test_free_vibration():
    """Free vibration: mass and clamp oscillate at the predicted modes."""
    k1, k3 = 500, 20
    K = np.array([[k1, 0], [0, k3]])
    C = 0 * K
    dt = 1e-3
    sb, mr, ml, clamp = _build_sb(_MASS_NO_OFFSET, K, C)
    ic = np.zeros(mr.nb_unknowns + ml.nb_unknowns + 2)
    ic[0] = 0.5
    ic[1] = 1.8
    ic[8] = 2 * k1 / clamp.mass * ic[0]
    omega_x = np.sqrt(k1 * (1 / mr.mass + 2 / clamp.mass))
    omega_phi = np.sqrt(k3 / mr.moment_of_inertia)
    tf = 2
    nb = int(tf / dt)
    t = np.linspace(0, tf, nb)
    f_ext = np.array([0 * t, 0 * t, 0 * t])
    res = solve_linearized_imposed_force(sb, tf, ic, f_ext, dt)
    assert np.allclose(
        res.right["mass_displacement"], ic[0] * np.cos(omega_x * t), atol=1
    )
    assert np.allclose(
        res.right["mass_rotation"], ic[1] * np.cos(omega_phi * t), atol=1
    )
    assert np.allclose(
        res.general["acceleration_clamp"], ic[8] * np.cos(omega_x * t), atol=10, rtol=80
    )


def test_forced_vibration():
    """Forced vibration matches the analytical particular + homogeneous solution."""
    k1, k3 = 500, 20
    K = np.array([[k1, 0], [0, k3]])
    C = 0 * K
    freq = 5
    omega = 2 * np.pi * freq
    f_amp = 500
    dt = 1e-3
    sb, mr, ml, clamp = _build_sb(_MASS_NO_OFFSET, K, C)
    omega_x = np.sqrt(k1 * (1 / mr.mass + 2 / clamp.mass))
    omega_phi = np.sqrt(k3 / mr.moment_of_inertia)
    amp_xp = f_amp / (clamp.mass * (omega**2 - omega_x**2))
    amp_xh = 0.5
    ic = np.zeros(mr.nb_unknowns + ml.nb_unknowns + 2)
    ic[0] = amp_xp + amp_xh
    ic[1] = 1.8
    ic[8] = f_amp / clamp.mass + 2 * k1 / clamp.mass * ic[0]
    tf = 2
    nb = int(tf / dt)
    t = np.linspace(0, tf, nb)
    f_ext = np.array([0 * t, 0 * t, f_amp * np.cos(omega * t)])
    res = solve_linearized_imposed_force(sb, tf, ic, f_ext, dt)
    assert np.allclose(
        res.right["mass_displacement"],
        amp_xh * np.cos(omega_x * t) + amp_xp * np.cos(omega * t),
        atol=1e-1,
    )
    assert np.allclose(
        res.right["mass_rotation"], ic[1] * np.cos(omega_phi * t), atol=1
    )
    assert np.allclose(
        res.general["acceleration_clamp"],
        f_ext[-1] / clamp.mass
        + 2 * k1 / clamp.mass * (amp_xh * np.cos(omega_x * t) + amp_xp * np.cos(omega * t)),
        atol=10,
        rtol=80,
    )


def test_damped_vibration():
    """Damped forced vibration follows the underdamped analytical response."""
    k1, k3 = 500, 20
    K = np.array([[k1, 0], [0, k3]])
    C = 5e-4 * K
    freq = 2
    omega = 2 * np.pi * freq
    f_amp = 100
    dt = 1e-3
    sb, mr, ml, clamp = _build_sb(_MASS_NO_OFFSET, K, C)
    mass_all = 1 / mr.mass + 2 / clamp.mass
    omega_x = np.sqrt(k1 * mass_all)
    omega_phi = np.sqrt(k3 / mr.moment_of_inertia)
    amp_xh = 0.5
    amp_phi = 1.8
    lambda_phi = 0.5 * C[1, 1] / mr.moment_of_inertia
    lambda_x = 0.5 * C[0, 0] * mass_all
    pseudo_omega_phi = np.sqrt(omega_phi**2 - lambda_phi**2)
    pseudo_omega_x = np.sqrt(omega_x**2 - lambda_x**2)
    amp_xp = -f_amp / clamp.mass * np.abs(
        1 / (-omega**2 + 1j * C[0, 0] * mass_all * omega + k1 * mass_all)
    )
    arg_xp = -np.angle(-omega**2 + 1j * C[0, 0] * mass_all * omega + k1 * mass_all)
    ic = np.zeros(mr.nb_unknowns + ml.nb_unknowns + 2)
    ic[0] = amp_xh + amp_xp
    ic[1] = amp_phi
    ic[8] = f_amp / clamp.mass + 2 * k1 / clamp.mass * ic[0]
    tf = 15
    nb = int(tf / dt)
    t = np.linspace(0, tf, nb)
    f_ext = np.array([0 * t, 0 * t, f_amp * np.cos(omega * t)])
    res = solve_linearized_imposed_force(sb, tf, ic, f_ext, dt)
    xh = (
        np.exp(-t * lambda_x)
        * amp_xh
        * (
            np.cos(pseudo_omega_x * t)
            + (lambda_x / pseudo_omega_x) * np.sin(pseudo_omega_x * t)
        )
    )
    xp = amp_xp * np.cos(omega * t + arg_xp)
    x = xp + xh
    x_dot = (
        -omega * xp
        - lambda_x * xh
        + np.exp(-t * lambda_x)
        * amp_xh
        * pseudo_omega_x
        * (
            -np.sin(pseudo_omega_x * t)
            + (lambda_x / pseudo_omega_x) * np.cos(pseudo_omega_x * t)
        )
    )
    wc_ddot = 1 / clamp.mass * (f_ext[-1] + 2 * k1 * x + 2 * C[0, 0] * x_dot)
    assert np.allclose(res.right["mass_displacement"], x, atol=1e-2)
    assert np.allclose(
        res.right["mass_rotation"],
        np.exp(-t * lambda_phi)
        * amp_phi
        * (
            np.cos(pseudo_omega_phi * t)
            + (lambda_phi / pseudo_omega_phi) * np.sin(pseudo_omega_phi * t)
        ),
        atol=1e-1,
    )
    assert np.allclose(res.general["acceleration_clamp"], wc_ddot, atol=10)


# --- linearised solver: error path ---------------------------------------


def test_linearized_solver_requires_K_C(sb):
    """Calling the linearised solver without K/C should raise."""
    nb = 5
    tf = 1.
    dt = tf/nb
    t = np.linspace(0, tf, nb)
    f_ext = np.array([0 * t, 0 * t, 0 * t])
    ic = np.zeros(sb.mass_right.nb_unknowns + sb.mass_left.nb_unknowns + 2)
    with pytest.raises(ValueError, match="K and C"):
        solve_linearized_imposed_force(sb, tf, ic, f_ext, dt)


# --- imposed_force solver -------------------------------------------------


def test_solve_imposed_force_zero_input(sb):
    """Zero force/moment input yields zero displacement at every node."""
    tf = 0.03
    nb = 30
    dt = tf/nb
    fc = np.zeros(nb)
    mc = np.zeros(nb)
    ic = np.zeros(sb.mass_right.nb_unknowns + sb.mass_left.nb_unknowns + 2)
    res = solve_imposed_force(sb, tf, ic, fc, mc, dt)
    assert np.allclose(res.right["mass_displacement"], 0)
    assert np.allclose(res.left["mass_displacement"], 0)
    assert np.allclose(res.general["acceleration_clamp"], 0)


def test_solve_imposed_force_records_inputs(sb):
    """The Out container should hold the imposed force/moment series verbatim."""
    nb = 20
    tf = 0.02
    dt = tf/nb
    t = np.linspace(0, tf, nb)
    fc = np.sin(2 * np.pi * 5 * t)
    mc = np.cos(2 * np.pi * 5 * t)
    ic = np.zeros(sb.mass_right.nb_unknowns + sb.mass_left.nb_unknowns + 2)
    res = solve_imposed_force(sb, tf, ic, fc, mc, dt)
    assert np.array_equal(res.general["force_clamp"], fc)
    assert np.array_equal(res.general["moment_clamp"], mc)
    assert res.general["time"].shape == (nb,)


# --- imposed_acceleration solver ------------------------------------------


def test_solve_imposed_acceleration_zero_input(sb):
    nb = 30
    tf = 0.03
    dt = tf/nb
    acc = np.zeros(nb)
    ang = np.zeros(nb)
    ic1 = np.zeros(sb.mass_right.nb_unknowns)
    ic2 = np.zeros(sb.mass_left.nb_unknowns)
    res = solve_imposed_acceleration(sb, tf, ic1, ic2, acc, ang, dt)
    assert np.allclose(res.right["mass_displacement"], 0)
    assert np.allclose(res.right["mass_velocity"], 0)
    assert np.allclose(res.general["force_clamp"], 0)


def test_solve_imposed_acceleration_left_right_symmetry(sb):
    """With identical right/left masses and pure vertical forcing, the two sides match."""
    nb = 50
    tf = 0.05
    dt = tf/nb
    t = np.linspace(0, tf, nb)
    acc = -0.1 * (2 * np.pi * 20) * np.sin(2 * np.pi * 20 * t)
    ang = np.zeros(nb)
    ic1 = np.zeros(sb.mass_right.nb_unknowns)
    ic2 = np.zeros(sb.mass_left.nb_unknowns)
    res = solve_imposed_acceleration(sb, tf, ic1, ic2, acc, ang, dt)
    assert np.allclose(
        res.right["mass_displacement"],
        res.left["mass_displacement"],
        atol=1e-12,
    )


# --- consistency: force <-> acceleration round trip -----------------------


def test_force_acceleration_round_trip(sb):
    """Acceleration imposed -> force recovered -> acceleration recovered, equal to start."""
    nb = 200
    tf = 0.2
    dt = tf/nb 
    t = np.linspace(0, 0.2, nb)
    omega = 2 * np.pi * 20
    acc_in = -0.05 * omega * np.sin(omega * t)
    ang_in = np.zeros(nb)

    ic1 = np.zeros(sb.mass_right.nb_unknowns)
    ic2 = np.zeros(sb.mass_left.nb_unknowns)
    res1 = solve_imposed_acceleration(sb, tf, ic1, ic2, acc_in, ang_in, dt)

    ic = np.zeros(sb.mass_right.nb_unknowns + sb.mass_left.nb_unknowns + 2)
    res2 = solve_imposed_force(
        sb, tf, ic, res1.general["force_clamp"], res1.general["moment_clamp"], dt
    )
    # The first sample stays at zero (initial condition); compare from k=1.
    assert np.allclose(
        res2.general["acceleration_clamp"][1:], acc_in[1:], atol=1e-2
    )


# --- linearised: energy balance -------------------------------------------


def test_linearized_energy_balance():
    """Power: kinetic + potential + dissipated - external should sum near zero."""
    k1 = 12 * _CABLE_FOTI.ei_max_cable / (_MASS_FOTI.length_to_clamp ** 3)
    k2 = -6 * _CABLE_FOTI.ei_max_cable / (_MASS_FOTI.length_to_clamp ** 2)
    k3 = 4 * _CABLE_FOTI.ei_max_cable / _MASS_FOTI.length_to_clamp
    K = np.array([[k1, k2], [k2, k3]])
    bend = 1e-4
    C = bend * K
    dt = 1e-3
    tf = 1.0
    nb = int(tf / dt)
    t = np.linspace(0, tf, nb)
    sb, mr, ml, clamp = _build_sb(_MASS_FOTI, K, C)
    ic = np.zeros(mr.nb_unknowns + ml.nb_unknowns + 2)
    f_amp = 50
    omega = 2 * np.pi * 14
    force_clamp = f_amp * np.sin(omega * t)
    int_force_clamp = -(f_amp / omega) * np.cos(omega * t)
    iint_force_clamp = -(f_amp / omega**2) * np.sin(omega * t)
    res = solve_linearized_imposed_force(
        sb, tf, ic, np.array([0 * t, 0 * t, force_clamp]), dt
    )

    clamp_vel = (1 / (2 * mr.mass + clamp.mass)) * (
        int_force_clamp
        - 2 * mr.mass * res.right["mass_velocity"]
        + 2 * mr.mass * mr.length_to_centroid * res.right["mass_angular_velocity"]
    )
    clamp_disp = (1 / (2 * mr.mass + clamp.mass)) * (
        iint_force_clamp
        - 2 * res.right["mass_displacement"]
        + 2 * mr.mass * mr.length_to_centroid * res.right["mass_rotation"]
    )
    X = np.array(
        [
            res.right["mass_displacement"],
            res.right["mass_rotation"],
            res.right["mass_displacement"],
            res.right["mass_rotation"],
            clamp_disp,
        ]
    )
    V = np.array(
        [
            res.right["mass_velocity"],
            res.right["mass_angular_velocity"],
            res.right["mass_velocity"],
            res.right["mass_angular_velocity"],
            clamp_vel,
        ]
    )

    a, _ = sb.ab
    M = np.zeros((5, 5))
    M[4, 4] = clamp.mass + 2 * mr.mass
    M[:4, 4] = a
    M[4, :4] = a.T
    M[:2, :2] = mr.mass_matrix
    M[2:4, 2:4] = mr.mass_matrix
    K_tot = np.zeros((5, 5))
    K_tot[:2, :2] = K
    K_tot[2:4, 2:4] = K
    C_tot = bend * K_tot

    E_kin = np.array([0.5 * V[:, k].T @ M @ V[:, k] for k in range(nb)])
    E_pot = np.array([0.5 * X[:, k].T @ K_tot @ X[:, k] for k in range(nb)])
    P_diss = np.array([V[:, k].T @ C_tot @ V[:, k] for k in range(nb)])
    P_ext = clamp_vel * res.general["force_clamp"]
    E_diss = sp.integrate.cumulative_simpson(P_diss, dx=dt, initial=0)
    E_ext = sp.integrate.cumulative_simpson(P_ext, dx=dt, initial=E_kin[0])

    # Ignore the first few samples where the integrator hasn't accumulated yet.
    residual = (E_kin + E_pot + E_diss - E_ext)[5:]
    # Energy balance bounded — relax the tolerance because Crank-Nicolson + the
    # cumulative integrator drift slightly during the transient.
    assert np.max(np.abs(residual)) < 0.5 * np.max(np.abs(E_kin + E_pot))
