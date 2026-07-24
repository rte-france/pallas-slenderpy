import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD
from slenderpy import simtools
from slenderpy.force import Excitation
from slenderpy.future.beam.beam import BeamBW
from slenderpy.future.stockbridge import (
    ClampParameters,
    MassParameters,
    MessengerCableParameters,
    Side,
    Stockbridge,
    plot_clamp,
    plot_clamp_all_versions,
    plot_mass,
    solve_dynamic_with_sb,
    solve_imposed_acceleration,
    solve_imposed_force,
    solve_linearized_imposed_force,
)
from slenderpy.wind import air_volumic_mass

MASS = MassParameters(
    length_to_clamp=0.1875,
    length_to_centroid=0.0325,
    mass=0.856,
    moment_of_inertia=0.001814,
)

CABLE = MessengerCableParameters(
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

CLAMP = ClampParameters(
    mass=0.7,
    moment_of_inertia=0.0025,
    half_length=0.03,
)


def basic_example():
    dt = 1e-3
    v_amp = 0.2
    freq = 24
    omega = 2 * np.pi * freq
    tf = 3.2
    nb_time_steps = int(tf / dt)
    t = np.linspace(0, tf, nb_time_steps)

    sb = Stockbridge(CLAMP, MASS, CABLE, MASS, CABLE, dt)
    acceleration_imposed = -v_amp * omega * np.sin(omega * t)
    acceleration_ang_imposed = np.zeros(nb_time_steps)

    ic1 = np.zeros(sb.mass_right.nb_unknowns)
    ic2 = np.zeros(sb.mass_left.nb_unknowns)
    res = solve_imposed_acceleration(
        sb, tf, ic1, ic2, acceleration_imposed, acceleration_ang_imposed, dt
    )

    plot_mass(res, Side.RIGHT)
    plot_clamp(res)
    plt.figure()
    plt.plot(res.right["curvature"][:, -1], res.right["moment_extremity"])
    plt.show()


def test_equivalence():
    dt = 1e-3
    v_amp = 0.2
    f_amp = 15
    sb = Stockbridge(CLAMP, MASS, CABLE, MASS, CABLE, dt)
    ic1 = np.zeros(sb.mass_right.nb_unknowns)
    ic2 = np.zeros(sb.mass_left.nb_unknowns)
    ic = np.zeros(sb.mass_right.nb_unknowns + sb.mass_left.nb_unknowns + 2)
    freq = np.array([2, 5, 10])
    omega = 2 * np.pi * freq
    tf = 3
    nb_time_steps = int(tf / dt)
    t = np.linspace(0, tf, nb_time_steps)
    acceleration_imposed = -v_amp * sum(k * np.sin(k * t) for k in omega)
    acceleration_ang_imposed = np.zeros(nb_time_steps)
    force_imposed = f_amp * sum(np.sin(k * t) for k in omega)
    moment_imposed = np.zeros(nb_time_steps)

    res1 = solve_imposed_acceleration(
        sb, tf, ic1, ic2, acceleration_imposed, acceleration_ang_imposed, dt
    )
    res2 = solve_imposed_force(
        sb, tf, ic, res1.general["force_clamp"], res1.general["moment_clamp"], dt
    )
    plot_clamp_all_versions(res1, res2, "acceleration_clamp", "force_clamp")

    res1 = solve_imposed_force(sb, tf, ic, force_imposed, moment_imposed, dt)
    res2 = solve_imposed_acceleration(
        sb,
        tf,
        ic1,
        ic2,
        res1.general["acceleration_clamp"],
        res1.general["acceleration_angular_clamp"],
        dt,
    )
    plot_clamp_all_versions(res1, res2, "force_clamp", "acceleration_clamp")

    plt.show()


def energy_linearized():
    k1 = 12 * CABLE.ei_max_cable / (MASS.length_to_clamp**3)
    k2 = -6 * CABLE.ei_max_cable / (MASS.length_to_clamp**2)
    k3 = 4 * CABLE.ei_max_cable / MASS.length_to_clamp
    K = np.array([[k1, k2], [k2, k3]])
    bend = 1e-4
    C = bend * K

    freq = 14
    tf = 3
    dt = 1e-3
    f_amp = 50
    sb = Stockbridge(CLAMP, MASS, CABLE, MASS, CABLE, K, C)
    ic = np.zeros(sb.mass_right.nb_unknowns + sb.mass_left.nb_unknowns + 2)
    omega = 2 * np.pi * freq
    nb_time_steps = int(tf / dt)
    t = np.linspace(0, tf, nb_time_steps)
    force_clamp = f_amp * np.sin(omega * t)
    int_force_clamp = -(f_amp / omega) * np.cos(omega * t)
    iint_force_clamp = -(f_amp / omega**2) * np.sin(omega * t)
    t_stop = 1.22
    step_stop = int(t_stop / dt)
    int_force_clamp[step_stop:] = 0
    iint_force_clamp[step_stop:] = 0
    force_clamp[step_stop:] = 0

    res = solve_linearized_imposed_force(
        sb, tf, ic, np.array([0 * t, 0 * t, force_clamp]), dt
    )

    clamp_vel = (1 / (2 * sb.mass_right.mass + sb.clamp.mass)) * (
        int_force_clamp
        - 2 * sb.mass_right.mass * res.right["mass_velocity"]
        + 2
        * sb.mass_right.mass
        * sb.mass_right.length_to_centroid
        * res.right["mass_angular_velocity"]
    )
    clamp_disp = (1 / (2 * sb.mass_right.mass + sb.clamp.mass)) * (
        iint_force_clamp
        - 2 * res.right["mass_displacement"]
        + 2
        * sb.mass_right.mass
        * sb.mass_right.length_to_centroid
        * res.right["mass_rotation"]
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

    a, b = sb.ab
    M = np.zeros((5, 5))
    M[4, 4] = CLAMP.mass + 2 * sb.mass_right.mass
    M[:4, 4] = a
    M[4, :4] = a.T
    M[:2, :2] = sb.mass_right.mass_matrix
    M[2:4, 2:4] = sb.mass_right.mass_matrix
    K_tot = np.zeros((5, 5))
    K_tot[:2, :2] = K
    K_tot[2:4, 2:4] = K
    C_tot = bend * K_tot

    P_diss = np.zeros(nb_time_steps)
    E_kin = np.zeros(nb_time_steps)
    E_pot = np.zeros(nb_time_steps)
    for k in range(nb_time_steps):
        P_diss[k] = V[:, k].T @ C_tot @ V[:, k]
        E_kin[k] = 0.5 * V[:, k].T @ M @ V[:, k]
        E_pot[k] = 0.5 * X[:, k].T @ K_tot @ X[:, k]

    P_kin = np.gradient(E_kin, dt)
    P_pot = np.gradient(E_pot, dt)
    P_ext = clamp_vel * res.general["force_clamp"]

    E_diss = sp.integrate.cumulative_simpson(P_diss, dx=dt, initial=0)
    E_ext = np.zeros(nb_time_steps)
    E_ext[:step_stop] = sp.integrate.cumulative_simpson(
        P_ext[:step_stop], dx=dt, initial=E_kin[0]
    )
    E_ext[step_stop:] = sp.integrate.cumulative_simpson(
        P_ext[step_stop:],
        dx=dt,
        initial=E_kin[step_stop] + E_pot[step_stop] + E_diss[step_stop],
    )
    E_diss = sp.integrate.cumulative_simpson(P_diss, dx=dt, initial=0)

    plt.figure()
    plt.title("power")
    plt.plot(t, P_kin, label="kin")
    plt.plot(t, P_pot, label="pot")
    plt.plot(t, P_diss, label="dissip")
    plt.plot(t, P_ext, label="ext")
    plt.plot(t, P_kin + P_pot + P_diss - P_ext, label="total")
    plt.legend()

    plt.figure()
    plt.title("energy")
    plt.plot(t, E_kin, label="kin")
    plt.plot(t, E_pot, label="pot")
    plt.plot(t, E_diss, label="dissip")
    plt.plot(t, E_ext, label="ext")
    plt.plot(t, E_kin + E_pot + E_diss - E_ext, label="total")
    plt.legend()

    plt.show()


def coupling():
    # Cable (ASTER 570)
    LSPAN = 440.0
    TENSION = 39e3
    CABLE_MASS = 1.57
    EI_MAX = 2155.07
    EI_MIN = 28.28
    CHI0 = 0.03
    DIAMETER = 31.1e-3

    # Aeolian excitation parameters
    MODE = 25
    STROUHAL = 0.2
    CL0 = 0.6
    TF = 10.0

    beamBW = BeamBW(
        length=LSPAN,
        boundary_conditions=FD.rot_free(0, 0, 0, 0),
        tension=TENSION,
        mass=CABLE_MASS,
        ei_max=EI_MAX,
        ei_min=EI_MIN,
        critical_curvature=CHI0,
    )

    freq = beamBW.natural_frequencies_rot_free(MODE, EI_MAX)[-1]
    nb_space = 20 * MODE
    dt = min(0.01 / freq, 1e-3)
    dr = 5 * dt
    parameters = simtools.Parameters(
        ns=nb_space, tf=TF, dt=dt, dr=dr, los=nb_space, pp=True
    )
    x = np.linspace(0, LSPAN, nb_space)

    wind_speed = DIAMETER * MODE * beamBW.natural_frequency() / STROUHAL
    estimated_amplitude = (
        LSPAN * 0.5 * air_volumic_mass() * DIAMETER * CL0 * wind_speed**2
    )

    def force(x, t, y, v):
        return Excitation(
            f=freq,
            a=4 * estimated_amplitude,
            s=(2 * MODE - 1) * (LSPAN + 0.5) / (2 * MODE),
            L=LSPAN,
            tf=TF,
            gravity=True,
            m=CABLE_MASS,
        )(x, t)[0]

    sol_static = beamBW.solve_static(
        n=nb_space, rhs=force(x, 0.0, None, 0), approx_curvature=False
    )

    res_simple = beamBW.solve_dynamic(
        parameters=parameters,
        initial_position=sol_static,
        initial_velocity=np.zeros(nb_space),
        force=force,
        approx_curvature=False,
        it_picard=20,
        tol_picard=1e-3,
        zeta=0,
    )

    pos_stockbridge = LSPAN / (2 * MODE)
    id_pos_stockbridge = max(
        1, min(nb_space - 2, int(np.round(pos_stockbridge / LSPAN * (nb_space - 1))))
    )

    force_array = np.zeros((nb_space, parameters.nt))
    for it_t in range(parameters.nt):
        force_array[:, it_t] = force(x, it_t * dt, None, None)

    sb = Stockbridge(CLAMP, MASS, CABLE, MASS, CABLE)
    ic1 = np.zeros(sb.mass_right.nb_unknowns)
    ic2 = np.zeros(sb.mass_left.nb_unknowns)
    sb_dict = {
        "sb1": {
            "stockbridge": sb,
            "position": pos_stockbridge,
            "initial condition right": ic1,
            "initial condition left": ic2,
        },
        "sb2": {
            "stockbridge": sb,
            "position": 5 * pos_stockbridge,
            "initial condition right": ic1,
            "initial condition left": ic2,
        },
    }

    res_cable, res_sb = solve_dynamic_with_sb(
        sb_dict,
        beamBW,
        parameters,
        initial_position=sol_static,
        initial_velocity=np.zeros(nb_space),
        force=force_array,
        approx_curvature=False,
        it_picard=20,
        tol_picard=1e-3,
        zeta=0,
    )

    t = res_cable["time"]

    plt.figure()
    plt.title("max-min over the time")
    plt.plot(
        x,
        np.max(res_simple["y"] - res_simple["y"][0, :], axis=0)
        - np.min(res_simple["y"] - res_simple["y"][0, :], axis=0),
        label="No sb",
        color="blue",
    )
    plt.plot(
        x,
        np.max(res_cable["y"] - res_cable["y"][0, :], axis=0)
        - np.min(res_cable["y"] - res_cable["y"][0, :], axis=0),
        label="2 sb",
        color="orange",
    )
    plt.xlabel("position (m)")
    plt.ylabel("deviation from the equilibrium position")
    plt.legend()

    plt.figure()
    plt.title("At first stockbridge location")
    plt.plot(
        t,
        res_simple["y"][:, id_pos_stockbridge] - res_simple["y"][0, id_pos_stockbridge],
        label="No sb",
        color="blue",
    )
    plt.plot(
        t,
        res_cable["y"][:, id_pos_stockbridge] - res_cable["y"][0, id_pos_stockbridge],
        label="2 sb",
        color="orange",
    )
    plt.xlabel("time (s)")
    plt.ylabel("deviation from the equilibrium position")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    basic_example()
    test_equivalence()
    energy_linearized()
    coupling()
