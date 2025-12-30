import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from slenderpy.future.beam.beam import BeamBW, BeamConst
import slenderpy.future.beam.fd_utils as FD
from slenderpy import simtools
from slenderpy.future._constant import _GRAVITY


def _plot_animation(x, sol_static, sol_dynamic, ymin, ymax, nb_time, dt):
    """Animation to plot the analytical and the numerical solution."""

    fig = plt.figure()
    (line_static,) = plt.plot([], [], "--", color="orange", label="Static")
    (line_dynamic,) = plt.plot([], [], color="blue", label="Dynamic")
    plt.legend()
    plt.xlim(x[0], x[-1])
    plt.ylim(ymin, ymax)
    time_text = plt.text(0.02, 0.95, "", transform=plt.gca().transAxes)

    def animate(i):
        line_static.set_data(x, sol_static)
        line_dynamic.set_data(x, sol_dynamic[i])
        time_text.set_text(f"t = {i * dt:.4f}")
        return line_static, line_dynamic

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=np.arange(0, nb_time + 1),
        interval=1,
        blit=False,
        repeat=True,
    )
    # ani.save('test_case__bretelle.mp4', writer='ffmpeg', fps=50)
    plt.show()


def static_gravity():
    lspan = 440
    nb_space = 400
    x = np.linspace(0, 440, nb_space)
    final_time = 10.0
    dt = 1e-2
    dr = 1e-1

    tension = 39e3
    mass = 1.57
    ei_max = 2155.07
    ei_min = 28.28
    chi0 = 0.03

    def force(x, t, y, v):
        return -_GRAVITY * np.ones(nb_space) * mass

    bc = FD.rot_free(0, 0, 0, 0)
    rhs = force(None, None, None, None)
    beam = BeamBW(
        length=lspan,
        boundary_conditions=bc,
        tension=tension,
        mass=mass,
        ei_max=ei_max,
        ei_min=ei_min,
        critical_curvature=chi0,
    )
    sol_static = beam.solve_static(n=nb_space, rhs=rhs, approx_curvature=False)

    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=dr, los=nb_space, pp=True
    )

    sol_dynamic = beam.solve_dynamic(
        parameters=parameters,
        initial_position=sol_static,
        initial_velocity=np.zeros(nb_space),
        force=force,
        approx_curvature=False,
        it_picard=3,
        tol_picard=1e-3,
    )

    y = sol_dynamic["y"]

    _plot_animation(x, sol_static, y, -10, 2, parameters.nr, dr)


def hyteresis():
    lspan = 440
    nb_space = 600
    x = np.linspace(0, 440, nb_space)
    final_time = 1.0
    dt = 1e-3
    dr = 1e-3

    tension = 39e3
    mass = 1.57
    ei_max = 2155.07
    ei_min = 28.28
    chi0 = 0.03

    bc = FD.rot_free(0, 0, 0, 0)
    beam = BeamBW(
        length=lspan,
        boundary_conditions=bc,
        tension=tension,
        mass=mass,
        ei_max=ei_max,
        ei_min=ei_min,
        critical_curvature=chi0,
    )

    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=dr, los=nb_space, pp=True
    )

    def force(x, t, y, v):
        return np.zeros(nb_space)

    ds = lspan / (nb_space - 1)
    D1 = FD.first_derivative(nb_space, ds)
    D2 = FD.second_derivative(nb_space, ds)

    def curvature(y):
        return D2 @ y / np.sqrt((np.ones(nb_space) + ((D1 @ y) ** 2)) ** 3)

    freq = 15
    y_initial = 3 * np.sin(2 * np.pi * freq * x / lspan)
    initial_curvature = curvature(y_initial)
    initial_moment = np.sign(initial_curvature) * beam._bending_moment(
        np.abs(initial_curvature)
    )
    c0 = np.max(np.abs(initial_curvature))
    M0 = np.max(np.abs(initial_moment))

    pos = nb_space // (4 * freq)

    res = beam.solve_dynamic(
        parameters=parameters,
        initial_position=y_initial,
        initial_velocity=np.zeros(nb_space),
        force=force,
        approx_curvature=False,
        it_picard=3,
        tol_picard=1e-3,
    )

    y = res["y"]
    c = res["c"]
    M = res["M"]

    c1 = np.linspace(0, c0, 50)
    M1 = beam._bending_moment(c1)
    plt.plot(c[:, pos], M[:, pos], label="Hysteresis")
    plt.plot(2 * c1 - c0, 2 * M1 - M0, color="orange", label="theoritical")
    plt.plot(2 * c1 - c0, -np.flip(2 * M1 - M0), color="orange")
    plt.xlabel("Curvature")
    plt.ylabel("Bending moment")
    plt.legend()

    _plot_animation(x, y_initial, y, -5, 5, parameters.nr, dr)


def energy():
    lspan = 440
    nb_space = 440
    x = np.linspace(0, 440, nb_space)
    final_time = 5.0
    dt = 1e-3
    dr = 1e-2

    tension = 39e3
    mass = 1.57
    ei_max = 2155.07
    ei_min = 28.28
    chi0 = 0.03

    def force(x, t, y, v):
        return -_GRAVITY * np.ones(nb_space) * mass

    bc = FD.rot_free(0, 0, 0, 0)
    rhs = -10 * np.ones(nb_space) * mass
    beam = BeamBW(
        length=lspan,
        boundary_conditions=bc,
        tension=tension,
        mass=mass,
        ei_min=ei_min,
        ei_max=ei_max,
        critical_curvature=chi0,
    )

    sol_static = beam.solve_static(n=nb_space, rhs=rhs, approx_curvature=False)

    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=dr, los=nb_space, pp=True
    )

    res = beam.solve_dynamic(
        parameters=parameters,
        initial_position=sol_static,
        initial_velocity=np.zeros(nb_space),
        force=force,
        approx_curvature=False,
        it_picard=10,
        tol_picard=1e-4,
    )

    y = res["y"]

    e_kin = res["e_kin"]
    e_bend = res["e_bend"]
    e_dissip = res["e_dissip"]
    e_tens = res["e_tens"]
    e_ext = res["e_ext"]

    times = parameters.time_vector_output()
    labels = ["kinetic", "bending", "e_dissip", "tension", "exterior"]

    plt.figure()
    plt.plot(times, res["p_kin"], label="kinetic")
    plt.plot(times, res["p_bend"], label="bending")
    plt.plot(times, res["p_dissip"], label="dissip")
    plt.plot(times, res["p_tens"], label="tension")
    plt.plot(times, res["p_ext"], label="exterior")
    plt.legend()
    plt.title("power")

    plt.figure()
    plt.plot(times, e_kin, label="kinetic")
    plt.plot(times, e_bend, label="bending")
    plt.plot(times, e_dissip, label="dissip")
    plt.plot(times, e_tens, label="tension")
    plt.plot(times, e_ext, label="exterior")
    plt.legend()
    plt.title("energy")

    plt.figure()
    plt.stackplot(times, [e_kin, e_bend, e_dissip, e_tens, e_ext], labels=labels)
    plt.legend()

    _plot_animation(x, sol_static, y, -10, 1, parameters.nr, dr)


def bretelle():
    lspan = 1.53
    nb_space = 100
    x = np.linspace(0, lspan, nb_space)
    final_time = 0.2
    dt = 1e-6
    dr = 1e-3

    tension = 20.0
    mass = 2.879
    ei_max = 5089.0
    ei_min = 67.7
    chi0 = 2.0e-5

    def force(x, t, y, v):
        return -_GRAVITY * np.ones(nb_space) * mass

    bc = FD.rot_free(0, 0, 0, 0)
    rhs = force(None, None, None, None)
    beam = BeamBW(
        length=lspan,
        boundary_conditions=bc,
        tension=tension,
        mass=mass,
        ei_min=ei_min,
        ei_max=ei_max,
        critical_curvature=chi0,
    )
    sol_static = beam.solve_static(n=nb_space, rhs=rhs, approx_curvature=False)

    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=dr, los=nb_space, pp=True
    )

    res = beam.solve_dynamic(
        parameters=parameters,
        initial_position=sol_static,
        initial_velocity=0.8 * np.sin(2 * np.pi * x / lspan),
        force=force,
        approx_curvature=False,
        it_picard=30,
        tol_picard=1e-5,
    )

    y = res["y"]
    c = res["c"]
    e_kin = res["e_kin"]
    e_bend = res["e_bend"]
    e_dissip = res["e_dissip"]
    e_tens = res["e_tens"]
    e_ext = res["e_ext"]

    times = parameters.time_vector_output()
    labels = ["kinetic", "bending", "dissip", "tension", "exterior"]

    plt.figure()
    plt.stackplot(times, [e_kin, e_bend, e_dissip, e_tens, e_ext], labels=labels)
    plt.legend()
    plt.title("Global energy balance")

    pos = nb_space // 4
    plt.figure()
    plt.plot(times, c[:, pos], label="curvature")
    plt.title("Curvature in function of time")

    _plot_animation(x, sol_static, y, -4e-2, 3e-2, parameters.nr, dr)


def damping():
    lspan = 440
    nb_space = 500
    x = np.linspace(0, 440, nb_space)
    final_time = 30.0
    dt = 1e-2
    dr = 1e-1

    tension = 39e3
    mass = 1.57
    ei = 2155.07

    def force(x, t, y, v):
        return -_GRAVITY * np.ones(nb_space) * mass

    bc = FD.rot_free(0, 0, 0, 0)
    rhs = -10 * np.ones(nb_space) * mass
    beam = BeamConst(
        length=lspan, boundary_conditions=bc, tension=tension, mass=mass, ei=ei
    )
    f0 = beam.natural_frequency()
    sol_static_more_gravity = beam.solve_static(
        n=nb_space, rhs=rhs, approx_curvature=False
    )
    sol_static_proper_gravity = beam.solve_static(
        n=nb_space, rhs=force(None, None, None, None), approx_curvature=False
    )

    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=dr, los=nb_space, pp=True
    )

    time = parameters.time_vector_output()
    pos = nb_space // 2

    plt.figure()
    plt.plot(
        time, sol_static_proper_gravity[pos] * np.ones(len(time)), label="steady state"
    )

    for zeta in [0.3, 0.5, 1.0, 2.0]:
        sol_dynamic = beam.solve_dynamic(
            parameters=parameters,
            initial_position=sol_static_more_gravity,
            initial_velocity=np.zeros(nb_space),
            force=force,
            approx_curvature=True,
            zeta=zeta,
            it_picard=3,
            tol_picard=1e-3,
        )

        plt.plot(time, sol_dynamic["y"][:, pos], label=f"zeta={zeta}")

    plt.legend()
    plt.title("Damping effect on the beam mid-point displacement")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [m]")
    plt.show()


if __name__ == "__main__":
    static_gravity()
    hyteresis()
    energy()
    bretelle()
    damping()
