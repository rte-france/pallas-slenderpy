import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import slenderpy.future.beam.bending as BD
import slenderpy.future.beam.curvature as CV
from slenderpy import simtools
from slenderpy.future._constant import _GRAVITY
from slenderpy.future.beam.dynamic import solve_dynamic
from slenderpy.future.beam.static.shape import solve
from slenderpy.future.boundary_condition import hinged
from slenderpy.future.components import Conductor, Span


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

    # Keep a reference alive so the animation is not garbage-collected.
    _ani = animation.FuncAnimation(
        fig,
        animate,
        frames=np.arange(0, nb_time + 1),
        interval=1,
        blit=False,
        repeat=True,
    )
    # _ani.save('test_case__bretelle.mp4', writer='ffmpeg', fps=50)
    plt.show()


conductor = Conductor(
    mass=1.57,
    ei_min=28.28,
    ei_max=2155.07,
    beta_flexion=6.438e-07,
)
span = Span(length=440.0, tension=39e3, boundary_conditions=hinged())


def static_gravity():
    nb_space = 400
    x = np.linspace(0, 440, nb_space)
    final_time = 10.0
    dt = 1e-2
    dr = 1e-1

    def force(x, t, y, v):
        return -_GRAVITY * np.ones(nb_space) * conductor.mass

    rhs = force(None, None, None, None)

    sol_static = solve(
        conductor,
        span,
        rhs=rhs,
        n=nb_space,
        model="varying",
        approx_curvature=True,
    )

    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=dr, los=nb_space, pp=True
    )

    sol_dynamic = solve_dynamic(
        conductor,
        span,
        model="varying",
        parameters=parameters,
        initial_position=sol_static,
        initial_velocity=np.zeros(nb_space),
        force=force,
        approx_curvature=False,
    )

    y = sol_dynamic["y"]

    _plot_animation(x, sol_static, y, -10, 2, parameters.nr, dr)


def hyteresis():
    nb_space = 600
    x = np.linspace(0, 440, nb_space)
    final_time = 1.0
    dt = 1e-3
    dr = 1e-3

    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=dr, los=nb_space, pp=True
    )

    def force(x, t, y, v):
        return np.zeros(nb_space)

    ds = span.length / (nb_space - 1)

    freq = 15
    y_initial = 3 * np.sin(2 * np.pi * freq * x / span.length)

    operator = CV.create(nb_space, ds, approx_curvature=False)
    bending = BD.create(conductor, span, "varying")
    initial_curvature = operator.value(y_initial)
    initial_moment = np.sign(initial_curvature) * bending.moment(
        np.abs(initial_curvature)
    )
    c0 = np.max(np.abs(initial_curvature))
    M0 = np.max(np.abs(initial_moment))

    pos = nb_space // (4 * freq)

    res = solve_dynamic(
        conductor,
        span,
        model="varying",
        parameters=parameters,
        initial_position=y_initial,
        initial_velocity=np.zeros(nb_space),
        force=force,
        approx_curvature=False,
    )

    y = res["y"]
    c = res["c"]
    M = res["M"]

    c1 = np.linspace(0, c0, 50)
    M1 = bending.moment(c1)
    plt.plot(c[:, pos], M[:, pos], label="Hysteresis")
    plt.plot(2 * c1 - c0, 2 * M1 - M0, color="orange", label="theoritical")
    plt.plot(2 * c1 - c0, -np.flip(2 * M1 - M0), color="orange")
    plt.xlabel("Curvature")
    plt.ylabel("Bending moment")
    plt.legend()

    _plot_animation(x, y_initial, y, -5, 5, parameters.nr, dr)


# Add energies in new beam module
# def energy():
#     lspan = 440
#     nb_space = 440
#     x = np.linspace(0, 440, nb_space)
#     final_time = 5.0
#     dt = 1e-3
#     dr = 1e-2

#     tension = 39e3
#     mass = 1.57
#     ei_max = 2155.07
#     ei_min = 28.28
#     chi0 = 0.03

#     def force(x, t, y, v):
#         return -_GRAVITY * np.ones(nb_space) * mass

#     bc = hinged(0, 0, 0, 0)
#     rhs = -10 * np.ones(nb_space) * mass
#     beam = BeamBW(
#         length=lspan,
#         boundary_conditions=bc,
#         tension=tension,
#         mass=mass,
#         ei_min=ei_min,
#         ei_max=ei_max,
#         critical_curvature=chi0,
#     )

#     sol_static = beam.solve_static(n=nb_space, rhs=rhs, approx_curvature=False)

#     parameters = simtools.Parameters(
#         ns=nb_space, tf=final_time, dt=dt, dr=dr, los=nb_space, pp=True
#     )

#     res = beam.solve_dynamic(
#         parameters=parameters,
#         initial_position=sol_static,
#         initial_velocity=np.zeros(nb_space),
#         force=force,
#         approx_curvature=False,
#         it_picard=10,
#         tol_picard=1e-4,
#     )

#     y = res["y"]

#     e_kin = res["e_kin"]
#     e_bend = res["e_bend"]
#     e_dissip = res["e_dissip"]
#     e_tens = res["e_tens"]
#     e_ext = res["e_ext"]
#     e_bound_tens = res["e_bound_tens"]

#     times = parameters.time_vector_output()
#     labels = ["kinetic", "bending", "e_dissip", "tension", "exterior"]

#     plt.figure()
#     plt.plot(times, res["p_kin"], label="kinetic")
#     plt.plot(times, res["p_bend"], label="bending")
#     plt.plot(times, res["p_dissip"], label="dissip")
#     plt.plot(times, res["p_tens"], label="tension")
#     plt.plot(times, res["p_ext"], label="exterior")
#     plt.legend()
#     plt.title("power")

#     plt.figure()
#     plt.plot(times, e_kin, label="kinetic")
#     plt.plot(times, e_bend, label="bending")
#     plt.plot(times, e_dissip, label="dissip")
#     plt.plot(times, e_tens, label="tension")
#     plt.plot(times, e_ext, label="exterior")
#     plt.plot(times, e_bound_tens, label="bound tension")
#     plt.plot(
#         times, e_kin + e_bend + e_dissip + e_tens - e_ext - e_bound_tens, label="total"
#     )
#     plt.legend()
#     plt.title("energy")

#     plt.figure()
#     plt.stackplot(times, [e_kin, e_bend, e_dissip, e_tens, -e_ext], labels=labels)
#     plt.legend()

#     _plot_animation(x, sol_static, y, -10, 1, parameters.nr, dr)


def bretelle():
    span = Span(length=1.53, tension=20.0, boundary_conditions=hinged())
    conductor = Conductor(
        mass=2.879, ei_min=67.7, ei_max=5089.0, beta_flexion=2.0e-5 / 20.0
    )
    nb_space = 100
    x = np.linspace(0, span.length, nb_space)
    final_time = 0.2
    dt = 1e-6
    dr = 1e-3

    def force(x, t, y, v):
        return -_GRAVITY * np.ones(nb_space) * conductor.mass

    rhs = force(None, None, None, None)

    sol_static = solve(
        conductor,
        span,
        rhs=rhs,
        n=nb_space,
        model="varying",
        approx_curvature=False,
    )

    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=dr, los=nb_space, pp=True
    )

    res = solve_dynamic(
        conductor,
        span,
        model="varying",
        parameters=parameters,
        initial_position=sol_static,
        initial_velocity=0.8 * np.sin(2 * np.pi * x / span.length),
        force=force,
        approx_curvature=False,
    )

    y = res["y"]
    c = res["c"]
    # e_kin = res["e_kin"]
    # e_bend = res["e_bend"]
    # e_dissip = res["e_dissip"]
    # e_tens = res["e_tens"]
    # e_ext = res["e_ext"]
    # e_bound_tens = res["e_bound_tens"]

    times = parameters.time_vector_output()
    # labels = ["kinetic", "bending", "dissip", "tension", "exterior"]

    # plt.figure()
    # plt.plot(times, res["p_kin"], label="kinetic")
    # plt.plot(times, res["p_bend"], label="bending")
    # plt.plot(times, res["p_dissip"], label="dissip")
    # plt.plot(times, res["p_tens"], label="tension")
    # plt.plot(times, res["p_ext"], label="exterior")
    # plt.plot(times, res["p_bound_tens"], label="boud, tension")
    # plt.plot(
    #     times,
    #     res["p_kin"]
    #     + res["p_bend"]
    #     + res["p_tens"]
    #     + res["p_dissip"]
    #     - res["p_ext"]
    #     - res["p_bound_tens"],
    #     label="total",
    # )
    # plt.legend()
    # plt.title("power")

    # plt.figure()
    # plt.plot(times, e_kin, label="kinetic")
    # plt.plot(times, e_bend, label="bending")
    # plt.plot(times, e_dissip, label="dissip")
    # plt.plot(times, e_tens, label="tension")
    # plt.plot(times, e_ext, label="exterior")
    # plt.plot(times, e_bound_tens, label="boud, tension")
    # plt.plot(
    #     times, e_kin + e_dissip + e_bend + e_tens - e_ext - e_bound_tens, label="total"
    # )
    # plt.legend()
    # plt.title("energy")

    # plt.figure()
    # plt.stackplot(times, [e_kin, e_bend, e_dissip, e_tens, -e_ext], labels=labels)
    # plt.legend()
    # plt.title("Global energy balance")

    pos = nb_space // 4
    plt.figure()
    plt.plot(times, c[:, pos], label="curvature")
    plt.title("Curvature in function of time")

    _plot_animation(x, sol_static, y, -4e-2, 3e-2, parameters.nr, dr)


def damping():
    nb_space = 500
    final_time = 30.0
    dt = 1e-2
    dr = 1e-1

    def force(x, t, y, v):
        return -_GRAVITY * np.ones(nb_space) * conductor.mass

    rhs = -10 * np.ones(nb_space) * conductor.mass

    sol_static_more_gravity = solve(
        conductor,
        span,
        rhs=rhs,
        n=nb_space,
        model="constant",
        approx_curvature=False,
    )

    sol_static_proper_gravity = solve(
        conductor,
        span,
        rhs=force(None, None, None, None),
        n=nb_space,
        model="constant",
        approx_curvature=False,
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
        sol_dynamic = solve_dynamic(
            conductor,
            span,
            model="constant",
            parameters=parameters,
            initial_position=sol_static_more_gravity,
            initial_velocity=np.zeros(nb_space),
            force=force,
            approx_curvature=True,
            zeta=zeta,
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
    # energy()
    bretelle()
    damping()
