import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from slenderpy.future.beam.beam import Beam, BeamEIVariable
import slenderpy.future.beam.fd_utils as FD
from slenderpy import simtools 
from slenderpy.future.cable.static.catenary import shape
from slenderpy.future._constant import _GRAVITY

def _plot_animation(x, sol_static, sol_dynamic, ymin, ymax, nb_time):
    """Animation to plot the analytical and the numerical solution."""

    fig = plt.figure()
    (line_static,) = plt.plot([], [], '--', color="orange", label="Static")
    (line_dynamic,) = plt.plot([], [], color="blue", label="Dynamic")
    plt.legend()
    plt.xlim(x[0], x[-1])
    plt.ylim(ymin, ymax)

    def animate(i):
        line_static.set_data(x,sol_static)
        line_dynamic.set_data(x, sol_dynamic[i])
        return line_static, line_dynamic

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=np.arange(0, nb_time + 1),
        interval=1,
        blit=True,
        repeat=True,
    )
    # ani.save("static_g_dynamic.mp4", writer="ffmpeg", fps=200)
    plt.show()

def static_gravity():
    lspan = 440
    nb_space = 400
    x = np.linspace(0, 440, nb_space)
    final_time = 10.
    dt = 1e-2

    tension = 39e3
    mass = 1.57
    ei_max = 2155.07
    ei_min = 28.28
    chi0 = 0.03

    bc = FD.rot_none(0,0,0,0)
    rhs = -10.*np.ones(nb_space)*mass
    beam = BeamEIVariable(length=lspan, boundary_condition=bc, tension=tension, mass=mass, ei_max=ei_max, ei_min=ei_min, critical_curvature=chi0)
    sol_static = beam.solve_static(n=nb_space, rhs=rhs, approx_curvature=False)

    def force(x,t,y,v):
        return -_GRAVITY*np.ones(nb_space)*mass

    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-2, los=nb_space
    )

    sol_dynamic = beam.solve_dynamic(
        parameters=parameters,
        initial_position=sol_static,
        initial_velocity=np.zeros(nb_space),
        initial_bending_moment=np.ones(nb_space),
        force=force,
        approx_curvature=False,
        it_picard=3,
        tol_picard=1e-3
    )

    y = sol_dynamic.data["y"]

    _plot_animation(x, sol_static, y, -10, 2, parameters.nr)

def dynamic_excitation():
    lspan = 440
    nb_space = 400
    x = np.linspace(0, 440, nb_space)
    final_time = 10.
    dt = 1e-2

    tension = 39e3
    mass = 1.57
    ei_max = 2155.07
    ei_min = 28.28
    chi0 = 0.03

    bc = FD.rot_none(0,0,0,0)
    rhs = -_GRAVITY*np.ones(nb_space)*mass
    beam = BeamEIVariable(length=lspan, boundary_condition=bc, tension=tension, mass=mass, ei_max=ei_max, ei_min=ei_min, critical_curvature=chi0)
    sol_static = beam.solve_static(n=nb_space, rhs=rhs, approx_curvature=False)

    freq = 1
    def force(x,t,y,v):
        return -_GRAVITY*np.ones(nb_space)*mass + np.sin(2*np.pi*freq*t)*np.ones(nb_space)

    parameters = simtools.Parameters(
        ns=nb_space, tf=final_time, dt=dt, dr=1e-2, los=nb_space
    )

    sol_dynamic = beam.solve_dynamic(
        parameters=parameters,
        initial_position=sol_static,
        initial_velocity=np.zeros(nb_space),
        initial_bending_moment=np.ones(nb_space),
        force=force,
        approx_curvature=False,
        it_picard=3,
        tol_picard=1e-3
    )

    y = sol_dynamic.data["y"]
    c = sol_dynamic.data["c"]
    M = sol_dynamic.data["M"]

    c_hysteresis = []
    M_hysteresis = []

    for k in range(1,parameters.nr):
        c_hysteresis.append(c[k][nb_space//2])
        M_hysteresis.append(M[k][nb_space//2])
    
    _plot_animation(x, sol_static, y, -10, 2, parameters.nr)

    plt.plot(c_hysteresis,M_hysteresis,'o')
    plt.show()


if __name__ == "__main__":
    static_gravity()
    dynamic_excitation()
