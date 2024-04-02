import matplotlib.pyplot as plt
import numpy as np

from slenderpy.future.cable.static import parabolic
from slenderpy.future.cable.static import nleq


def parabola_plot():
    """Quick plot script to make visual check"""

    # cable data

    lspan = 100.
    tension = 0.012 * 1.85E+05
    sld = 10.
    linm = 1.57
    axs = 3.65E+07

    # [parabolic] compute cable stuff

    # static shape
    xp = np.linspace(0, lspan, 1001)
    yp = parabolic.shape(xp, lspan, tension, sld, linm)
    # cable length
    length = parabolic.length(lspan, tension, sld, linm)
    # cable sag
    sag = parabolic.sag(lspan, tension, sld, linm)
    # sag points
    xs = parabolic.argsag(lspan, tension, sld, linm)
    ys = parabolic.shape(xs, lspan, tension, sld, linm)

    # print
    print(f"[parabola] cable length {length:.6f}")
    print(f"[parabola] sag {sag:.6f}")

    # [nleq] compute cable stuff

    # solve equilibrium, get cable length and static shape
    length_, lve = nleq.solve(lspan, tension, sld, linm, axs)
    xn, yn = nleq.shape(1001, lspan, tension, sld, linm, axs, lcab=length_, lve=lve)
    # cable sag
    sag_ = nleq.sag(lspan, tension, sld, linm, axs, lcab=length_, lve=lve)
    # sag points
    lt = nleq.argsag(lspan, tension, sld, linm, axs, lcab=length_, lve=lve)
    xt, yt = nleq.shape(lt, lspan, tension, sld, linm, axs, lcab=length_, lve=lve)

    # print
    print(f"[nleq] cable length {length_:.6f}")
    print(f"[nleq] sag {sag_:.6f}")

    # plot
    plt.figure()
    plt.plot([0., lspan], [0, sld], ls='--', c='gray')
    plt.plot(xp, yp, label='Parabola', c='C0')
    plt.plot([xs, xs], [ys, xs * sld / lspan], ls='--', c='C0')
    plt.plot(xn, yn, label='NL eq.', c='C1')
    plt.plot([xt, xt], [yt, xt * sld / lspan], ls='--', c='C1')
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid(True)

    return


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    plt.close('all')

    parabola_plot()
