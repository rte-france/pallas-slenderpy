import matplotlib.pyplot as plt
import numpy as np

from slenderpy.future.cable.static import parabolic


def parabola_plot():
    """Quick plot script to make visual check"""

    # cable data

    lspan = 100.
    tension = 0.012 * 1.85E+05
    sld = 10.
    linm = 1.57
    # axs = 3.65E+07

    # compute cable stuff

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

    # plot
    plt.figure()
    plt.plot(xp, xp * sld / lspan, ls='--', c='gray')
    plt.plot(xp, yp, label='Parabola', c='C0')
    plt.plot([xs, xs], [ys, xs * sld / lspan], ls='--', c='C0')

    plt.legend()
    plt.grid(True)

    return


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    plt.close('all')

    parabola_plot()
