# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from slenderpy._constant import _GRAVITY


# import slenderpy.cable.staticp as scsp


def parabola_plot():
    """Quick plot script to make visual check"""

    # cable data

    lspan = 100.
    tension = 0.012 * 1.85E+05
    sld = 10.
    linm = 1.57
    axs = 3.65E+07

    # compute cable stuff

    # linear weight
    linw = linm * _GRAVITY
    # static shape
    xp = np.linspace(0, lspan, 1001)
    yp = scsp.ypos(xp, lspan, tension, sld, linw)
    # cable length
    length = scsp.length(lspan, tension, sld, linm)
    # cable sag
    sag = scsp.sag(lspan, tension, sld, linm)
    # sag points
    xs = scsp.argsag(lspan, tension, sld, linm)
    ys = scsp.ypos(xs, lspan, tension, sld, linw)

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
