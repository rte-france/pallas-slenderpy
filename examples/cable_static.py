import matplotlib.pyplot as plt
import numpy as np

from slenderpy.future._constant import _GRAVITY
from slenderpy.future.cable.static import blondel
from slenderpy.future.cable.static import catenary
from slenderpy.future.cable.static import nleq
from slenderpy.future.cable.static import parabolic


def _aster570():
    linm = 1.571
    axs = 3.653e07
    rts = 1.853e05
    alpha = 2.300e-05
    return linm, axs, rts, alpha


def parabolic_vs_catenary_vs_nleq():
    """Check differences between parabolic and nleq models when everything
    except mechanical tension is set."""

    # conductor properties
    linm, axs, rts, _ = _aster570()

    # span properties
    lspan = 400
    tension = rts * np.array([0.02, 0.05, 0.1, 0.2])
    sld = 20.0

    # position (parabolic)
    xa1 = np.linspace(0, lspan * np.ones_like(tension), 401)
    ya1 = parabolic.shape(xa1, lspan, tension, sld, linm)

    # position (catenary)
    xa2 = np.array(xa1)
    ya2 = catenary.shape(xa1, lspan, tension, sld, linm)

    # position (nleq)
    lcab, lve = nleq.solve(lspan, tension, sld, linm, axs, rtol=1.0e-99, maxiter=2)
    s = np.linspace(0, lcab * np.ones_like(tension), 401)
    xa3, ya3 = nleq.shape(s, lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)

    # sag-to-length ratio
    ratio = nleq.sag(lspan, tension, sld, linm, axs) / np.sqrt(lspan**2 + sld**2)

    plt.figure()
    plt.title(
        "Parabolic vs nleq shapes for different tensions (r is sag to span length ratio)"
    )
    for n in range(len(tension)):
        plt.plot(
            xa1[:, n], ya1[:, n], c=f"C{n}", ls="-", label=f"parabolic r={ratio[n]:.3f}"
        )
        plt.plot(xa2[:, n], ya2[:, n], c=f"C{n}", ls="--", label="catenary")
        plt.plot(xa3[:, n], ya3[:, n], c=f"C{n}", ls=":", lw=2, label="nleq")
    plt.grid(True)
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.legend()

    return


def compare_all(lspan=400.0, ratio=0.25, sld=0.0):
    """Quick plot script to make visual check"""

    # lspan = 100.
    # ratio = 0.01
    # sld = 10.

    linm, axs, rts, _ = _aster570()
    tension = ratio * rts

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
    # stress
    sx = np.max(parabolic.stress(xs, lspan, tension, sld, linm))
    sa = parabolic.mean_stress(lspan, tension, sld, linm)

    # [nleq] compute cable stuff

    # solve equilibrium, get cable length and static shape
    lcab, lve = nleq.solve(lspan, tension, sld, linm, axs)
    xn, yn = nleq.shape(1001, lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)
    # cable length
    length_ = nleq.length(lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)
    # cable sag
    sag_ = nleq.sag(lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)
    # sag points
    lt = nleq.argsag(lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)
    xt, yt = nleq.shape(lt, lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)
    # stress
    sx_ = np.max(nleq.stress(1001, lspan, tension, sld, linm, axs, lcab=lcab, lve=lve))
    sa_ = nleq.mean_stress(lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)

    # [catenary] compute cable stuff

    # static shape
    xc = np.linspace(0, 1, 101)
    yc = catenary.shape(xc, lspan, tension, sld, linm)
    # cable length
    lc = catenary.length(lspan, tension, sld, linm)
    # cable sag
    sagc = catenary.sag(lspan, tension, sld, linm)
    # sag points
    xu = catenary.argsag(lspan, tension, sld, linm)
    yu = catenary.shape(xu, lspan, tension, sld, linm)
    # stress
    nn = float("nan")
    # > not implemented yet

    # [print]
    print(f"[all] flat length  {lspan:.6f}")
    print(f"[all] 3dim length  {np.sqrt(lspan**2 + sld**2):.6f}")
    print(f"[pbl] cable length {length:.6f}")
    print(f"[cat] cable length {lc:.6f}")
    print(f"[nle] cable len 0  {lcab:.6f}")
    print(f"[nle] cable len 1  {length_:.6f}")
    print(f"[pbl] sag          {sag:.6f}")
    print(f"[cat] sag          {sagc:.6f}")
    print(f"[nle] sag          {sag_:.6f}")
    print(f"[pbl] max stress   {sx:.6f}")
    print(f"[cat] max stress   {nn:.6f}")
    print(f"[nle] max stress   {sx_:.6f}")
    print(f"[pbl] avg stress   {sa:.6f}")
    print(f"[cat] avg stress   {nn:.6f}")
    print(f"[nle] avg stress   {sa_:.6f}")

    # [plot]
    plt.figure()
    plt.plot([0.0, lspan], [0, sld], ls="--", c="gray")
    plt.plot(xp, yp, label="Parabola", c="C0")
    plt.plot([xs, xs], [ys, xs * sld / lspan], ls="--", c="C0")
    plt.plot(xn, yn, label="NL eq.", c="C1")
    plt.plot([xt, xt], [yt, xt * sld / lspan], ls="--", c="C1")
    plt.plot(xc, yc, c="C2", label="catenary")
    plt.plot([xu, xu], [yu, xu * sld / lspan], ls="--", c="C2")
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid(True)
    plt.title(
        f"Span with $L_p$={lspan:.0f}, $H$={tension / 1000.:.0f} kN, $h$={sld:.0f} m, "
        f"$m$={linm:.2f} kg.m$^{-1}$, $EA$={axs * 1.E-06} MN",
        usetex=True,
    )

    return


def test_blondel():

    # conductor properties
    linm, axs, rts, alpha = _aster570()

    # span properties
    lspan = 400
    tensions = rts * np.array([0.02, 0.05, 0.1, 0.2])
    slds = np.array([0.0, 20.0, 50.0, 100.0])

    # temperatures
    Tref = 15.0
    Tend = Tref + np.linspace(-20, +80, 101)
    dt = Tend - Tref

    #
    fig, ax = plt.subplots(nrows=len(slds), ncols=len(tensions))
    for i in range(len(slds)):
        for j in range(len(tensions)):

            sld = slds[i]
            tension = tensions[j]

            # compute new tensions
            w = linm * _GRAVITY * lspan
            tb = blondel.tension(w, tension, Tref, Tend, axs, alpha)
            tp = parabolic.thermal_expansion_tension(
                lspan, tension, sld, Tref, Tend, linm, alpha
            )
            tn = nleq.thermal_expansion_tension(
                lspan, tension, sld, Tref, Tend, linm, axs, alpha
            )

            ax[i, j].plot(dt, tb - tension, label="blondel")
            ax[i, j].plot(dt, tp - tension, label="parabolic")
            ax[i, j].plot(dt, tn - tension, label="nleq")
            ax[i, j].grid(True)
            ax[i, j].set_title(f"With H={tension/1000.} kN and h={sld} m")

    ax[-1, -1].legend()
    for i in range(len(slds)):
        ax[i, 0].set_ylabel("Tension difference (N)")
    for j in range(len(tensions)):
        ax[-1, j].set_xlabel("Temperature difference (K)")


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("TkAgg")
    plt.close("all")

    parabolic_vs_catenary_vs_nleq()

    compare_all(lspan=400.0, ratio=0.25, sld=10.0)
    print()
    compare_all(lspan=100.0, ratio=0.01, sld=15.0)
    #
    test_blondel()
