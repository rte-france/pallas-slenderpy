import matplotlib.pyplot as plt
import numpy as np

from slenderpy.future.cable.static import nleq
from slenderpy.future.cable.static import parabolic


def _aster570():
    linm = 1.571
    axs = 3.653e07
    rts = 1.853e05
    return linm, axs, rts


def parabolic_vs_nleq():
    """Check differences between parabolic and nleq models when everything
    except mechanical tension is set."""

    # conductor properties
    linm, axs, rts = _aster570()

    # span properties
    lspan = 400
    tension = rts * np.array([0.02, 0.05, 0.1, 0.2])
    sld = 20.0

    # position (parabolic)
    xa1 = np.linspace(0, lspan * np.ones_like(tension), 401)
    ya1 = parabolic.shape(xa1, lspan, tension, sld, linm)

    # position (nleq)
    lcab, lve = nleq.solve(lspan, tension, sld, linm, axs, rtol=1.0e-99, maxiter=2)
    s = np.linspace(0, lcab * np.ones_like(tension), 401)
    xa2, ya2 = nleq.shape(s, lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)

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
        plt.plot(xa2[:, n], ya2[:, n], c=f"C{n}", ls="--", label="nleq")
    plt.grid(True)
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.legend()

    return


def compare_all(lspan=400.0, ratio=0.25, sld=0.0):
    """Quick plot script to make visual check"""

    # NB: we can see that the sag/argsag functions in cable are max chord and not sag ...

    # lspan = 100.
    # ratio = 0.01
    # sld = 10.

    linm, axs, rts = _aster570()
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

    # [old] [catenary]
    from slenderpy.future._constant import _GRAVITY
    import slenderpy.cable as sc

    # static shape
    s = np.linspace(0, 1, 101)
    a = tension / (linm * _GRAVITY)
    y = sc._alts(s, lspan, a, sld)

    # sag points
    xc = sc.argsag(lspan, a, sld) * lspan
    yc = sc._alts(xc / lspan, lspan, a, sld)

    # [print]
    print(f"[all] flat length  {lspan:.6f}")
    print(f"[all] 3dim length  {np.sqrt(lspan**2 + sld**2):.6f}")
    print(f"[pbl] cable length {length:.6f}")
    print(f"[cat] cable length {sc.catenary_length(lspan, a, sld):.6f}")
    print(f"[nle] cable len 0  {lcab:.6f}")
    print(f"[nle] cable len 1  {length_:.6f}")
    print(f"[pbl] sag          {sag:.6f}")
    print(f"[cat] sag          {sc.sag(lspan, a, sld):.6f}")
    print(f"[nle] sag          {sag_:.6f}")
    print(f"[pbl] max stress   {sx:.6f}")
    print(f"[nle] max stress   {sx_:.6f}")
    print(f"[pbl] avg stress   {sa:.6f}")
    print(f"[nle] avg stress   {sa_:.6f}")

    # [plot]
    plt.figure()
    plt.plot([0.0, lspan], [0, sld], ls="--", c="gray")
    plt.plot(xp, yp, label="Parabola", c="C0")
    plt.plot([xs, xs], [ys, xs * sld / lspan], ls="--", c="C0")
    plt.plot(xn, yn, label="NL eq.", c="C1")
    plt.plot([xt, xt], [yt, xt * sld / lspan], ls="--", c="C1")
    plt.plot(s * lspan, y, c="C2", label="catenary")
    plt.plot([xc, xc], [yc, xc * sld / lspan], ls="--", c="C2")
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


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("TkAgg")
    plt.close("all")

    parabolic_vs_nleq()
    compare_all(lspan=400.0, ratio=0.25, sld=10.0)
    print()
    compare_all(lspan=100.0, ratio=0.01, sld=15.0)
