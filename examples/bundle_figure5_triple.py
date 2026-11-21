"""Triple bundle in triangular arrangement: free vibration with full coupling.

Reproduces the geometric layout of Figure 5 (right) of the Eurobios/RTE note
ESCB-2020-11-12: three conductors placed at the vertices of an equilateral
triangle (side 0.40 m), held by a single mid-span spacer with three edges
that link every pair of conductors. Cable 0 is released from a Gaussian
initial deflection; energy reaches cables 1 and 2 only through the spacer.

Run:
    python examples/bundle_figure5_triple.py

Produces a 3-subplot figure: bundle layout sketch, RMS displacement of each
cable over time, and a mid-span snapshot showing the energy redistribution.
"""

import matplotlib.pyplot as plt
import numpy as np

from slenderpy import simulation
from slenderpy.cable.bundle.bundle import Bundle, InitialConditions
from slenderpy.cable.bundle.cable import Cable
from slenderpy.cable.bundle.spacer import Spacer, SpacerEdge


def _aster570():
    return Cable(
        mass=1.57,
        diameter=0.031,
        EA=3.76e07,
        length=200.0,
        tension=3.7e04,
        h=0.0,
    )


def run() -> None:
    cable = _aster570()

    l0_01 = 0.50  
    l0_12 = 0.3
    l0_02 = 0.70  

    
    b0, n0 = 0.0, 0.0
    
    b1, n1 = l0_01, 0.0
    
    b2 = (l0_01**2 + l0_02**2 - l0_12**2) / (2.0 * l0_01)
    n2 = np.sqrt(l0_02**2 - b2**2)

    geometry = [
        (b0, n0),
        (b1, n1),
        (b2, n2),
    ]

    k_n = 5.0e4
    c_n = 200.0
    k_b = 5.0e4
    c_b = 200.0
    is_2d = True
    delta_n=0.0
    delta_b=0.0
    j_b=200
    j_t=200
    width=0.035

    spacer = Spacer(
        s=0.5,
        edges=(
            SpacerEdge(i=0, j=1, k_n=k_n, c_n=c_n, k_b=k_b, c_b=c_b, 
                  is_2d = is_2d, delta_n = delta_n, delta_b = delta_b, 
                  j_b = j_b, j_t = j_t, width = width, l0 = l0_01),
            SpacerEdge(i=1, j=2, k_n=k_n, c_n=c_n, k_b=k_b, c_b=c_b, 
                  is_2d = is_2d, delta_n = delta_n, delta_b = delta_b, 
                  j_b = j_b, j_t = j_t, width = width, l0 = l0_12),
            SpacerEdge(i=0, j=2, k_n=k_n, c_n=c_n, k_b=k_b, c_b=c_b, 
                  is_2d = is_2d, delta_n = delta_n, delta_b = delta_b, 
                  j_b = j_b, j_t = j_t, width = width, l0 = l0_02),
        ), is_2d = is_2d, width = width
    )

    bundle = Bundle(cables=[cable, cable, cable], spacers=[spacer])

    ns = 81
    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
    params = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=6.0,
        dt=2.0e-3,
        dr=2.0e-2,
        los=los,
        pp=False,
    )

    s_grid = np.linspace(0.0, 1.0, ns)
    bump = 0.01 * np.exp(-((s_grid - 0.30) ** 2) / (2.0 * 0.05**2))
    bump[0] = 0.0
    bump[-1] = 0.0
    ic = InitialConditions(un=[bump, np.zeros(ns), np.zeros(ns)])
    is_uniform = False
    res = bundle.solve(is_uniform, params, force=None, zt=0.0, initial_conditions=ic)

    t_axis = np.asarray(res.lot())
    un_all = np.stack(
        [res.data[f"un_{i}"].values for i in range(3)], axis=0
    )  # shape (3, nt, n_los)
    rms = np.sqrt(np.mean(un_all**2, axis=2))  # shape (3, nt)

    fig = plt.figure(figsize=(14, 5))
    ax_geom = fig.add_subplot(1, 3, 1)
    ax_rms = fig.add_subplot(1, 3, 2)
    ax_snap = fig.add_subplot(1, 3, 3)

    # Bundle layout sketch (view from the support end-plate).
    for idx, (b, n) in enumerate(geometry):
        ax_geom.plot(b, n, "o", ms=18, color=f"C{idx}")
        ax_geom.annotate(
            f"cable {idx}", (b, n), xytext=(8, 8), textcoords="offset points"
        )
    # Draw spacer edges.
    for ei, ej in [(0, 1), (1, 2), (0, 2)]:
        ax_geom.plot(
            [geometry[ei][0], geometry[ej][0]],
            [geometry[ei][1], geometry[ej][1]],
            "k-",
            lw=1.0,
            alpha=0.5,
        )
    ax_geom.set_xlabel("binormal offset [m]")
    ax_geom.set_ylabel("normal offset [m]")
    ax_geom.set_title("Bundle layout (triangle, side 40 cm)")
    ax_geom.set_aspect("equal")
    ax_geom.grid(True, alpha=0.3)

    # Cables 1 and 2 are geometrically equivalent w.r.t. cable 0 in the
    # equilateral triangle (the spacer creates a symmetric coupling), so
    # their RMS curves overlap exactly. Draw cable 1 with a thicker line
    # under cable 2 so both colours remain visible.
    ax_rms.plot(t_axis, rms[1] * 1e3, label="cable 1", color="C1", lw=3.0)
    ax_rms.plot(t_axis, rms[2] * 1e3, label="cable 2", color="C2", lw=1.2)
    ax_rms.plot(t_axis, rms[0] * 1e3, label="cable 0 (excited)", color="C0", lw=1.2)
    ax_rms.set_xlabel("time [s]")
    ax_rms.set_ylabel("RMS normal displacement [mm]")
    ax_rms.set_title("Energy transfer through the 3-edge spacer")
    ax_rms.legend()
    ax_rms.grid(True, alpha=0.3)

    # Snapshots: t = 0, 1, 2, 3 s along the span (cable 0 only - showing
    # how the initial bump propagates and partly reflects).
    snap_times = [0.0, 1.0, 2.0, 3.0]
    s_phys = np.array(los) * cable.length
    for st in snap_times:
        k_snap = int(np.argmin(np.abs(t_axis - st)))
        for i in range(3):
            ax_snap.plot(
                s_phys,
                un_all[i, k_snap] * 1e3,
                color=f"C{i}",
                alpha=0.4 + 0.2 * snap_times.index(st),
                label=f"cable {i}, t={t_axis[k_snap]:.1f}s",
                # if st == snap_times[0]
                # else None,
                lw = 3.0 if i==1 else 1.2,
            )
    ax_snap.set_xlabel("s [m]")
    ax_snap.set_ylabel("normal displacement [mm]")
    ax_snap.set_title("Mid-snapshots along the span")
    ax_snap.legend(loc="lower left", fontsize=8)
    ax_snap.grid(True, alpha=0.3)

    fig.suptitle(
        "Triangular triple bundle, mid-span 3-edge spacer - "
        "Figure 5 (right) of the note"
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
