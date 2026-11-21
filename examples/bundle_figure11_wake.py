"""Triple bundle aligned with the wind: drag deficit on leeward cables.

Reproduces the topology of Figure 11 of the Eurobios/RTE note
ESCB-2020-11-12: three conductors arranged along the wind direction so
that downstream cables sit in the wake of upstream ones. The aerodynamic
loading uses :mod:`slenderpy.wake` to recover the drag/lift deficit from
the Wu et al. (2018) fit (Annex A of the note).

The setup is intentionally simpler than the schematic in the note (which
shows cables crossing along the span): three cables on a straight line
perpendicular to the span, separated by 10 d. The leeward cables see a
reduced drag and therefore drift less downwind than the windward cable.

Run:
    python examples/bundle_figure11_wake.py

Produces a 3-subplot figure with the bundle layout, the mid-span time
trace of binormal displacement, and the effective drag coefficient on
the leeward cables.
"""

import matplotlib.pyplot as plt
import numpy as np

from slenderpy import simulation, wake
from slenderpy.cable.bundle.bundle import Bundle
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
    d = cable.diameter

    # Three cables on a straight line, spaced by 10 d ~ 31 cm. Wind blows
    # along +binormal (cross-span horizontal).
    spacing = 10.0 * d
    geometry = [(0.0, 0.0), (spacing, 0.0), (2.0 * spacing, 0.0)]

    # Four spacers evenly along the span, each linking 0-1 and 1-2 so the
    # bundle stays in formation.
    k_perp = 1.0e5
    c_perp = 200.0
    spacer_positions = [0.2, 0.4, 0.6, 0.8]
    spacers = [
        Spacer(
            s=sp,
            edges=(
                SpacerEdge(i=0, j=1, k_perp=k_perp, c_perp=c_perp),
                SpacerEdge(i=1, j=2, k_perp=k_perp, c_perp=c_perp),
            ),
        )
        for sp in spacer_positions
    ]
    bundle = Bundle(cables=[cable, cable, cable], spacers=spacers)

    ns = 121
    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
    params = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=15.0,
        dt=1.0e-3,
        dr=2.0e-2,
        los=los,
        pp=False,
    )

    wind_speed = 10.0  # m/s, steady
    force = wake.aero_force_bundle(
        bundle_geometry=geometry,
        diameter=d,
        wind_speed=wind_speed,
        cd_windward=1.0,
        cl_windward=0.0,
    )

    res = bundle.solve(params, force=force, zt=2.0e-3)

    t_axis = np.asarray(res.lot())
    ub_all = np.stack(
        [res.data[f"ub_{i}"].values for i in range(3)], axis=0
    )  # (3, nt, n_los)
    # Mid-span column.
    mid_col = ub_all.shape[2] // 2
    ub_mid = ub_all[:, :, mid_col]

    # Compute effective drag at mid-span for each cable from the last
    # state (after transient): re-evaluate the wake coefficients at the
    # current relative positions.
    # last_un = [res.data[f"un_{i}"].values[-1] for i in range(3)]
    last_ub = [res.data[f"ub_{i}"].values[-1] for i in range(3)]
    cd_curve = []
    for i in range(3):
        # Use cable i as the target, compute relative wind/geometry at all
        # mid-positions. With wind along +B and zero normal offsets, the
        # streamwise distance is the cumulative spacing minus uB drifts.
        y_rel_min = np.inf
        for j in range(3):
            if i == j:
                continue
            dB = (geometry[i][0] - geometry[j][0]) + last_ub[i] - last_ub[j]
            # Only j upstream of i (positive y_rel).
            mid_dB = dB[mid_col]
            if 4.0 * d <= mid_dB <= 20.0 * d and mid_dB < y_rel_min:
                y_rel_min = mid_dB
        if np.isfinite(y_rel_min):
            cd_i, _ = wake.wu2018_coefs(y_rel_min / d, 0.0)
            cd_curve.append(float(cd_i))
        else:
            cd_curve.append(1.0)

    fig = plt.figure(figsize=(14, 5))
    ax_geom = fig.add_subplot(1, 3, 1)
    ax_drift = fig.add_subplot(1, 3, 2)
    ax_cd = fig.add_subplot(1, 3, 3)

    # Layout sketch (top view, looking down).
    for idx, (b, n) in enumerate(geometry):
        ax_geom.plot(b, n, "o", ms=18, color=f"C{idx}")
        ax_geom.annotate(
            f"cable {idx}\n(y/d={idx*spacing/d:.0f})",
            (b, n),
            xytext=(8, 8),
            textcoords="offset points",
        )
    ax_geom.annotate(
        "wind",
        xy=(0.10, 0.10),
        xycoords="axes fraction",
        xytext=(0.40, 0.10),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        color="red",
        fontsize=12,
    )
    ax_geom.set_xlabel("binormal axis [m]")
    ax_geom.set_ylabel("normal axis [m]")
    ax_geom.set_title(f"Bundle layout (in-line, spacing {spacing*100:.0f} cm)")
    ax_geom.set_aspect("equal")
    ax_geom.set_ylim(-0.05, 0.10)
    ax_geom.grid(True, alpha=0.3)

    for i in range(3):
        ax_drift.plot(t_axis, ub_mid[i] * 100.0, label=f"cable {i}", color=f"C{i}")
    ax_drift.set_xlabel("time [s]")
    ax_drift.set_ylabel("mid-span binormal displacement [cm]")
    ax_drift.set_title(f"Downwind drift at {wind_speed:.0f} m/s")
    ax_drift.legend()
    ax_drift.grid(True, alpha=0.3)

    ax_cd.bar(range(3), cd_curve, color=[f"C{i}" for i in range(3)])
    for i, v in enumerate(cd_curve):
        ax_cd.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom")
    ax_cd.set_xticks(range(3))
    ax_cd.set_xticklabels([f"cable {i}" for i in range(3)])
    ax_cd.set_ylabel("effective cd at mid-span")
    ax_cd.set_title("Wake drag deficit (Wu 2018)")
    ax_cd.set_ylim(0.0, 1.3)
    ax_cd.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "In-line triple bundle exposed to a 10 m/s steady wind - "
        "Figure 11 of the note (simplified)"
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
