import matplotlib.pyplot as plt
import numpy as np

from slenderpy import simulation
from slenderpy.cable.bundle.bundle import Bundle, InitialConditions
from slenderpy.cable.bundle.cable import Cable
from slenderpy.cable.bundle.spacer import twin_spacer, Spacer, SpacerEdge
import slenderpy.force as force


def _aster570():
    return Cable(
        mass=1.57,
        diameter=0.03105,
        EA=3.76e07,
        length=40.0,
        tension=2.7795e04,
        h=2.09,
    )



#########
# EXEMPLE DERVAUX
#########

def run() -> None:
    cable = _aster570()
    bundle = Bundle(
        cables=[cable, cable],
        spacers=[twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
                             is_2d = True, delta_n=0.0, delta_b=0.0, j_b=200, j_t=200, width=0.0325, l0 = 0.4)],
    )

    ns = 81
    is_uniform = False
    s = np.linspace(0.0, 1.0, ns)
    ds = np.diff(s)

    L = bundle.cables[0].length
    m = bundle.cables[0].mass
    g = bundle.cables[0].g
    tAd = float(np.sqrt(L / g))
    uAd = L / tAd

    spacer_data, ds, ns = bundle._prepare_spacers(ds, L, m, g, tAd, is_uniform)
    
    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()

    params = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=35.0,
        dt=2.0e-3,
        dr=2.0e-2,
        los=los,
        pp=False,
    )

    s_centre = 0.30
    sigma = 0.05
    bump = 0.01 * np.exp(-((np.concatenate((np.array([0.0]), np.cumsum(ds))) - s_centre) ** 2) / (2.0 * sigma**2))
    bump[0] = 0.0
    bump[-1] = 0.0
    ic = InitialConditions(un=[bump, np.zeros(ns)])

    res, _ = bundle.solve(is_uniform, spacer_data, ds, ns, params, force=None, zt=0.0, initial_conditions=ic)

    t_axis = np.asarray(res.lot())
    un0 = res.data["un_0"].values  
    un1 = res.data["un_1"].values

    rms0 = np.sqrt(np.mean(un0**2, axis=1))
    rms1 = np.sqrt(np.mean(un1**2, axis=1))

    fig, (ax_map, ax_rms) = plt.subplots(1, 2, figsize=(12, 5))

    s_phys = np.array(los) * cable.length
    vmax = max(np.max(np.abs(un0)), np.max(np.abs(un1)))
    ax_map.imshow(
        np.concatenate([un0, un1], axis=1),
        aspect="auto",
        extent=(s_phys[0], 2.0 * s_phys[-1] - s_phys[0], t_axis[-1], t_axis[0]),
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax_map.axvline(s_phys[-1], color="k", lw=1.2)
    ax_map.set_xlabel("s [m]   (cable 0  |  cable 1)")
    ax_map.set_ylabel("time [s]")
    ax_map.set_title("normal displacement, cable 0 (left) | cable 1 (right)")

    ax_rms.plot(t_axis, rms0 * 1e3, label="cable 0 (excited)", color="C0")
    ax_rms.plot(t_axis, rms1 * 1e3, label="cable 1 (passive)", color="C1")
    ax_rms.set_xlabel("time [s]")
    ax_rms.set_ylabel("RMS normal displacement [mm]")
    ax_rms.set_title("Energy transfer through the central spacer")
    ax_rms.legend()
    ax_rms.grid(True, alpha=0.3)

    fig.suptitle(
        "Twin bundle, mid-span spacer, Gaussian IC on cable 0 "
    )
    fig.tight_layout()
    plt.show()


# if __name__ == "__main__":
#     run()

