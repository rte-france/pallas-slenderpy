"""Twin bundle: free vibration with energy transfer through a single spacer.

Reproduces the geometric layout of Figure 5 (left) of the Eurobios/RTE note
ESCB-2020-11-12: two parallel ACSR conductors held by a single mid-span
spacer. Cable 0 is released from a Gaussian initial deflection while
cable 1 is at rest; the spacer transfers energy between the two cables.

Run:
    python examples/bundle_figure5_twin.py

Produces a 2-subplot figure:
- left  : space-time map of cable 0 and cable 1 normal displacement
- right : RMS displacement of each cable over time (the spacer transfers
  energy: cable 1's RMS rises while cable 0's relaxes)
"""

import matplotlib.pyplot as plt
import numpy as np

from slenderpy import simulation
from slenderpy.force.wind import ConstantWind
from slenderpy.wake import aero_force_bundle
from slenderpy.cable.bundle.bundle import Bundle, InitialConditions
from slenderpy.cable.bundle.cable import Cable
from slenderpy.cable.bundle.spacer import twin_spacer, Spacer, SpacerEdge


def _cable_0():
    return Cable(
        mass=1.25,
        diameter=2.54e-2,
        EA=2.9e07,
        length=240.0,
        tension=1.9e04,
        h=0.0,
    )

def _cable_1():
    return Cable(
        mass=1.25,
        diameter=2.54e-2,
        EA=2.9e07,
        length=240.0,
        tension=1.9e04,
        h=0.0,
    )



#########
# EXEMPLE P62 SNEGOVSKIY 3 SUB-SPANS
#########

def run() -> None:
    cable0 = _cable_0()
    cable1 = _cable_1()

    k_n = 1e6
    c_n = 0.0
    k_b = 1e6
    c_b = 0.0
    is_2d = True
    delta_n = 0.0
    delta_b = 0.0
    j_b = 0.0
    j_t = 0.0
    width = 0.065
    l0 = 15*2.54e-2


    spacer0 = Spacer(
        s=1./3,
        edges=(SpacerEdge(i=0, j=1, k_n=k_n, c_n=c_n, k_b=k_b, c_b=c_b, 
                  is_2d = is_2d, delta_n = delta_n, delta_b = delta_b, j_b = j_b, j_t = j_t, 
                  width = width, l0 = l0), ), 
                  is_2d = is_2d, width = width)

    spacer1 = Spacer(
        s=2./3,
        edges=(SpacerEdge(i=0, j=1, k_n=k_n, c_n=c_n, k_b=k_b, c_b=c_b, 
                  is_2d = is_2d, delta_n = delta_n, delta_b = delta_b, j_b = j_b, j_t = j_t, 
                  width = width, l0 = l0), ), 
                  is_2d = is_2d, width = width)

    bundle = Bundle(
        cables=[cable0, cable1],
        spacers=[spacer0, spacer1],
    )

    geometry = [(0.0, 0.0), (l0, -2.5*2.54e-2)]

    ns = 321
    is_uniform = False
    s = np.linspace(0.0, 1.0, ns)
    ds = np.diff(s)

    # Per-cable non-dimensionalisation. All cables share L, m, g (enforced
    # at construction), so tAd, uAd are identical
    L = bundle.cables[0].length
    m = bundle.cables[0].mass
    g = bundle.cables[0].g
    tAd = float(np.sqrt(L / g))
    uAd = L / tAd

    # Spacer pre-processing: precompute interior-node index and non-dim
    # coefficients per edge.
    spacer_data, ds, ns = bundle._prepare_spacers(ds, L, m, g, tAd, is_uniform)
    print("len ds = ", len(ds))
    print("ns = ", ns)
    
    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()

    params = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=10.0,
        dt=0.001,
        dr=0.001,
        los=los,
        pp=False,
    )

    my_wind = ConstantWind(speed=10.0)

    force_callback = aero_force_bundle(
    bundle_geometry = geometry, 
    diameter=0.0254, 
    wind=my_wind,         
    aero_model="price",     
    air_density=1.22
)

    s_nd = np.concatenate((np.array([0.0]), np.cumsum(ds)))

    bump = np.where(s_nd <= 1/3, 0.01 * np.sin(3.0 * np.pi * s_nd), 0.0)
    ic = InitialConditions(un=[bump, np.zeros(ns)], ub=None, vn=None, vb=None)
    print("len bump = ", len(bump))

    res, _ = bundle.solve(is_uniform, spacer_data, ds, ns, params, force=None, zt=0.0, initial_conditions=ic)

    t_axis = np.asarray(res.lot())
    
    # Récupération du déplacement Normal (un) pour le câble 0 (celui qui a reçu le bump initial)
    un_excited = res.data["un_0"]
    # Si un_excited est un objet pandas (selon votre version), décommentez la ligne suivante :
    if hasattr(un_excited, "values"): un_excited = un_excited.values

    # Création de l'indice pour la première sous-portée uniquement (x = 40m)
    # Sur une portée de 240m, x=40m correspond à la position relative 40/240 = 1/6
    los_arr = np.array(los)
    target_subspan_1 = 1/6.0
    node_idx = np.abs(los_arr - target_subspan_1).argmin()

    # Tracé direct
    fig1 = plt.figure(figsize=[12, 6])
    ax = fig1.add_subplot(111)
    fig1.suptitle("Analyse modale (Sans vent) : Mesure de la fréquence propre de sous-portée", fontsize=14)
    
    # Extraction du signal au nœud ciblé
    u_signal = un_excited[:, node_idx]
    
    ax.plot(t_axis, u_signal, "-", linewidth=1.5, color='b', label="Milieu sous-portée 1 (x = 40 m)")
        
    ax.set_xlabel("Temps t [s]")
    ax.set_ylabel("Déplacement Normal [m]")
    ax.grid(True)
    ax.legend()
    
    # Optionnel : Zoomer sur les 5 premières secondes pour bien mesurer la période entre deux crêtes
    # ax.set_xlim(0, 5.0) 
    
    fig1.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()



