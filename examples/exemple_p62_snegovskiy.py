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
        tension=1.91e04,
        h=0.0,
    )



#########
# EXEMPLE P62 SNEGOVSKIY 3 SUB-SPANS
#########

def run() -> None:
    cable0 = _cable_0()
    cable1 = _cable_1()

    k_n = 1e6
    c_n = 200.0
    k_b = 1e6
    c_b = 200.0
    is_2d = True
    delta_n = 0.0
    delta_b = 0.0
    j_b = 200.0
    j_t = 200.0
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
    
    my_wind = ConstantWind(speed=10.0)

    force_callback = aero_force_bundle(
    bundle_geometry = geometry, 
    diameter=0.0254, 
    wind=my_wind,         
    aero_model="price",     
    air_density=1.204
)
    s_adapted = np.concatenate(([0.0], np.cumsum(ds)))

    # =======================================================
    # 1. PHASE DE RELAXATION
    # =======================================================
    params_relax = simulation.Parameters(
        ns=ns, t0=0.0, tf=20.0, 
        dt=0.01, dr=1.0,  # dr=1.0 pour ne pas saturer la mémoire (1 point par sec suffit)
        los=s_adapted.tolist(), # <-- CLÉ N°1 : on sauvegarde TOUTE la vraie grille !
        pp=False
    )
    
    res_relax, _ = bundle.solve(
        is_uniform, spacer_data, ds, ns, params_relax, 
        force=force_callback, zt=2.0, initial_conditions=None
    )
    



    # #########SYM
    # Extraction pure : les tableaux font déjà 323 points, plus besoin de np.pad !
    un_eq = [res_relax.data[f"un_{i}"].values[-1, :] for i in range(2)]
    ub_eq = [res_relax.data[f"ub_{i}"].values[-1, :] for i in range(2)]
    ic_eq = InitialConditions(un=un_eq, ub=ub_eq)

    # # =======================================================
    # # 2. VRAIE SIMULATION WIO
    # # =======================================================
    params_wio = simulation.Parameters(
        ns=ns, t0=20.0, tf=300.0, 
        dt=0.002,  # <-- CLÉ N°2 : on réduit dt pour stabiliser les micro-segments du maillage
        dr=0.02,   # On augmente un peu l'intervalle de sauvegarde pour compenser dt
        los=s_adapted[1:-1].tolist(), # Ici on peut reprendre votre los tronqué classique
        pp=True
    )

    res, _ = bundle.solve(
        is_uniform, spacer_data, ds, ns, params_wio, 
        force=force_callback, zt=0.0, initial_conditions=ic_eq
    )



    # un_eq = [res_relax.data[f"un_{i}"].values[-1, :].copy() for i in range(2)]
    # ub_eq = [res_relax.data[f"ub_{i}"].values[-1, :].copy() for i in range(2)]
    
    # # 2. Ajout de micro-turbulences naturelles (Bruit blanc sur les vitesses)
    # np.random.seed(42) # Fixe l'aléatoire pour que vos tests soient comparables
    
    # # On génère un bruit d'un dixième de millimètre par seconde
    # bruit_vn = np.random.normal(0, 1e-4, ns)
    # bruit_vb = np.random.normal(0, 1e-4, ns)
    
    # # On applique ce bruit uniquement au câble 1 (le câble sous le vent, moteur du WIO)
    # vn_eq = [np.zeros(ns), bruit_vn]
    # vb_eq = [np.zeros(ns), bruit_vb]
    
    # # 3. Injection dans les conditions initiales
    # ic_eq = InitialConditions(un=un_eq, ub=ub_eq, vn=vn_eq, vb=vb_eq)

    # # =======================================================
    # # VRAIE SIMULATION WIO
    # # =======================================================
    # # (Assurez-vous d'avoir tf=300 ou plus, car la transition peut prendre du temps)
    # res, _ = bundle.solve(
    #     is_uniform, spacer_data, ds, ns, params_wio, 
    #     force=force_callback, zt=0.0, initial_conditions=ic_eq
    # )



    t_axis = np.asarray(res.lot())
    
    # Récupération du déplacement Y (binormal) pour le câble sous le vent (index 1)
    # ou un selon ce que vous préférez tracer
    y_leeward = res.data["ub_1"].values  # shape (nt, n_los)

    # Création des indices des sous-portées (milieux)
    los_arr = np.array(s_adapted[1:-1].tolist())
    subspan_targets = [1/6.0, 3/6.0, 5/6.0]  # milieux des 3 sous-portées de 80m
    subspans = [np.abs(los_arr - target).argmin() for target in subspan_targets]

    # Tracé direct style "autre code"
    fig1 = plt.figure(figsize=[14, 8])
    ax = fig1.add_subplot(111)
    fig1.suptitle("Y-component motions in the leeward subconductors, middle of subspans 1 and 3", fontsize=12)
    
    colors = ['r', 'b', 'k']
    for i in range(len(subspans)):
        label = f"subspan {i+1}"
        node_idx = subspans[i]
        y_signal = y_leeward[:, node_idx]
        ax.plot(t_axis, y_signal, "-", linewidth=1.5, color=colors[i], label=label)
        
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Y displacement [m]")
    ax.grid(True)
    ax.legend()
    fig1.tight_layout()
    
    # --- SAUVEGARDE DE LA FIGURE ---
    # Vous pouvez changer l'extension (.png, .pdf, .svg, etc.) et le dossier de destination
    fig1.savefig("mouvements_sous_portees_asym.png", dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    run()



