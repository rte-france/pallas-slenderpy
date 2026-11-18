"""Bundle solver: N sub-conductors coupled by spacers.

The bundle is integrated in time with a Newmark beta = 1/4 scheme,
identical to the single-cable solver in
:mod:`slenderpy.cable.dynamic.cable`, but the implicit operator is
assembled monolithically across all sub-conductors. Spacer reactions
enter the implicit block matrix so the scheme is unconditionally stable
even for stiff spacers (k_perp >> 1e6 N/m).

The user-supplied external force callback is treated explicitly in state
(evaluated at t and t+dt with the current state, then averaged). Spacer
reactions are linear and are handled implicitly through the block matrix.

Finite-difference operators come from :mod:`slenderpy.fdm_utils` and the
small non-dimensionalisation logic is implemented inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from slenderpy import _progress_bar as _spb
from slenderpy import fd_utils as _fdm
from slenderpy import simulation
from slenderpy.cable.bundle.cable import Cable
from slenderpy.cable.bundle.spacer import Spacer





def plot_mesh_adaptation(ds_old: np.ndarray, ds_new: np.ndarray, x_center: float, w_nd: float, case_name: str):
    """
    Mesh verification: spacer geometry plot
    """
    nodes_old = np.concatenate(([0.0], np.cumsum(ds_old)))
    nodes_new = np.concatenate(([0.0], np.cumsum(ds_new)))
    
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Nodes
    ax.plot(nodes_old, np.ones_like(nodes_old), 'bo-', label='Maillage initial', alpha=0.5, clip_on=False)
    ax.plot(nodes_new, np.zeros_like(nodes_new), 'ro-', label='Maillage modifié', alpha=0.8, clip_on=False)
    
    x_left = x_center - w_nd / 2.0
    x_right = x_center + w_nd / 2.0
    
    # Spacer
    rect_old = patches.Rectangle((x_left, 0.8), w_nd, 0.4, linewidth=1, edgecolor='g', facecolor='g', alpha=0.2, label='Entretoise (position cible)')
    rect_new = patches.Rectangle((x_left, -0.2), w_nd, 0.4, linewidth=1, edgecolor='g', facecolor='g', alpha=0.2)
    ax.add_patch(rect_old)
    ax.add_patch(rect_new)
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Nouveau ds\n(Adapté)', 'Ancien ds\n(Uniforme)'])
    ax.set_xlabel("Position adimensionnée 's' le long du câble")
    ax.set_title(f"Vérification du maillage : {case_name}")

    # Neighbour nodes
    left_nodes_old = nodes_old[nodes_old < x_left]
    n_left_old = left_nodes_old[-1] if len(left_nodes_old) > 0 else 0.0
    
    left_nodes_new = nodes_new[nodes_new < x_left]
    n_left_new = left_nodes_new[-1] if len(left_nodes_new) > 0 else 0.0
 
    right_nodes_old = nodes_old[nodes_old > x_right]
    n_right_old = right_nodes_old[0] if len(right_nodes_old) > 0 else 1.0
    
    right_nodes_new = nodes_new[nodes_new > x_right]
    n_right_new = right_nodes_new[0] if len(right_nodes_new) > 0 else 1.0
    

    x_min_plot = min(n_left_old, n_left_new)
    x_max_plot = max(n_right_old, n_right_new)
    

    span = x_max_plot - x_min_plot
    if span <= 0: 
        span = w_nd
        
    ax.set_xlim(max(0.0, x_min_plot - 0.05 * span), min(1.0, x_max_plot + 0.05 * span))
    ax.set_ylim(-0.5, 1.5)
    
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()





_VARS = ("ut", "un", "ub", "ef")


@dataclass
class InitialConditions:
    """Per-sub-conductor initial state for :meth:`Bundle.solve`.

    Each field is a list of arrays, one per sub-conductor in the same
    order as the ``cables`` argument of :class:`Bundle`. A field set to
    ``None`` defaults the whole component to zero; an individual entry
    set to ``None`` inside a list defaults that single sub-conductor to
    zero. Arrays are dimensional (m for displacements, m/s for
    velocities) and must have length ``parameters.ns``.
    """

    un: Optional[List[Optional[np.ndarray]]] = None
    ub: Optional[List[Optional[np.ndarray]]] = None
    vn: Optional[List[Optional[np.ndarray]]] = None
    vb: Optional[List[Optional[np.ndarray]]] = None


class Bundle:
    """N sub-conductors integrated jointly under spacer coupling.

    Parameters
    ----------
    cables : list of Cable
        Sub-conductors (at least one). All sub-conductors share the same
        span length ``L``; this is enforced at construction. For the first
        iteration all sub-conductors are also required to share the same
        linear mass, so that Newton's third law on spacer reactions yields
        symmetric non-dimensional coefficients. Lifting this restriction
        is straightforward; see the assembly code.
    spacers : list of Spacer
        Spacer descriptors. May be empty (degenerate bundle = independent
        cables).
    is_2d: boolean
        Inicates if the bundle spacers are 2D or not.
    """

    def __init__(self, cables: List[Cable], spacers: List[Spacer]) -> None:
        if len(cables) < 1:
            raise ValueError("Bundle needs at least one cable")
        L0 = cables[0].length
        m0 = cables[0].mass
        for c in cables[1:]:
            if not np.isclose(c.length, L0):
                raise ValueError("all cables in a bundle must share the same length")
            if not np.isclose(c.mass, m0):
                raise ValueError(
                    "all cables in a bundle must share the same linear mass "
                    "(first iteration restriction; lift later if needed)"
                )
        for sp_ in spacers:
            for e in sp_.edges:
                if e.i >= len(cables) or e.j >= len(cables):
                    raise ValueError(
                        f"SpacerEdge references cable index out of range "
                        f"(have {len(cables)} cables, edge has i={e.i}, j={e.j})"
                    )
        self.cables = list(cables)
        self.spacers = list(spacers)
        self.is_2d = False

        if spacers:  
            spacer_2d_flags = [sp.is_2d for sp in spacers]
            if len(set(spacer_2d_flags)) > 1:
                raise ValueError("All Spacers within a Bundle must have the same dimension (= same is_2d value)")
            self.is_2d = spacer_2d_flags[0]



    @property
    def n_cables(self) -> int:
        return len(self.cables)

    def solve(
        self, is_uniform, spacer_data, ds, ns,
        parameters: simulation.Parameters,
        force: Optional[Callable] = None,
        zt: float = 0.0,
        initial_conditions: Optional[InitialConditions] = None
    ) -> simulation.Results:
        """Run the dynamic simulation.

        Parameters
        ----------
        parameters : slenderpy.simulation.Parameters
            Simulation parameters.
        force : callable, optional
            Bundle force callback with signature
            ``force(s, t, un_list, ub_list, vn_list, vb_list) -> (fn_list, fb_list)``
            where each ``*_list`` is a list of length ``n_cables`` of
            length-``ns`` arrays. Returned ``fn_list`` and ``fb_list``
            contain force per unit length (N/m) for each sub-conductor.
            Default: zero force.
        zt : float, optional
            Modal viscous damping ratio, applied uniformly to all
            sub-conductors. Default 0.
        initial_conditions : InitialConditions, optional
            Per-sub-conductor initial state. Default: zero everywhere.

        Returns
        -------
        slenderpy.simulation.Results
            Results object with per-sub-conductor displacement and axial
            force traces named ``ut_i``, ``un_i``, ``ub_i``, ``ef_i``.
        """
        nc = self.n_cables
        L = self.cables[0].length
        m = self.cables[0].mass
        g = self.cables[0].g
        tAd = float(np.sqrt(L / g))
        uAd = L / tAd
        vt2 = np.array([cb.vt2 for cb in self.cables])
        vl2 = np.array([cb.vl2 for cb in self.cables])


        n = ns - 2
        # FD operators (shared across all cables since the grid is shared).
        # C : (ns, ns) first-derivative operator, A : (n, n) second-derivative
        # operator with Dirichlet zero boundary.
        C = _fdm.d1M(ds)
        A = _fdm.d2M(ds)
        A_csr = A.tocsr()
        Adiag0 = A.diagonal(k=0)
        Adiag1 = A.diagonal(k=1)
        Adiagm1 = A.diagonal(k=-1)

        # Non-dim time stepping. Match legacy convention: dt is divided by
        # tAd, ht = dt/2, tau = -ht**2 (the implicit operator coefficient).
        t = parameters.t0 / tAd
        tf = parameters.tf / tAd
        dt = (tf - t) / parameters.nt
        ht = 0.5 * dt
        tau = -(ht**2)

        # Damping per cable (z_i depends on vt2_i which differs per cable
        # if tensions differ).
        z = -2.0 * np.pi * np.sqrt(vt2) * zt * ht

        un = np.zeros((nc, ns))
        ub = np.zeros((nc, ns))
        vn = np.zeros((nc, ns))
        vb = np.zeros((nc, ns))
        if initial_conditions is not None:
            _fill_initial(un, initial_conditions.un, scale=1.0 / L)
            _fill_initial(ub, initial_conditions.ub, scale=1.0 / L)
            _fill_initial(vn, initial_conditions.vn, scale=1.0 / uAd)
            _fill_initial(vb, initial_conditions.vb, scale=1.0 / uAd)
        s = np.concatenate((np.array([0.0]), np.cumsum(ds)))

        

        # Output container. Per-cable variable names so simulation.multiplot
        # superimposes sub-conductors at each output location.
        lov: List[str] = []
        for v in _VARS:
            for i in range(nc):
                lov.append(f"{v}_{i}")
        lov_dims = tuple([2] * len(lov))
        res = simulation.Results(
            lot=parameters.time_vector_output().tolist(),
            lov=lov,
            lov_dims=lov_dims,
            los=parameters.los,
        )
        self._record(res, 0, s, ds, C, vt2, un, ub)

        if force is None:
            force = _zero_bundle_force

        res.start_timer()
        pb = _spb.generate(parameters.pp, parameters.nt, desc=__name__)
        for k in range(parameters.nt):
            # Per-cable strain integral and effective wave-speed b_i.
            e_vals = np.empty(nc)
            b = np.empty(nc)
            for i in range(nc):
                h_i = -un[i, :] / vt2[i] + 0.5 * (
                    (C @ un[i, :]) ** 2 + (C @ ub[i, :]) ** 2
                )
                e_vals[i] = 0.5 * np.sum((h_i[:-1] + h_i[1:]) * ds)
                b[i] = vt2[i] + vl2[i] * e_vals[i]

            # User force, evaluated at current state at both t and t+dt.
            fn1, fn2, fb1, fb2 = _eval_force(
                force, s, t, dt, un, ub, vn, vb, tAd, L, uAd, m, g
            )

            Rvn = np.empty((nc, n))
            Rvb = np.empty((nc, n))
            for i in range(nc):
                Rvn[i, :] = (
                    (dt * b[i]) * (A_csr @ (un[i, 1:-1] + 0.5 * ht * vn[i, 1:-1]))
                    + (1.0 + z[i]) * vn[i, 1:-1]
                    + ht * (fn1[i, 1:-1] + fn2[i, 1:-1])
                    # Gravity-coupled restoring term (Lee's formulation,
                    # legacy cable.py:638-642); absent from the binormal
                    # direction by symmetry of the catenary.
                    + dt * vl2[i] / vt2[i] * e_vals[i]
                )
                Rvb[i, :] = (
                    (dt * b[i]) * (A_csr @ (ub[i, 1:-1] + 0.5 * ht * vb[i, 1:-1]))
                    + (1.0 + z[i]) * vb[i, 1:-1]
                    + ht * (fb1[i, 1:-1] + fb2[i, 1:-1])
                )

            # Add explicit half of spacer reactions to the RHS, and assemble
            # block matrix with implicit half on the LHS.
            M_block, Rvn, Rvb = _assemble_with_spacers(
                nc,
                n,
                z,
                b,
                tau,
                dt,
                ht,
                Adiag0,
                Adiag1,
                Adiagm1,
                A_csr,
                spacer_data,
                un,
                ub,
                vn,
                vb,
                Rvn,
                Rvb,
            )

            # Solve for v^{n+1} jointly across all cables.
            # Stack as [vn_0, vb_0, vn_1, vb_1, ...] of length 2*nc*n.
            rhs = np.empty(2 * nc * n)
            for i in range(nc):
                rhs[(2 * i) * n : (2 * i + 1) * n] = Rvn[i, :]
                rhs[(2 * i + 1) * n : (2 * i + 2) * n] = Rvb[i, :]
            sol = scipy.sparse.linalg.spsolve(M_block, rhs)

            # Newmark position update and state replacement.
            for i in range(nc):
                vnb_i = sol[(2 * i) * n : (2 * i + 1) * n]
                vbb_i = sol[(2 * i + 1) * n : (2 * i + 2) * n]
                un[i, 1:-1] += ht * (vn[i, 1:-1] + vnb_i)
                ub[i, 1:-1] += ht * (vb[i, 1:-1] + vbb_i)
                vn[i, 1:-1] = vnb_i
                vb[i, 1:-1] = vbb_i

            t += dt
   

            if (k + 1) % parameters.rr == 0:
                self._record(res, (k // parameters.rr) + 1, s, ds, C, vt2, un, ub)
                pb.update(parameters.rr)

        pb.close()
        res.stop_timer()

        # Re-dimensionalise displacements and axial force. Time is already
        # stored in seconds (parameters.time_vector_output() returns
        # dimensional times in the user's units), so it does not need
        # rescaling here.
        for i in range(nc):
            for v in ("ut", "un", "ub"):
                res.data[f"{v}_{i}"] *= L
            res.data[f"ef_{i}"] *= self.cables[i].EA

        return res, ds

    def _prepare_spacers(
        self, ds: np.ndarray, L: float, m: float, g: float, tAd: float, is_uniform: bool
    ) -> List[dict]:
        """Resolve spacer locations to grid indices and compute non-dim
        coefficients per edge. Returned list mirrors ``self.spacers``."""
        ns = len(ds) + 1
        data = []

        # 1D SPACER
        if not self.is_2d:
            for spacer in self.spacers:
                # Map continuous s_k in ]0, 1[ to the nearest interior node
                # index (0-based, in [0, n-1] where n = ns - 2). Reject ends.
                full_idx = int(round(spacer.s * (ns - 1)))
                interior_idx = full_idx - 1
                if interior_idx < 0 or interior_idx >= ns - 2:
                    raise ValueError(
                        f"spacer at s={spacer.s} maps to a boundary node; "
                        f"choose s strictly inside ]0, 1["
                    )
            
                # Local cell width: use the larger of the two adjacent cells
                # so the localised point load is on the order of F / (ds * L).
                ds_local = 0.5 * (ds[full_idx - 1] + ds[full_idx])
                ds_dim = ds_local * L
                interior_idx_left = interior_idx
                interior_idx_right = interior_idx
                w_dim = spacer.width  
                w_nd = w_dim / L

                edges_nd = []
                for e in spacer.edges:
                    K_nd_n = e.k_n / (4 * m * g * ds_dim) * L
                    C_nd_n = e.c_n / (4 * m * g * ds_dim * tAd) * L
                    K_nd_b = e.k_b / (4 * m * g * ds_dim) * L
                    C_nd_b = e.c_b / (4 * m * g * ds_dim * tAd) * L
                    J_b_nd = (e.j_b*ds_local/(e.l0**3))*(0)
                    J_t_nd = (e.j_t*ds_local/(e.l0**3))*(0)
                    
                    edges_nd.append(
                        {
                            "i": e.i,
                            "j": e.j,
                            "K_nd_n": K_nd_n,
                            "C_nd_n": C_nd_n,
                            "K_nd_b": K_nd_b,
                            "C_nd_b": C_nd_b,
                            "delta_n": e.delta_n / L,
                            "delta_b": e.delta_b / L,
                            "J_b_nd": J_b_nd,
                            "J_t_nd": J_t_nd,
                            "width_nd": w_nd,
                        }
                    )
                
                data.append({
                    "interior_idx_left": interior_idx_left,
                    "interior_idx_right": interior_idx_right,
                    "edges": edges_nd
                })
                
            new_ds = ds


        # 2D SPACER
        else:
            for spacer in self.spacers:

                if is_uniform:
                    # UNIFORM MESH
                    w_nd = ds[0]
                    
                    full_idx_left = int(spacer.s * (ns - 1))
                    full_idx_right = full_idx_left + 1
                    
                    full_idx = full_idx_left 
                    ds_local = w_nd 
                    new_ds = ds
                    
                else:
                    # IRREGULAR MESH 
                    w_dim = spacer.width  
                    w_nd = w_dim/L
                    ds_local = w_nd
                
                    # Spacer edges
                    x_center = spacer.s 
                    x_left = x_center - w_nd / 2.0
                    x_right = x_center + w_nd / 2.0

                    # Closest mesh nodes of the spacer edges
                    coord_nodes = np.cumsum(ds)
                    idx_left = (np.abs(coord_nodes - x_left)).argmin()
                    idx_right = (np.abs(coord_nodes - x_right)).argmin()

                    ds_old_copy = ds.copy()

                    if idx_left != idx_right:
                        ds[idx_left] += x_left - coord_nodes[idx_left]
                        ds[idx_left + 1] = w_nd
                        ds[idx_left + 2] = coord_nodes[idx_right + 1] - x_right
                        new_ds = ds

                        # Deleting the nodes inside the spacer
                        if (idx_left + 1) != idx_right:
                            n_spacer = idx_right - (idx_left + 1)
                            ns -= n_spacer
                            new_ds = ds[:ns-1]
                            ds = new_ds

                        # plot_mesh_adaptation(ds_old_copy, new_ds, x_center, w_nd, "Réutilisation de noeuds (idx_left != idx_right)")

                    else: # Adding a node
                        new_ds = np.zeros(ns)
                        ns += 1
                        new_ds[:idx_left] = ds[:idx_left]
                        new_ds[idx_left] = ds[idx_left] + x_left - coord_nodes[idx_left]
                        new_ds[idx_left + 1] = w_nd
                        new_ds[idx_left + 2] = coord_nodes[idx_right + 1] - x_right
                        new_ds[idx_left + 3:] = ds[idx_left+2:]
                    
                        # plot_mesh_adaptation(ds_old_copy, new_ds, x_center, w_nd, "Ajout d'un noeud (idx_left == idx_right)")

                        ds = new_ds

                    full_idx_left = idx_left + 1
                    full_idx_right = idx_left + 2
                    
                    print(f"Adaptation maillage : ")
                    print(f"Bords physiques visés : {x_left}m -> {x_right}m")
                    print(f"Indices du maillage   : {full_idx_left} -> {full_idx_right}")
                    print(f"Nouvelle longueur câble : {np.sum(ds) * L:.3f}m (Vérification)")

                interior_idx_left = full_idx_left - 1
                interior_idx_right = full_idx_right - 1
                print('interior_idx_left = ', interior_idx_left, 'interior_idx_right = ', interior_idx_right)

                if interior_idx_left < 0 or interior_idx_right >= ns - 2:
                    raise ValueError(
                        f"2D spacer at s={spacer.s} overlaps boundary nodes : move it further inside the span."
                    )
                
                ds_dim = w_nd * L
                
                edges_nd = []
                for e in spacer.edges:
                    K_nd_n = e.k_n / (4 * m * g * ds_dim) * L
                    C_nd_n = e.c_n / (4 * m * g * ds_dim * tAd) * L
                    K_nd_b = e.k_b / (4 * m * g * ds_dim) * L
                    C_nd_b = e.c_b / (4 * m * g * ds_dim * tAd) * L
                    J_b_nd = (e.j_b * ds_dim / (e.l0**3))
                    J_t_nd = (e.j_t * ds_dim / (e.l0**3)) 
                    
                    edges_nd.append(
                        {
                            "i": e.i,
                            "j": e.j,
                            "K_nd_n": K_nd_n,
                            "C_nd_n": C_nd_n,
                            "K_nd_b": K_nd_b,
                            "C_nd_b": C_nd_b,
                            "delta_n": e.delta_n / L,
                            "delta_b": e.delta_b / L,
                            "J_b_nd": J_b_nd,
                            "J_t_nd": J_t_nd,
                            "width_nd": w_nd,
                        }
                    )
            
                data.append({
                    "interior_idx_left": interior_idx_left,
                    "interior_idx_right": interior_idx_right,
                    "edges": edges_nd
                })
            
        return data, new_ds, ns

    @staticmethod
    def _record(
        res: simulation.Results,
        k: int,
        s: np.ndarray,
        ds: np.ndarray,
        C: scipy.sparse.dia_matrix,
        vt2: np.ndarray,
        un: np.ndarray,
        ub: np.ndarray,
    ) -> None:
        nc = un.shape[0]
        names = []
        values = []
        for i in range(nc):
            ut_i, ef_i = _utef(un[i, :], ub[i, :], C, s, ds, vt2[i])
            names.extend([f"ut_{i}", f"un_{i}", f"ub_{i}", f"ef_{i}"])
            values.extend([ut_i, un[i, :], ub[i, :], ef_i])
        res.update(k, s, names, values)


def _fill_initial(
    target: np.ndarray,
    component: Optional[List[Optional[np.ndarray]]],
    scale: float,
) -> None:
    """Copy non-None per-sub-conductor initial values into ``target``,
    multiplied by ``scale`` (the non-dimensionalisation factor). A None
    component, or a None entry within the component list, leaves the
    corresponding sub-conductor at its pre-filled value (zero).
    """
    if component is None:
        return
    for i, arr in enumerate(component):
        if arr is not None:
            target[i, :] = np.asarray(arr, dtype=float) * scale


def _zero_bundle_force(
    s: np.ndarray,
    t: float,
    un_list,
    ub_list,
    vn_list,
    vb_list,
) -> Tuple[list, list]:
    nc = len(un_list)
    fn = [np.zeros_like(s) for _ in range(nc)]
    fb = [np.zeros_like(s) for _ in range(nc)]
    return fn, fb


def _eval_force(
    force: Callable,
    s: np.ndarray,
    t: float,
    dt: float,
    un: np.ndarray,
    ub: np.ndarray,
    vn: np.ndarray,
    vb: np.ndarray,
    tAd: float,
    L: float,
    uAd: float,
    m: float,
    g: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Both calls pass the current state; only the time argument differs.

    This matches the legacy single-cable semantics and is what makes
    the user force explicit while the cable operator stays implicit.
    """
    nc = un.shape[0]
    un_list = [un[i, :] * L for i in range(nc)]
    ub_list = [ub[i, :] * L for i in range(nc)]
    vn_list = [vn[i, :] * uAd for i in range(nc)]
    vb_list = [vb[i, :] * uAd for i in range(nc)]
    t_dim = t * tAd
    tp_dim = (t + dt) * tAd
    fn1, fb1 = force(s, t_dim, un_list, ub_list, vn_list, vb_list)
    fn2, fb2 = force(s, tp_dim, un_list, ub_list, vn_list, vb_list)
    scale = 1.0 / (m * g)
    fn1_arr = np.stack([np.asarray(x, dtype=float) * scale for x in fn1])
    fn2_arr = np.stack([np.asarray(x, dtype=float) * scale for x in fn2])
    fb1_arr = np.stack([np.asarray(x, dtype=float) * scale for x in fb1])
    fb2_arr = np.stack([np.asarray(x, dtype=float) * scale for x in fb2])
    return fn1_arr, fn2_arr, fb1_arr, fb2_arr


def _utef(
    un: np.ndarray,
    ub: np.ndarray,
    C: scipy.sparse.dia_matrix,
    s: np.ndarray,
    ds: np.ndarray,
    vt2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tangential offset and axial force in Lee's formulation.

    Faithful reimplementation of _cable_utils.utef.
    """
    h = -1.0 / vt2 * un + 0.5 * ((C @ un) ** 2 + (C @ ub) ** 2)
    H = 0.5 * (h[:-1] + h[1:]) * ds
    ut = np.sum(H) * s - np.cumsum(np.concatenate(([0.0], H)))
    e = (C @ ut) + 0.5 * ((C @ ut) ** 2 + (C @ un) ** 2 + (C @ ub) ** 2)
    ef = np.log(np.sqrt(1.0 + 2.0 * e))
    return ut, ef


def _assemble_with_spacers(
    nc: int,
    n: int,
    z: np.ndarray,
    b: np.ndarray,
    tau: float,
    dt: float,
    ht: float,
    Adiag0: np.ndarray,
    Adiag1: np.ndarray,
    Adiagm1: np.ndarray,
    A_csr: scipy.sparse.csr_matrix,
    spacer_data: List[dict],
    un: np.ndarray,
    ub: np.ndarray,
    vn: np.ndarray,
    vb: np.ndarray,
    Rvn: np.ndarray,
    Rvb: np.ndarray,
) -> Tuple[scipy.sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Assemble the global implicit operator including spacer coupling,
    and inject the explicit half of the spacer reactions into the RHS.

    State ordering in the global system: for sub-conductor i, the block
    [vn_i] occupies rows/cols [(2 i) n, (2 i + 1) n), and [vb_i] occupies
    rows/cols [(2 i + 1) n, (2 i + 2) n).
    """
    # Per-cable diagonal blocks: banded (3 diagonals) for vn and for vb.
    blocks = []
    for i in range(nc):
        d0 = (1.0 - z[i]) + tau * b[i] * Adiag0
        d1 = tau * b[i] * Adiag1
        dm1 = tau * b[i] * Adiagm1
        M_i = scipy.sparse.diags(
            [dm1, d0, d1], offsets=[-1, 0, 1], shape=(n, n), format="lil"
        )
        blocks.append(M_i)  # for vn_i
        blocks.append(M_i.copy())  # for vb_i

    # Spacer contributions. For each edge of each spacer, at interior
    # node n_k, add implicit coupling on velocity (both n and b
    # components, since the spacer is isotropic in the perpendicular
    # plane) and inject the explicit-state portion of the reaction into
    # the RHS.

    
    for sp_d in spacer_data:
        idx_L = sp_d["interior_idx_left"]
        idx_R = sp_d.get("interior_idx_right")
        if idx_R is None:
            idx_R = idx_L
  
        full_L = idx_L + 1
        full_R = idx_R + 1

        for e in sp_d["edges"]:
            i = e["i"]
            j = e["j"]
            K_n = e["K_nd_n"]
            Cd_n = e["C_nd_n"]
            K_b = e["K_nd_b"]
            Cd_b = e["C_nd_b"]
            Jb = e["J_b_nd"]
            Jt = e["J_t_nd"]
            dn = e["delta_n"]
            db = e["delta_b"]

            coup_K_n = dt * 0.5 * (K_n * ht + Cd_n)
            coup_K_b = dt * 0.5 * (K_b * ht + Cd_b)
            coup_J_n = dt * 0.5 * (Jb * ht)
            coup_J_b = dt * 0.5 * (Jt * ht)

            M_i_vn = blocks[2 * i]
            M_j_vn = blocks[2 * j]
            M_i_vb = blocks[2 * i + 1]
            M_j_vb = blocks[2 * j + 1]

            # Composante normale (vn)
            M_i_vn[idx_L, idx_L] += (coup_K_n + coup_J_n)
            M_i_vn[idx_L, idx_R] += (coup_K_n - coup_J_n)
            M_i_vn[idx_R, idx_L] += (coup_K_n - coup_J_n)
            M_i_vn[idx_R, idx_R] += (coup_K_n + coup_J_n)
            
            M_j_vn[idx_L, idx_L] += (coup_K_n + coup_J_n)
            M_j_vn[idx_L, idx_R] += (coup_K_n - coup_J_n)
            M_j_vn[idx_R, idx_L] += (coup_K_n - coup_J_n)
            M_j_vn[idx_R, idx_R] += (coup_K_n + coup_J_n)

            # Composante binormale (vb)
            M_i_vb[idx_L, idx_L] += (coup_K_b + coup_J_b)
            M_i_vb[idx_L, idx_R] += (coup_K_b - coup_J_b)
            M_i_vb[idx_R, idx_L] += (coup_K_b - coup_J_b)
            M_i_vb[idx_R, idx_R] += (coup_K_b + coup_J_b)
            
            M_j_vb[idx_L, idx_L] += (coup_K_b + coup_J_b)
            M_j_vb[idx_L, idx_R] += (coup_K_b - coup_J_b)
            M_j_vb[idx_R, idx_L] += (coup_K_b - coup_J_b)
            M_j_vb[idx_R, idx_R] += (coup_K_b + coup_J_b)

            # Explicit-state portion of the reaction at this node, in n
            # and b components, on cable i (cable j is opposite).
            # The reaction force on cable i (non-dim, per unit length, at
            # node k_idx) is:
            #   f_i_n^expl = -K*(un_i - un_j - dn) - 0.5*(Cd + K*ht)*(vn_i - vn_j)
            #   f_i_b^expl = -K*(ub_i - ub_j - db) - 0.5*(Cd + K*ht)*(vb_i - vb_j)
            # multiplied by dt to enter the v-update equation.
            dun_2d = (un[i, full_L] + un[i, full_R]) - (un[j, full_L] + un[j, full_R]) - 2.0 * dn
            dvn_2d = (vn[i, full_L] + vn[i, full_R]) - (vn[j, full_L] + vn[j, full_R])
            dub_2d = (ub[i, full_L] + ub[i, full_R]) - (ub[j, full_L] + ub[j, full_R]) - 2.0 * db
            dvb_2d = (vb[i, full_L] + vb[i, full_R]) - (vb[j, full_L] + vb[j, full_R])

            f_i_n_trans = -K_n * dun_2d - 0.5 * (Cd_n + K_n * ht) * dvn_2d
            f_i_b_trans = -K_b * dub_2d - 0.5 * (Cd_b + K_b * ht) * dvb_2d

            f_i_n_tors_L = -Jb * ((un[i, full_L] - un[i, full_R]) - (un[j, full_L] - un[j, full_R]))
            f_i_b_tors_L = -Jt * ((ub[i, full_L] - ub[i, full_R]) - (ub[j, full_L] - ub[j, full_R]))

            Rvn[i, idx_L] += dt * (f_i_n_trans + f_i_n_tors_L)
            Rvn[j, idx_L] -= dt * (f_i_n_trans + f_i_n_tors_L) 
            Rvb[i, idx_L] += dt * (f_i_b_trans + f_i_b_tors_L)
            Rvb[j, idx_L] -= dt * (f_i_b_trans + f_i_b_tors_L)

            Rvn[i, idx_R] += dt * (f_i_n_trans - f_i_n_tors_L)
            Rvn[j, idx_R] -= dt * (f_i_n_trans - f_i_n_tors_L)
            Rvb[i, idx_R] += dt * (f_i_b_trans - f_i_b_tors_L)
            Rvb[j, idx_R] -= dt * (f_i_b_trans - f_i_b_tors_L)

    # Build block-diagonal portion first, then inject off-diagonal spacer
    # couplings. Block ordering: [vn_0, vb_0, vn_1, vb_1, ...].
    M_block = scipy.sparse.block_diag(blocks, format="lil")

    for sp_d in spacer_data:
        idx_L = sp_d["interior_idx_left"]
        idx_R = sp_d.get("interior_idx_right")
        if idx_R is None:
            idx_R = idx_L

        for e in sp_d["edges"]:
            i = e["i"]
            j = e["j"]
            K_n = e["K_nd_n"]
            Cd_n = e["C_nd_n"]
            K_b = e["K_nd_b"]
            Cd_b = e["C_nd_b"]
            Jb = e["J_b_nd"]
            Jt = e["J_t_nd"]
            dn = e["delta_n"]
            db = e["delta_b"]

            coup_K_n = dt * 0.5 * (K_n * ht + Cd_n)
            coup_K_b = dt * 0.5 * (K_b * ht + Cd_b)
            coup_J_n = dt * 0.5 * (Jb * ht)
            coup_J_b = dt * 0.5 * (Jt * ht)


            base_i_vn =  (2 * i) * n
            base_j_vn = (2 * j) * n
            base_i_vb = (2 * i + 1) * n
            base_j_vb = (2 * j + 1) * n

            # Composante normale (vn)
            M_block[base_i_vn + idx_L, base_j_vn + idx_L] -= (coup_K_n + coup_J_n)
            M_block[base_i_vn + idx_L, base_j_vn + idx_R] -= (coup_K_n - coup_J_n)
            M_block[base_i_vn + idx_R, base_j_vn + idx_L] -= (coup_K_n - coup_J_n)
            M_block[base_i_vn + idx_R, base_j_vn + idx_R] -= (coup_K_n + coup_J_n)


            M_block[base_j_vn + idx_L, base_i_vn + idx_L] -= (coup_K_n + coup_J_n)
            M_block[base_j_vn + idx_L, base_i_vn + idx_R] -= (coup_K_n - coup_J_n)
            M_block[base_j_vn + idx_R, base_i_vn + idx_L] -= (coup_K_n - coup_J_n)
            M_block[base_j_vn + idx_R, base_i_vn + idx_R] -= (coup_K_n + coup_J_n)

            # Composante binormale (vb)
            M_block[base_i_vb + idx_L, base_j_vb + idx_L] -= (coup_K_b + coup_J_b)
            M_block[base_i_vb + idx_L, base_j_vb + idx_R] -= (coup_K_b - coup_J_b)
            M_block[base_i_vb + idx_R, base_j_vb + idx_L] -= (coup_K_b - coup_J_b)
            M_block[base_i_vb + idx_R, base_j_vb + idx_R] -= (coup_K_b + coup_J_b)

            M_block[base_j_vb + idx_L, base_i_vb + idx_L] -= (coup_K_b + coup_J_b)
            M_block[base_j_vb + idx_L, base_i_vb + idx_R] -= (coup_K_b - coup_J_b)
            M_block[base_j_vb + idx_R, base_i_vb + idx_L] -= (coup_K_b - coup_J_b)
            M_block[base_j_vb + idx_R, base_i_vb + idx_R] -= (coup_K_b + coup_J_b)
        
    return M_block.tocsr(), Rvn, Rvb
