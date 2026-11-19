import matplotlib.pyplot as plt
import numpy as np

from slenderpy import simulation
from slenderpy.cable.bundle.bundle import Bundle, InitialConditions
from slenderpy.cable.bundle.cable import Cable
from slenderpy.cable.bundle.spacer import twin_spacer, Spacer, SpacerEdge


def _aster570():
    return Cable(
        mass=1.57,
        diameter=0.031,
        EA=3.76e07,
        length=240.0,
        tension=3.7e04,
        h=0.0,
    )

#########
# SPATIAL CONVERGENCE STUDY
#########

# Uniform mesh, 2D spacer: width forced to width = ds
def run_convergence_sim_uni(ns, tf=1.0, dt=5.0e-4):
    """
    Displacement, boundaries included, at t=tf.
    """
    cable = _aster570()
    L = cable.length
    
    # Spacer width
    ds_phys = L / (ns - 1)
    
    bundle = Bundle(
        cables=[cable, cable],
        spacers=[twin_spacer(
            s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
            is_2d=True, delta_n=0.0, delta_b=0.0,
            j_b=200, j_t=200, width=ds_phys, l0=0.40 
        )],
    )

    s = np.linspace(0.0, 1.0, ns)
    ds = np.diff(s)

    # Per-cable non-dimensionalisation. All cables share L, m, g (enforced at construction), so tAd, uAd are identical
    L = bundle.cables[0].length
    m = bundle.cables[0].mass
    g = bundle.cables[0].g
    tAd = float(np.sqrt(L / g))
    uAd = L / tAd

    is_uniform = True 

    # Spacer pre-processing: precompute interior-node index and non-dim coefficients per edge
    spacer_data, ds, ns = bundle._prepare_spacers(ds, L, m, g, tAd, is_uniform)

    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
    
    # Last time step
    params = simulation.Parameters(
        ns=ns, t0=0.0, tf=tf, dt=dt, dr=tf, los=los, pp=False,
    )

    s_centre = 0.30
    sigma = 0.05
    bump = 0.01 * np.exp(-((np.concatenate((np.array([0.0]), np.cumsum(ds))) - s_centre) ** 2) / (2.0 * sigma**2))
    bump[0] = 0.0
    bump[-1] = 0.0
    
    ic = InitialConditions(un=[bump, np.zeros(ns)])
    
    res, new_ds = bundle.solve(is_uniform, spacer_data, ds, ns, params, force=None, zt=0.0, initial_conditions=ic)

    # Cable 1
    un1_interior = res.data["un_1"].values[-1, :]
    
    # Full vector
    un1_full = np.concatenate(([0.0], un1_interior, [0.0]))
    return un1_full, new_ds

def spatial_convergence_study():
    ns_list = [41, 81, 161, 321]
    ns_ref = 641
    L = 200.0
    
    print(f"Calcul de la solution de référence (ns={ns_ref})...")
    u_ref, ds_ref = run_convergence_sim_uni(ns_ref)
    
    errors = []
    ds_values = []
    
    for ns in ns_list:
        print(f"Calcul pour ns={ns}...")
        u_coarse, ds_coarse  = run_convergence_sim_uni(ns)
        
        ds = L / (ns - 1)
        ds_values.append(ds)
        
        step = (ns_ref - 1) // (ns - 1)
        
        u_ref_downsampled = u_ref[::step]
        
        # Error norm L2 
        l2_error = np.sqrt(np.mean((u_coarse - u_ref_downsampled)**2))
        errors.append(l2_error)
        
        print(f"  -> ds = {ds:.4f} m | L2 Error = {l2_error:.4e}")

    pente, intercept = np.polyfit(np.log(ds_values), np.log(errors), 1)
    print(f"Slope computed by regression: {pente:.3f}")

    plt.figure(figsize=(8, 6))
    
    plt.loglog(ds_values, errors, 'bo-', 
               label=f'Error of the model (Measured slope: {pente:.2f})', 
               linewidth=2)
    
    ds_array = np.array(ds_values)
    ref_slope = (ds_array / ds_array[-1])**2 * errors[-1]
    plt.loglog(ds_values, ref_slope, 'k--', label='Theoritical slope (Order 2)')
    
    plt.xlabel('Mesh size $ds$ [m]')
    plt.ylabel('Error norm $L_2$ with reference mesh [m]')
    plt.title('Spatial convergence study (Spacer width = $ds$)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     spatial_convergence_study()





#1D uniform mesh and 2D non-uniform mesh with fixed-width 2D spacer
def run_convergence_sim(mode, ns, tf=1.0, dt=5.0e-4, fixed_width=0.05):
    """
    1D convergence study withe uniform mesh.
    In 2D, the width is constant, forcing a non-uniform mesh.
    """
    cable = _aster570()
    
    if mode == "1D":
        spacers = [twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
                               is_2d=False, delta_n=0.0, delta_b=0.0)]
        is_uniform = True
    elif mode == "2D":
        spacers = [twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
                               is_2d=True, delta_n=0.0, delta_b=0.0,
                               j_b=200, j_t=200, width=fixed_width, l0=0.40)]
        is_uniform = False

    bundle = Bundle(cables=[cable, cable], spacers=spacers)


    s = np.linspace(0.0, 1.0, ns)
    ds = np.diff(s)

    # Per-cable non-dimensionalisation. All cables share L, m, g (enforced
    # at construction), so tAd, uAd are identical
    L = bundle.cables[0].length
    m = bundle.cables[0].mass
    g = bundle.cables[0].g
    tAd = float(np.sqrt(L / g))
    uAd = L / tAd

    # Spacer pre-processing: precompute interior-node index and non-dim coefficients per edge
    spacer_data, ds, ns = bundle._prepare_spacers(ds, L, m, g, tAd, is_uniform)

    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
    
    params = simulation.Parameters(
        ns=ns, t0=0.0, tf=tf, dt=dt, dr=tf, los=los, pp=False,
    )

    s_centre = 0.30
    sigma = 0.05
    bump = 0.01 * np.exp(-((np.concatenate((np.array([0.0]), np.cumsum(ds))) - s_centre) ** 2) / (2.0 * sigma**2))
    bump[0] = 0.0
    bump[-1] = 0.0
    ic = InitialConditions(un=[bump, np.zeros(ns)])
    
    res, ds = bundle.solve(is_uniform, spacer_data, ds, ns, params, force=None, zt=0.0, initial_conditions=ic)

    # Cable 1 last time step
    un1_interior = res.data["un_1"].values[-1, :]
    un1_full = np.concatenate(([0.0], un1_interior, [0.0]))

    return un1_full, ds


def order_of_convergence_study():
    ns_list = [41, 81, 161, 321]
    ns_ref = 641
    L = 200.0

    # Reference solution
    print(f"Calcul des références (ns={ns_ref})...")
    u_ref_1d, _ = run_convergence_sim("1D", ns_ref)
    u_ref_2d, ds_ref = run_convergence_sim("2D", ns_ref)
    
    err_1d, err_2d, ds_values = [], [], []
    
    # Coarse meshes
    for ns in ns_list:
        print(f"Calcul pour ns={ns}...")
        u_coarse_1d, _ = run_convergence_sim("1D", ns)
        u_coarse_2d, ds_coarse = run_convergence_sim("2D", ns)
        
        ds = L / (ns - 1)
        ds_values.append(ds)
        step = (ns_ref - 1) // (ns - 1)
        
        u_ref_1d_down = u_ref_1d[::step]
        x_coarse_2d = np.concatenate(([0.0], np.cumsum(ds_coarse)))
        x_ref_2d = np.concatenate(([0.0], np.cumsum(ds_ref)))

        u_ref_2d_interp = np.interp(x_coarse_2d, x_ref_2d, u_ref_2d)
                
        # Error norm L2
        err_1d.append(np.sqrt(np.mean((u_coarse_1d - u_ref_1d_down)**2)))
        err_2d.append(np.sqrt(np.mean((u_coarse_2d - u_ref_2d_interp)**2)))
        
    pente_1d, intercept_1d = np.polyfit(np.log(ds_values), np.log(err_1d), 1)
    pente_2d, intercept_2d = np.polyfit(np.log(ds_values), np.log(err_2d), 1)
    
    print(f"1D computed slope : {pente_1d:.3f}")
    print(f"2D computed slope : {pente_2d:.3f}")


    plt.figure(figsize=(9, 7))
    ds_arr = np.array(ds_values)
    
    plt.loglog(ds_arr, err_2d, 'ro-', 
               label=f'2D model (modified mesh) - Slope : {pente_2d:.2f}', 
               markersize=6)

    slope_O2 = (ds_arr / ds_arr[-1])**2 * err_1d[-1]
    slope_O1 = (ds_arr / ds_arr[-1])**1 * err_2d[-1]
    
    # plt.loglog(ds_arr, slope_O2, 'b--', alpha=0.6, label='Theoritical slope order 2')
    plt.loglog(ds_arr, slope_O1, 'r--', alpha=0.6, label='Theoritical slope order 1')
    
    plt.xlabel('Mesh size $\Delta s$ [m]')
    plt.ylabel('Error norm $L_2$ [m]')
    plt.title("Spatial convergence")
    plt.grid(True, which="both", ls=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     order_of_convergence_study()












#########
# ETUDE DE CONVERGENCE EN TEMPS
#########

def run_time_convergence_sim(mode, dt, ns=351, tf=1.0, fixed_width=0.05):
    """
    Temporal convergence simulation for a fixed spatial mesh.
    mode="1D" uses a 1D spacer (uniform mesh).
    mode="2D" uses a 2D spacer (modified non-uniform mesh).
    mode="2D_uni" uses a 2D spacer (uniform mesh).
    """

    cable = _aster570()
    L = cable.length
    
    if mode == "1D":
        spacers = [twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
                               is_2d=False, delta_n=0.0, delta_b=0.0)]
        is_uniform = True
    elif mode == "2D":
        spacers = [twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
                               is_2d=True, delta_n=0.0, delta_b=0.0,
                               j_b=200, j_t=200, width=fixed_width, l0=0.40)]
        is_uniform = False
    elif mode == "2D_uni":
        spacers = [twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
                               is_2d=True, delta_n=0.0, delta_b=0.0,
                               j_b=200, j_t=200, width=fixed_width, l0=0.40)]
        is_uniform = True

    bundle = Bundle(cables=[cable, cable], spacers=spacers)


    # cable0 = _aster570()
    # cable1 = _aster570()
    # L = cable0.length

    # if mode == "1D":
    #     spacer0 = Spacer(
    #         s=1./5,
    #         edges=(SpacerEdge(i=0, j=1, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0, 
    #                 is_2d = False, delta_n = 0.0, delta_b = 0.0, j_b = 200.0, j_t = 200.0, 
    #                 width = fixed_width, l0 = 0.4), ), 
    #                 is_2d = False, width = fixed_width)

    #     spacer1 = Spacer(
    #         s=2./5,
    #         edges=(SpacerEdge(i=0, j=1, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0, 
    #                 is_2d = False, delta_n = 0.0, delta_b = 0.0, j_b = 200.0, j_t = 200.0, 
    #                 width = fixed_width, l0 = 0.4), ), 
    #                 is_2d = False, width = fixed_width)

    #     spacer2 = Spacer(
    #         s=3./5,
    #         edges=(SpacerEdge(i=0, j=1, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0, 
    #                 is_2d = False, delta_n = 0.0, delta_b = 0.0, j_b = 200.0, j_t = 200.0, 
    #                 width = fixed_width, l0 = 0.4), ), 
    #                 is_2d = False, width = fixed_width)

    #     spacer3 = Spacer(
    #         s=4./5,
    #         edges=(SpacerEdge(i=0, j=1, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0, 
    #                 is_2d = False, delta_n = 0.0, delta_b = 0.0, j_b = 200.0, j_t = 200.0, 
    #                 width = fixed_width, l0 = 0.4), ), 
    #                 is_2d = False, width = fixed_width)

    #     is_uniform = True
    
    # elif mode == "2D":
    #     spacer0 = Spacer(
    #         s=1./5,
    #         edges=(SpacerEdge(i=0, j=1, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0, 
    #                 is_2d = True, delta_n = 0.0, delta_b = 0.0, j_b = 200.0, j_t = 200.0, 
    #                 width = fixed_width, l0 = 0.4), ), 
    #                 is_2d = True, width = fixed_width)

    #     spacer1 = Spacer(
    #         s=2./5,
    #         edges=(SpacerEdge(i=0, j=1, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0, 
    #                 is_2d = True, delta_n = 0.0, delta_b = 0.0, j_b = 200.0, j_t = 200.0, 
    #                 width = fixed_width, l0 = 0.4), ), 
    #                 is_2d = True, width = fixed_width)
        
    #     spacer2 = Spacer(
    #         s=3./5,
    #         edges=(SpacerEdge(i=0, j=1, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0, 
    #                 is_2d = True, delta_n = 0.0, delta_b = 0.0, j_b = 200.0, j_t = 200.0, 
    #                 width = fixed_width, l0 = 0.4), ), 
    #                 is_2d = True, width = fixed_width)

    #     spacer3 = Spacer(
    #         s=4./5,
    #         edges=(SpacerEdge(i=0, j=1, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0, 
    #                 is_2d = True, delta_n = 0.0, delta_b = 0.0, j_b = 200.0, j_t = 200.0, 
    #                 width = fixed_width, l0 = 0.4), ), 
    #                 is_2d = True, width = fixed_width)

    #     is_uniform = False
    
    # elif mode == "2D_uni":
    #     spacer0 = Spacer(
    #         s=1./5,
    #         edges=(SpacerEdge(i=0, j=1, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0, 
    #                 is_2d = True, delta_n = 0.0, delta_b = 0.0, j_b = 200.0, j_t = 200.0, 
    #                 width = fixed_width, l0 = 0.4), ), 
    #                 is_2d = True, width = fixed_width)

    #     spacer1 = Spacer(
    #         s=2./5,
    #         edges=(SpacerEdge(i=0, j=1, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0, 
    #                 is_2d = True, delta_n = 0.0, delta_b = 0.0, j_b = 200.0, j_t = 200.0, 
    #                 width = fixed_width, l0 = 0.4), ), 
    #                 is_2d = True, width = fixed_width)

    #     spacer2 = Spacer(
    #         s=3./5,
    #         edges=(SpacerEdge(i=0, j=1, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0, 
    #                 is_2d = True, delta_n = 0.0, delta_b = 0.0, j_b = 200.0, j_t = 200.0, 
    #                 width = fixed_width, l0 = 0.4), ), 
    #                 is_2d = True, width = fixed_width)

    #     spacer3 = Spacer(
    #         s=4./5,
    #         edges=(SpacerEdge(i=0, j=1, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0, 
    #                 is_2d = True, delta_n = 0.0, delta_b = 0.0, j_b = 200.0, j_t = 200.0, 
    #                 width = fixed_width, l0 = 0.4), ), 
    #                 is_2d = True, width = fixed_width)

    #     is_uniform = True


    # bundle = Bundle(
    #     cables=[cable0, cable1],
    #     spacers=[spacer0, spacer1, spacer2, spacer3],
    # )

    s = np.linspace(0.0, 1.0, ns)
    ds = np.diff(s)

    m = bundle.cables[0].mass
    g = bundle.cables[0].g
    tAd = float(np.sqrt(L / g))

    spacer_data, ds, ns_out = bundle._prepare_spacers(ds, L, m, g, tAd, is_uniform)
    los = np.linspace(0.0, 1.0, ns_out)[1:-1].tolist()
    
    params = simulation.Parameters(
        ns=ns_out, t0=0.0, tf=tf, dt=dt, dr=tf, los=los, pp=False,
    )

    s_centre = 1./3
    sigma = 0.05
    bump = 0.01 * np.exp(-((np.concatenate((np.array([0.0]), np.cumsum(ds))) - s_centre) ** 2) / (2.0 * sigma**2))
    bump[0] = 0.0
    bump[-1] = 0.0
    ic = InitialConditions(un=[bump, np.zeros(ns_out)])
    
    res, ds_final = bundle.solve(is_uniform, spacer_data, ds, ns_out, params, force=None, zt=0.0, initial_conditions=ic)

    un1_interior = res.data["un_1"].values[-1, :]
    un1_full = np.concatenate(([0.0], un1_interior, [0.0]))
    return un1_full

def time_convergence_study_multi():
    ns_fixed = 351
    dt_list = [1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3]
    dt_ref = 3.125e-4
    tf = 5.0

    print(f"Calcul des références (dt={dt_ref})...")
    u_ref_1d = run_time_convergence_sim("1D", dt_ref, ns=ns_fixed, tf=tf)
    u_ref_2d = run_time_convergence_sim("2D", dt_ref, ns=ns_fixed, tf=tf)
    u_ref_2d_uni = run_time_convergence_sim("2D_uni", dt_ref, ns=ns_fixed, tf=tf)
    
    err_1d, err_2d, err_2d_uni = [], [], []
    
    for dt in dt_list:
        print(f"Calcul pour dt={dt}...")
        u_coarse_1d = run_time_convergence_sim("1D", dt, ns=ns_fixed, tf=tf)
        u_coarse_2d = run_time_convergence_sim("2D", dt, ns=ns_fixed, tf=tf)
        u_coarse_2d_uni = run_time_convergence_sim("2D_uni", dt, ns=ns_fixed, tf=tf)
      
        err_1d.append(np.sqrt(np.mean((u_coarse_1d - u_ref_1d)**2)))
        err_2d.append(np.sqrt(np.mean((u_coarse_2d - u_ref_2d)**2)))
        err_2d_uni.append(np.sqrt(np.mean((u_coarse_2d_uni - u_ref_2d_uni)**2)))
        
    pente_1d, _ = np.polyfit(np.log(dt_list), np.log(err_1d), 1)
    pente_2d, _ = np.polyfit(np.log(dt_list), np.log(err_2d), 1)
    pente_2d_uni, _ = np.polyfit(np.log(dt_list), np.log(err_2d_uni), 1)
    
    print(f"1D computed time slope : {pente_1d:.3f}")
    print(f"2D computed time slope : {pente_2d:.3f}")
    print(f"2D uniform computed time slope : {pente_2d_uni:.3f}")

    plt.figure(figsize=(9, 7))
    dt_arr = np.array(dt_list)
    
    plt.loglog(dt_arr, err_1d, 'bs-', label=f'1D model - Slope: {pente_1d:.2f}')
    plt.loglog(dt_arr, err_2d, 'ro-', label=f'2D model non-uniform mesh - Slope: {pente_2d:.2f}')
    plt.loglog(dt_arr, err_2d_uni, 'g*-', label=f'2D model uniform mesh - Slope: {pente_2d_uni:.2f}')

    slope_O2 = (dt_arr / dt_arr[-1])**2 * err_2d[-1]
    
    plt.loglog(dt_arr, slope_O2, 'k--', alpha=0.6, label='Theoretical slope order 2')
    
    plt.xlabel('Time step $\Delta t$ [s]')
    plt.ylabel('Error norm $L_2$ [m]')
    plt.title("Temporal convergence")
    plt.grid(True, which="both", ls=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plt.show()

time_convergence_study_multi()