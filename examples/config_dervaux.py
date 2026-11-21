import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from slenderpy import simulation
from slenderpy.cable.bundle.bundle import Bundle, InitialConditions
from slenderpy.cable.bundle.cable import Cable
from slenderpy.cable.bundle.spacer import twin_spacer, Spacer, SpacerEdge
from slenderpy.force_bundle import Excitation


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

    ic = InitialConditions(un=[np.zeros(ns), np.zeros(ns)])

    force_motor = Excitation(f = 5.0,
        a = 2.0,
        s = 1./L,
        nc = 2,
        c = 0,
        m = m,
        L = L,
        t0 = 0.0,
        tf = 35.0,
        gravity = True,
        g= 9.81)

    res, _ = bundle.solve(is_uniform, spacer_data, ds, ns, params, force=force_motor, zt=0.0, initial_conditions=ic)

    t_axis = np.asarray(res.lot())
    los_arr = np.asarray(res.los())
    s_grid = np.concatenate(([0.0], np.cumsum(ds)))

    un0 = res.data["un_0"].values 
    un1 = res.data["un_1"].values

    s_motor = 1./L 
    s_sensor = 0.5                      
    target_s = [s_motor, s_sensor]
    positions = ["motor position", "midspan"]

    zeros_state = [np.zeros_like(s_grid), np.zeros_like(s_grid)]
    fn0_hist = np.zeros((len(t_axis), len(s_grid)))
    fn1_hist = np.zeros((len(t_axis), len(s_grid)))
    for k, tk in enumerate(t_axis):
        fn_list, _ = force_motor(
            s_grid, float(tk), zeros_state, zeros_state, zeros_state, zeros_state
        )
        fn0_hist[k, :] = fn_list[0]
        fn1_hist[k, :] = fn_list[1]

    for s_pos, pos in zip(target_s, positions):

        i_los = int(np.argmin(np.abs(los_arr - s_pos)))
        z1 = un0[:, i_los]
        z2 = un1[:, i_los]

        i_grid = int(np.argmin(np.abs(s_grid[1:-1] - s_pos))) + 1
        cell_len = 0.5 * (s_grid[i_grid + 1] - s_grid[i_grid - 1]) * float(cable.length)
        force1 = fn0_hist[:, i_grid] * cell_len
        force2 = fn1_hist[:, i_grid] * cell_len

        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(211)
        ax.plot(t_axis, z1, label=f"Normal displacement at {pos} (cable 1)")
        ax.plot(t_axis, z2, label=f"Normal displacement at {pos} (cable 2)")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Displacement [m]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = fig.add_subplot(212)
        ax.plot(t_axis, force1, label=f"Normal force at {pos} (cable 1)")
        ax.plot(t_axis, force2, label=f"Normal force at {pos} (cable 2)")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Force [N]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        dt_out = t_axis[1] - t_axis[0]
        freq = np.fft.fftfreq(t_axis.shape[-1], dt_out)
        keep = freq >= 0
        freq_pos = freq[keep]

        fourier_z1 = np.fft.fft(z1 - np.mean(z1))
        norm_z1 = np.abs(fourier_z1)[keep]

        fourier_z2 = np.fft.fft(z2 - np.mean(z2))
        norm_z2 = np.abs(fourier_z2)[keep]

        ax.plot(freq_pos, norm_z1, "tab:blue", label=f"FFT(z) at {pos} (cable 1)")
        ax.plot(freq_pos, norm_z2, "tab:orange", label=f"FFT(z) at {pos} (cable 2)")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("|FFT(z)|")
        ax.grid(True)
        ax.legend()

        for norm_z, color in ((norm_z1, "tab:blue"), (norm_z2, "tab:orange")):
            peaks, _ = signal.find_peaks(norm_z)
            if peaks.size > 0:
                values = norm_z[peaks]
                top_idx = np.flip(np.argsort(values))[:5]
                f_peaks = freq_pos[peaks]
                for idx_p in top_idx:
                    ax.text(f_peaks[idx_p], values[idx_p], f"f={f_peaks[idx_p]:.2f}Hz", color=color)

        fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    run()

