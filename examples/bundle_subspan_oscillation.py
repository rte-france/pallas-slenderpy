"""Twin bundle subspan response: wake-coupled aerodynamics on a flexible bundle.

Demonstrates the dynamic behaviour of a twin conductor bundle held in
formation by four spacers, exposed to a steady cross-wind. The aerodynamic
loading uses :mod:`slenderpy.wake` (Wu et al. 2018 fit, Annex A of the
Eurobios/RTE note ESCB-2020-11-12), so the leeward conductor sees a
reduced drag and a position-dependent lift relative to the windward one.

Background
----------
Wake-induced subspan oscillations are a well documented aeroelastic
phenomenon on twin and quad bundles: the leeward conductor moves on
elliptic orbits in the plane perpendicular to the wind, at the natural
frequency of the conductor segment between two adjacent spacers (the
"subspan"). Typical setups have separation-to-diameter ratios y/d in
[12, 20], wind speeds 7 to 18 m/s, and subspan frequencies 0.5 to 3 Hz.
See Snegovskiy (2010) PhD thesis chapter 1, Wang and Lilien (1998), and
CIGRE TB 277 for a thorough overview.

Setup
-----
- Twin bundle, ASTER570 sub-conductors
- Span 200 m, tension 37 kN (~ 20 % of breaking load), height difference 0
- Separation 0.45 m (y/d = 14.5)
- 4 spacers at s = 0.2, 0.4, 0.6, 0.8 -> 5 subspans of 40 m
- Steady wind 5 m/s along +binormal (kept low so that the downwind drift
  stays well below the bundle separation, avoiding the leeward/windward
  role switch that would corrupt the wake assignment)
- Initial half-sine perturbation in the normal direction on the leeward
  conductor over the middle subspan to excite the subspan mode

Theoretical subspan frequency for a string of length L_sub = 40 m,
tension 37 kN, mass 1.57 kg/m:
    f_sub = (1 / (2 L_sub)) * sqrt(T / m) ~= 1.92 Hz

Limitations
-----------
The Wu 2018 fit is quasi-steady and the coefficients depend only on the
relative cable position; in the present implementation this is sufficient
to reproduce the wake drag deficit and the position-dependent lift, but
not the unsteady mechanisms that drive self-excited subspan flutter. The
example therefore shows forced response plus the natural subspan mode of
the leeward conductor, with the lift coefficient acting as a soft
coupling that modulates the orbit shape. Reproducing self-excited subspan
oscillations would require an angle-of-attack dependent aero model (e.g.
Den Hartog galloping criterion) which is not part of the current code.

Run:
    python examples/bundle_subspan_oscillation.py
"""

import matplotlib.pyplot as plt
import numpy as np

from slenderpy import simulation, wake
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
    d = cable.diameter
    L = cable.length

    # Horizontal twin bundle, 45 cm separation in the binormal direction.
    separation = 0.45
    geometry = [(0.0, 0.0), (separation, 0.0)]
    y_over_d = separation / d
    print(f"Bundle geometry: y/d = {y_over_d:.1f}")

    # Four very rigid spacers carving the span into five 80 m subspans.
    # k_perp = 1e9 follows the regression test
    # test_subspan_string_frequency_with_two_spacers: the spacers behave
    # as rigid pins and the subspans inherit the clamped-string modes.
    k_perp = 1.0e9
    c_perp = 0.0
    spacer_positions = [0.2, 0.4, 0.6, 0.8]
    spacers = [
        Spacer(
            s=sp,
            edges=(SpacerEdge(i=0, j=1, k_perp=k_perp, c_perp=c_perp),),
        )
        for sp in spacer_positions
    ]
    bundle = Bundle(cables=[cable, cable], spacers=spacers)

    L_sub = L * 0.2  # 80 m
    f_sub_theory = (1.0 / (2.0 * L_sub)) * np.sqrt(cable.tension / cable.mass)
    print(f"Theoretical subspan frequency: {f_sub_theory:.3f} Hz")

    # Output every 10 ms; integration step 1 ms. Run long enough to
    # resolve >= 30 subspan periods with good FFT resolution.
    ns = 161
    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
    params = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=25.0,
        dt=1.0e-3,
        dr=1.0e-2,
        los=los,
        pp=False,
    )

    wind_speed = 5.0  # m/s, low enough to keep the bundle drift well
    # below the windward/leeward gap (no role switch over 30 s).
    force = wake.aero_force_bundle(
        bundle_geometry=geometry,
        diameter=d,
        wind_speed=wind_speed,
        cd_windward=1.0,
        cl_windward=0.0,
    )

    # Initial perturbation: half-sine on the central subspan (s in [0.4, 0.6])
    # of cable 1 (leeward), amplitude 1 cm in the normal direction. This
    # excites the first subspan mode cleanly.
    s_grid = np.linspace(0.0, 1.0, ns)
    sub_left = 0.4
    sub_right = 0.6
    bump = np.zeros(ns)
    inside = (s_grid >= sub_left) & (s_grid <= sub_right)
    bump[inside] = 0.01 * np.sin(
        np.pi * (s_grid[inside] - sub_left) / (sub_right - sub_left)
    )
    ic = InitialConditions(un=[np.zeros(ns), bump])

    res = bundle.solve(params, force=force, zt=2.0e-3, initial_conditions=ic)

    t_axis = np.asarray(res.lot())
    un0 = res.data["un_0"].values  # (nt, n_los)
    un1 = res.data["un_1"].values
    ub0 = res.data["ub_0"].values
    ub1 = res.data["ub_1"].values

    # Mid-subspan column indices in res.los() (n_los = ns - 2; physical s
    # values are los).
    los_arr = np.array(los)
    subspan_centres = [0.1, 0.3, 0.5, 0.7, 0.9]
    cc = [int(np.argmin(np.abs(los_arr - sc))) for sc in subspan_centres]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # (a) Orbits at each subspan centre on the leeward cable. Strip the
    # mean (downwind drift) so the orbit centres on zero.
    ax_orb = fig.add_subplot(gs[0, 0])
    for k, sc in zip(cc, subspan_centres):
        ub_k = ub1[:, k] - ub1[:, k].mean()
        un_k = un1[:, k] - un1[:, k].mean()
        ax_orb.plot(ub_k * 1e3, un_k * 1e3, label=f"s/L = {sc:.1f}", lw=0.7)
    ax_orb.set_xlabel("binormal disp. [mm] (drift removed)")
    ax_orb.set_ylabel("normal disp. [mm] (drift removed)")
    ax_orb.set_title("Leeward cable orbits at subspan centres")
    ax_orb.legend(fontsize=8, loc="upper right")
    ax_orb.set_aspect("equal", adjustable="datalim")
    ax_orb.grid(True, alpha=0.3)

    # (b) FFT of uN at the centre of the middle subspan. Skip the first
    # second of data to remove the initial transient.
    dr_out = float(t_axis[1] - t_axis[0])
    skip = int(1.0 / dr_out)
    sig = un1[skip:, cc[2]] - un1[skip:, cc[2]].mean()
    n_fft = len(sig)
    freqs = np.fft.rfftfreq(n_fft, dr_out)
    fft_mag = np.abs(np.fft.rfft(sig)) / n_fft
    peak_idx = int(np.argmax(fft_mag[1:]) + 1)
    f_peak = freqs[peak_idx]

    ax_fft = fig.add_subplot(gs[0, 1])
    ax_fft.plot(freqs, fft_mag * 1e3, color="C1")
    ax_fft.axvline(
        f_sub_theory,
        color="k",
        ls="--",
        lw=1.0,
        label=f"theory {f_sub_theory:.2f} Hz",
    )
    ax_fft.axvline(
        f_peak, color="C2", ls=":", lw=1.5, label=f"measured {f_peak:.2f} Hz"
    )
    ax_fft.set_xlim(0.0, 5.0)
    ax_fft.set_xlabel("frequency [Hz]")
    ax_fft.set_ylabel("FFT amplitude [mm]")
    ax_fft.set_title("Spectrum of uN at middle-subspan centre")
    ax_fft.legend(fontsize=8)
    ax_fft.grid(True, alpha=0.3)

    # (c) Max-amplitude envelope of uN per cable over time (subtracting
    # the time-mean per location to remove the static drift).
    un0_dyn = un0 - un0.mean(axis=0, keepdims=True)
    un1_dyn = un1 - un1.mean(axis=0, keepdims=True)
    env0 = np.max(np.abs(un0_dyn), axis=1)
    env1 = np.max(np.abs(un1_dyn), axis=1)
    ax_env = fig.add_subplot(gs[0, 2])
    ax_env.plot(t_axis, env0 * 1e3, label="cable 0 (windward)", color="C0")
    ax_env.plot(t_axis, env1 * 1e3, label="cable 1 (leeward)", color="C1")
    ax_env.set_xlabel("time [s]")
    ax_env.set_ylabel("max |uN| - mean(uN) [mm]")
    ax_env.set_title("Dynamic envelope of normal displacement")
    ax_env.legend()
    ax_env.grid(True, alpha=0.3)

    # (d) Mean binormal drift along the span at the end of the simulation
    # (last 5 seconds averaged) - shows the static drag deficit.
    avg_window = t_axis >= (t_axis[-1] - 5.0)
    ub0_mean = ub0[avg_window].mean(axis=0)
    ub1_mean = ub1[avg_window].mean(axis=0)
    ax_drift = fig.add_subplot(gs[1, 0])
    s_phys = los_arr * L
    ax_drift.plot(s_phys, ub0_mean * 100.0, label="cable 0 (windward)", color="C0")
    ax_drift.plot(s_phys, ub1_mean * 100.0, label="cable 1 (leeward)", color="C1")
    for sp in spacer_positions:
        ax_drift.axvline(sp * L, color="k", ls=":", lw=0.6)
    ax_drift.set_xlabel("s [m]")
    ax_drift.set_ylabel("mean binormal disp. [cm]")
    ax_drift.set_title("Steady downwind drift (last 5 s)")
    ax_drift.legend()
    ax_drift.grid(True, alpha=0.3)

    # (e) Time trace of uN at middle-subspan centre, leeward cable.
    ax_trace = fig.add_subplot(gs[1, 1])
    ax_trace.plot(
        t_axis, (un1[:, cc[2]] - un1[:, cc[2]].mean()) * 1e3, color="C1", lw=0.7
    )
    ax_trace.set_xlabel("time [s]")
    ax_trace.set_ylabel("uN (drift removed) [mm]")
    ax_trace.set_title("uN at middle-subspan centre, leeward")
    ax_trace.grid(True, alpha=0.3)

    # (f) Space-time map of uN on the leeward cable.
    ax_map = fig.add_subplot(gs[1, 2])
    vmax = np.max(np.abs(un1_dyn)) * 1e3
    im = ax_map.imshow(
        un1_dyn * 1e3,
        aspect="auto",
        extent=(s_phys[0], s_phys[-1], t_axis[-1], t_axis[0]),
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    for sp in spacer_positions:
        ax_map.axvline(sp * L, color="k", ls=":", lw=0.6)
    ax_map.set_xlabel("s [m]")
    ax_map.set_ylabel("time [s]")
    ax_map.set_title("Space-time map of uN (leeward, drift removed)")
    fig.colorbar(im, ax=ax_map, label="uN [mm]")

    fig.suptitle(
        f"Subspan response, twin bundle 400 m, y/d = {y_over_d:.1f}, "
        f"wind {wind_speed:.0f} m/s, subspan natural freq. "
        f"~{f_sub_theory:.2f} Hz"
    )

    print(f"Measured FFT peak: {f_peak:.3f} Hz "
          f"(theory {f_sub_theory:.3f} Hz, "
          f"relative error {100 * abs(f_peak - f_sub_theory) / f_sub_theory:.1f} %)")

    plt.show()


if __name__ == "__main__":
    run()
