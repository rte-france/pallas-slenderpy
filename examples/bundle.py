"""Bundle aeolian-vibration demos with interactive 3D visualization.

Two demos are provided:

- ``twin_bundle_aeolian``: two sub-conductors of a 220 kV-style span
  linked by one mid-span spacer, cable 0 excited at 5 Hz.
- ``triple_bundle_aeolian``: three sub-conductors in a flat horizontal
  layout linked by three asymmetric spacers along the span (a full
  spacer with edges 0-1 and 1-2 at s = 0.30, a half spacer with edge
  0-1 only at s = 0.50, a half spacer with edge 1-2 only at s = 0.70).
  Cable 0 is excited at 8 Hz so the dynamics sit in the 5-10 Hz band.
  A steady **lateral wind** is applied uniformly in the binormal
  direction on all three cables, ramped on over 0.5 s, so the whole
  bundle drifts sideways while the spacers keep the geometry coherent.
  Energy reaches cable 2 only by chaining through cable 1.

Run with the ``examples`` extra installed (which pulls in plotly):

    python -m pip install ".[examples]"
    python examples/bundle.py

A browser tab opens with the triple-bundle 3D scene. Drag to rotate,
scroll to zoom, use the slider at the bottom to scrub through time,
and the play button to auto-animate. To view the twin demo instead,
import and call ``twin_bundle_aeolian()`` directly.
"""

from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go

from slenderpy import simulation
from slenderpy.cable.bundle.bundle import Bundle
from slenderpy.cable.bundle.cable import Cable
from slenderpy.cable.bundle.spacer import Spacer, SpacerEdge, twin_spacer
from slenderpy.legacy import force as legacy_force


def visualize_bundle_3d(
    res: simulation.Results,
    bundle: Bundle,
    bundle_geometry: List[Tuple[float, float]],
    *,
    displacement_scale: float = 50.0,
    show_static_shape: bool = True,
    title: str = "",
) -> go.Figure:
    """Build an interactive 3D animation of the bundle dynamics.

    Parameters
    ----------
    res : slenderpy.simulation.Results
        Output of ``Bundle.solve``. Must carry per-sub-conductor
        ``un_i`` and ``ub_i`` traces.
    bundle : Bundle
        The bundle that produced ``res``. Used for span length, the
        rest-shape catenary, and the spacer topology.
    bundle_geometry : list of (y_offset, z_offset) in metres
        One entry per sub-conductor in the same order as
        ``bundle.cables``. Encodes where each sub-conductor sits in
        the rest cross-section: e.g. ``[(0.0, 0.0), (0.40, 0.0)]`` for
        a horizontal twin bundle with 40 cm spacing.
    displacement_scale : float, default 50.0
        Magnification factor on the dynamic ``un``, ``ub`` offsets.
        Cable sag is on the order of metres while transverse vibration
        amplitudes are on the order of millimetres, so a scale of
        ~50 is usually needed for the motion to be visible.
    show_static_shape : bool, default True
        Overlay each sub-conductor's rest catenary as a dashed grey
        line.
    title : str
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Call ``fig.show()`` to open in a browser tab or
        ``fig.write_html(path)`` to save.
    """
    if len(bundle_geometry) != bundle.n_cables:
        raise ValueError(
            f"bundle_geometry has {len(bundle_geometry)} entries but the "
            f"bundle has {bundle.n_cables} sub-conductors"
        )

    nc = bundle.n_cables
    cable0 = bundle.cables[0]
    L = cable0.length

    los_arr = np.array(res.los())
    x_m = los_arr * L
    times = np.asarray(res.lot())
    n_frames = len(times)

    # Static catenary on the output grid. All sub-conductors share the
    # same rest catenary because the Bundle invariant requires identical
    # length and linear mass; tensions could differ in principle but the
    # demo uses identical sub-conductors.
    cat = np.asarray(cable0.static_shape(los_arr))

    def _cable_xyz(i: int, k_t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_off, z_off = bundle_geometry[i]
        un_i = res.data[f"un_{i}"].values[k_t, :] * displacement_scale
        ub_i = res.data[f"ub_{i}"].values[k_t, :] * displacement_scale
        y = y_off + ub_i
        z = z_off - cat + un_i
        return x_m, y, z

    def _spacer_segments(k_t: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return the (x, y, z) endpoint pairs of every spacer edge for
        time index k_t. Each segment is a length-2 array per axis."""
        segments = []
        for sp_ in bundle.spacers:
            x_k = sp_.s * L
            for edge in sp_.edges:
                x_i, y_i, z_i = _cable_xyz(edge.i, k_t)
                x_j, y_j, z_j = _cable_xyz(edge.j, k_t)
                yi = float(np.interp(x_k, x_i, y_i))
                zi = float(np.interp(x_k, x_i, z_i))
                yj = float(np.interp(x_k, x_j, y_j))
                zj = float(np.interp(x_k, x_j, z_j))
                segments.append(
                    (
                        np.array([x_k, x_k]),
                        np.array([yi, yj]),
                        np.array([zi, zj]),
                    )
                )
        return segments

    cable_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    def _build_traces(k_t: int) -> List[go.Scatter3d]:
        traces: List[go.Scatter3d] = []
        for i in range(nc):
            x_i, y_i, z_i = _cable_xyz(i, k_t)
            traces.append(
                go.Scatter3d(
                    x=x_i,
                    y=y_i,
                    z=z_i,
                    mode="lines",
                    line=dict(width=5, color=cable_colors[i % len(cable_colors)]),
                    name=f"sub-conductor {i}",
                )
            )
        for sp_idx, (x_seg, y_seg, z_seg) in enumerate(_spacer_segments(k_t)):
            traces.append(
                go.Scatter3d(
                    x=x_seg,
                    y=y_seg,
                    z=z_seg,
                    mode="lines+markers",
                    line=dict(width=6, color="black"),
                    marker=dict(size=4, color="black"),
                    name=f"spacer edge {sp_idx}",
                    showlegend=(sp_idx == 0),
                )
            )
        return traces

    initial_traces = _build_traces(0)

    if show_static_shape:
        for i in range(nc):
            y_off, z_off = bundle_geometry[i]
            initial_traces.append(
                go.Scatter3d(
                    x=x_m,
                    y=np.full_like(x_m, y_off),
                    z=z_off - cat,
                    mode="lines",
                    line=dict(width=2, color="lightgrey", dash="dash"),
                    name=f"static {i}",
                    hoverinfo="skip",
                    showlegend=(i == 0),
                )
            )

    frames = [
        go.Frame(data=_build_traces(k_t), name=str(k_t)) for k_t in range(n_frames)
    ]

    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(k_t)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
            label=f"{times[k_t]:.2f}",
        )
        for k_t in range(n_frames)
    ]

    fig = go.Figure(
        data=initial_traces,
        frames=frames,
        layout=go.Layout(
            title=title,
            scene=dict(
                xaxis_title="span position (m)",
                yaxis_title="binormal (m)",
                zaxis_title="vertical (m)",
                aspectmode="data",
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.05,
                    y=0.0,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=40, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    currentvalue=dict(prefix="t = ", suffix=" s"),
                    pad=dict(t=40),
                    steps=slider_steps,
                )
            ],
        ),
    )
    return fig


def twin_bundle_aeolian() -> go.Figure:
    cable = Cable(
        mass=1.57,
        diameter=0.031,
        EA=3.76e07,
        length=200.0,
        tension=3.7e04,
        h=0.0,
    )
    bundle = Bundle(
        cables=[cable, cable],
        spacers=[twin_spacer(s=0.5, k_perp=5.0e4, c_perp=200.0)],
    )

    ns = 81
    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
    pm = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=6.0,
        dt=2.0e-3,
        dr=2.0e-2,
        los=los,
        pp=True,
    )

    excite = legacy_force.Excitation(
        f=5.0,
        a=0.5,
        s=0.25 * cable.length,
        m=1.0,
        L=cable.length,
        t0=0.0,
        tf=2.0,
        gravity=False,
    )

    def bundle_force(s, t, un_list, ub_list, vn_list, vb_list):
        # Cable 0 only receives the direct excitation; cable 1 is the
        # leeward partner that responds only through the spacer.
        fn0, fb0 = excite(s, t, un_list[0], ub_list[0], vn_list[0], vb_list[0])
        zeros = np.zeros_like(s)
        return [fn0, zeros], [fb0, zeros]

    res = bundle.solve(pm, force=bundle_force, zt=0.0)

    return visualize_bundle_3d(
        res,
        bundle,
        bundle_geometry=[(0.0, 0.0), (0.40, 0.0)],
        title="Twin bundle, mid-span spacer, cable 0 excited at 5 Hz",
    )


def triple_bundle_aeolian() -> go.Figure:
    """Three sub-conductors in a flat horizontal layout linked by three
    asymmetric spacers at varied span positions. Cable 0 is excited at
    8 Hz; energy reaches cable 2 only by chaining through cable 1
    because no spacer edge connects cables 0 and 2 directly."""
    cable = Cable(
        mass=1.57,
        diameter=0.031,
        EA=3.76e07,
        length=200.0,
        tension=3.7e04,
        h=0.0,
    )
    k_perp = 5.0e4
    c_perp = 200.0
    spacers = [
        # Full spacer at s = 0.30: edges (0, 1) and (1, 2).
        Spacer(
            s=0.30,
            edges=(
                SpacerEdge(i=0, j=1, k_perp=k_perp, c_perp=c_perp),
                SpacerEdge(i=1, j=2, k_perp=k_perp, c_perp=c_perp),
            ),
        ),
        # Half spacer at s = 0.50: edge (0, 1) only.
        Spacer(
            s=0.50,
            edges=(SpacerEdge(i=0, j=1, k_perp=k_perp, c_perp=c_perp),),
        ),
        # Half spacer at s = 0.70: edge (1, 2) only.
        Spacer(
            s=0.70,
            edges=(SpacerEdge(i=1, j=2, k_perp=k_perp, c_perp=c_perp),),
        ),
    ]
    bundle = Bundle(cables=[cable, cable, cable], spacers=spacers)

    # Higher excitation frequency than the twin demo requires finer dt
    # and a denser spatial grid to resolve the shorter wavelength.
    ns = 121
    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
    pm = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=3.0,
        dt=1.0e-3,
        dr=2.0e-2,
        los=los,
        pp=True,
    )

    excite_freq = 8.0  # Hz, inside the requested 5-10 Hz band
    excite = legacy_force.Excitation(
        f=excite_freq,
        a=0.5,
        s=0.20 * cable.length,
        m=1.0,
        L=cable.length,
        t0=0.0,
        tf=1.5,
        gravity=False,
    )

    # Steady lateral (binormal) wind, ramped smoothly over 0.5 s so the
    # initial transient does not excite spurious high-frequency content.
    # 0.5 N/m corresponds roughly to a moderate ~10 m/s wind on a 31 mm
    # conductor and produces a midspan static drift of ~7 cm at this
    # tension (delta = w L^2 / (8 T)).
    wind_lateral = 0.5  # N/m
    wind_ramp_tau = 0.5  # s

    def bundle_force(s, t, un_list, ub_list, vn_list, vb_list):
        # Cable 0 receives the 8 Hz vertical excitation. All three cables
        # receive the lateral wind drag.
        fn0, fb0 = excite(s, t, un_list[0], ub_list[0], vn_list[0], vb_list[0])
        zeros = np.zeros_like(s)
        ramp = min(1.0, t / wind_ramp_tau)
        fb_wind = ramp * wind_lateral * np.ones_like(s)
        return (
            [fn0, zeros, zeros],
            [fb0 + fb_wind, fb_wind, fb_wind],
        )

    res = bundle.solve(pm, force=bundle_force, zt=0.0)

    # Flat horizontal layout: 40 cm between adjacent sub-conductors.
    spacing = 0.40
    geometry = [(0.0, 0.0), (spacing, 0.0), (2.0 * spacing, 0.0)]

    return visualize_bundle_3d(
        res,
        bundle,
        bundle_geometry=geometry,
        # Wind-induced lateral drift is ~7 cm; magnify by 5 (not 50)
        # so the drift stays inside the bundle envelope while the
        # ~1 cm aeolian vibration remains visible.
        displacement_scale=10.0,
        title=(
            f"Flat triple bundle (40 cm spacing), 3 asymmetric spacers, "
            f"cable 0 excited at {excite_freq:.0f} Hz, lateral wind "
            f"{wind_lateral:.1f} N/m"
        ),
    )


if __name__ == "__main__":
    fig = triple_bundle_aeolian()
    fig.show()
