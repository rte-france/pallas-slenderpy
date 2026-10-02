"""Free vibration of a suspended cable released from a perturbed catenary.

Out of plane the cable vibrates at the taut-string frequency; in plane the
first symmetric mode is stiffened by the cable stretching (Irvine), so its
frequency sits above the taut-string one.
"""

import matplotlib.pyplot as plt
import numpy as np

from slenderpy.future import simulation
from slenderpy.future.cable import dynamic, frequency
from slenderpy.future.components import Conductor, Span

conductor = Conductor(mass=1.571, diameter=0.0313, axial_stiffness=3.76e07)
span = Span(length=400.0, tension=3.7e04)
parameters = simulation.Parameters(ns=101, tf=120.0, dt=2.0e-03, dr=2.0e-02, los=[0.5])

# 0.5 m half-sine offsets, out of plane and in plane
x, y, z = dynamic.equilibrium(conductor, span, parameters.ns)
bump = 0.5 * np.sin(np.pi * x / span.length)
res = dynamic.solve(
    conductor, span, parameters, initial_position=np.stack([x, y + bump, z + bump])
)
spc = simulation.spectrum(res)
f_taut = frequency.natural(span.length, span.tension, span.sld, conductor.mass)

fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6))
mid_z = z[parameters.ns // 2]
top.plot(res.lot(), res["y"].values[:, 0], label="y (out of plane)")
top.plot(res.lot(), res["z"].values[:, 0] - mid_z, label="z - static (in plane)")
bottom.semilogy(spc.lot(), spc["y"].values[:, 0], label="y")
bottom.semilogy(spc.lot(), spc["z"].values[:, 0], label="z")
for k in range(1, 4):
    bottom.axvline(k * f_taut, c="k", ls=":", lw=1)
bottom.axvline(f_taut, c="k", ls=":", lw=1, label="taut-string harmonics")
top.set(title="mid-span displacement", xlabel="time (s)", ylabel="m")
bottom.set(title="spectrum", xlabel="frequency (Hz)", ylabel="amplitude (m)")
bottom.set_xlim(0.0, 4.0 * f_taut)
for ax in (top, bottom):
    ax.grid(True)
    ax.legend(loc="upper right")
fig.suptitle("400 m span, tension 37 kN, free vibration")
fig.tight_layout()
plt.show()
