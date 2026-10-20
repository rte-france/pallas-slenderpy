"""Free vibration of a suspended cable released from a perturbed catenary.

Out of plane the cable vibrates at the taut-string frequency; in plane the
first symmetric mode is stiffened by the cable stretching (Irvine), so its
frequency sits above the taut-string one. The run uses the implicit strain,
which conserves the energy: the second figure stacks the kinetic, tension and
axial energies, whose total stays constant.
"""

import matplotlib.pyplot as plt
import numpy as np

from slenderpy import simulation
from slenderpy.cable import dynamic, frequency
from slenderpy.components import Conductor, Span

conductor = Conductor(mass=1.571, diameter=0.0313, axial_stiffness=3.76e07)
span = Span(length=400.0, tension=3.7e04)
parameters = simulation.Parameters(ns=101, tf=120.0, dt=2.0e-03, dr=2.0e-02, los=[0.5])

# 0.5 m half-sine offsets, out of plane and in plane
x, y, z = dynamic.equilibrium(conductor, span, parameters.ns)
bump = 0.5 * np.sin(np.pi * x / span.length)
res = dynamic.solve(
    conductor,
    span,
    parameters,
    initial_position=np.stack([x, y + bump, z + bump]),
    implicit_strain=True,
    energy=True,
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

# energy balance; each term is shifted by its minimum over the run, which
# keeps the layers positive and only moves the total by a constant
terms = [
    ("kinetic", res["kinetic_energy"].values),
    ("tension", res["tension_energy"].values),
    ("axial", res["axial_energy"].values),
]
layers = [values - values.min() for _, values in terms]
balance = sum(values for _, values in terms)
drift = np.abs(balance - balance[0]).max() / np.abs(balance).max()
fig, ax = plt.subplots(figsize=(10, 4))
ax.stackplot(res.lot(), layers, labels=[name for name, _ in terms])
ax.plot(res.lot(), sum(layers), "k", lw=1, label="total")
ax.set(xlabel="time (s)", ylabel="energy - its minimum (J)")
ax.set_title(f"energy balance, implicit strain (relative drift {drift:.1e})")
ax.grid(True)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
plt.show()
