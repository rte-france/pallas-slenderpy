"""Free vibration of a clamped span released from 1.5 times its static shape.

The constant laws keep their amplitude; the Bouc-Wen law dissipates energy
through its hysteresis at the clamps, so its vibration decays. The second
figure shows that hysteresis: the moment-curvature loops at the clamp, inside
the static envelope of the law.
"""

import matplotlib.pyplot as plt
import numpy as np

from slenderpy import simulation
from slenderpy.beam import bending
from slenderpy.beam.dynamic import solve_dynamic
from slenderpy.beam.static import shape
from slenderpy.boundary_condition import clamped
from slenderpy.components import Conductor, Span
from slenderpy.force.core import Gravity

conductor = Conductor(mass=1.57, ei_min=28.28, ei_max=2155.07, beta_flexion=6.438e-07)
span = Span(length=20.0, tension=2.0e03, boundary_conditions=clamped())
parameters = simulation.Parameters(
    ns=1001, tf=20.0, dt=2.0e-03, dr=1.0e-02, los=[0.0, 0.5]
)
weight = Gravity(conductor.mass)
laws = [
    ("constant", conductor.ei_max, "constant $EI_{max}$"),
    ("constant", conductor.ei_min, "constant $EI_{min}$"),
    ("varying", None, "Bouc-Wen"),
]

fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for model, ei, label in laws:
    rest = shape.solve(
        conductor,
        span,
        np.full(parameters.ns, -9.81 * conductor.mass),
        parameters.ns,
        model=model,
        ei=ei,
    )
    res = solve_dynamic(
        conductor,
        span,
        parameters,
        model=model,
        ei=ei,
        force=weight,
        initial_position=1.5 * rest,
    )
    t = res.lot()
    top.plot(t, 1e3 * (res["z"].values[:, 1] - rest[parameters.ns // 2]), label=label)
    bottom.plot(t, res["moment"].values[:, 0], label=label)
top.set(ylabel="mid-span z - static (mm)", title="released from 1.5 x static shape")
bottom.set(xlabel="time (s)", ylabel="moment at the clamp (N.m)")
for ax in (top, bottom):
    ax.grid(True)
top.legend(loc="upper right")
fig.suptitle("20 m clamped span, tension 2 kN")
fig.tight_layout()

# hysteresis of the last (Bouc-Wen) run at the clamp, against the static law
curvature, moment = res["curvature"].values[:, 0], res["moment"].values[:, 0]
envelope = np.linspace(curvature.min(), curvature.max(), 200)
law = bending.create(conductor, span, "varying")
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(curvature, moment, lw=0.8, label="Bouc-Wen, dynamic")
ax.plot(envelope, law.moment(envelope), "k--", label="static envelope")
ax.set(xlabel="curvature at the clamp (1/m)", ylabel="moment at the clamp (N.m)")
ax.set_title("hysteresis at the clamp")
ax.grid(True)
ax.legend()
fig.tight_layout()
plt.show()
