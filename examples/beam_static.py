"""Static shape of a clamped span under its weight, and its bending boundary layer.

Away from the clamps the three bending laws give the same sag; within a few
``sqrt(EI / tension)`` of a clamp the curvature, and with it the stiffness of
the Bouc-Wen law, changes by orders of magnitude.
"""

import matplotlib.pyplot as plt
import numpy as np

from slenderpy.beam.static import shape
from slenderpy.boundary_condition import clamped
from slenderpy.components import Conductor, Span

conductor = Conductor(mass=1.57, ei_min=28.28, ei_max=2155.07, beta_flexion=6.438e-07)
span = Span(length=50.0, tension=2.0e04, boundary_conditions=clamped())
n = 5001  # 1 cm steps, to resolve the EI min boundary layer (4 cm)
x = np.linspace(0.0, span.length, n)
weight = np.full(n, -9.81 * conductor.mass)  # N/m
laws = [
    ("constant", conductor.ei_max, "constant $EI_{max}$"),
    ("constant", conductor.ei_min, "constant $EI_{min}$"),
    ("varying", None, "Bouc-Wen"),
]

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
near = x[1:-1] <= 1.5
for model, ei, label in laws:
    z = shape.solve(conductor, span, weight, n, model=model, ei=ei)
    curvature = np.diff(z, 2) / (x[1] - x[0]) ** 2  # at the interior nodes
    left.plot(x, z, label=label)
    right.semilogy(x[1:-1][near], np.abs(curvature[near]), label=label)
right.axhline(conductor.beta_flexion * span.tension, c="k", ls=":", label=r"$\chi_0$")
left.set(title="sag over the span", xlabel="x (m)", ylabel="z (m)")
right.set(title="curvature near the clamp", xlabel="x (m)", ylabel="|curvature| (1/m)")
for ax in (left, right):
    ax.grid(True)
right.legend()
fig.suptitle("50 m clamped span under its weight, tension 20 kN")
fig.tight_layout()
plt.show()
