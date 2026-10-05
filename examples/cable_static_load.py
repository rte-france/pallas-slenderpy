"""Static shape of a cable under a steady wind and under an ice load.

The load comes on top of the weight, already in the catenary: the wind blows
the cable out of plane, the ice adds sag. ``shape.solve`` returns the global
position of the nodes.
"""

import matplotlib.pyplot as plt
import numpy as np

from slenderpy.cable import dynamic
from slenderpy.cable.static import shape
from slenderpy.components import Conductor, Span
from slenderpy.force.wind import ConstantWind, WindDrag

conductor = Conductor(mass=1.571, diameter=0.0313, axial_stiffness=3.76e07)
span = Span(length=400.0, tension=3.7e04)
ns = 201

# steady 25 m/s wind, drag evaluated on the structure at rest
x, y, z = dynamic.equilibrium(conductor, span, ns)
zeros = np.zeros(ns)
drag = WindDrag(diameter=conductor.diameter, wind=ConstantWind(25.0))
wind_y, wind_z = drag(x, 0.0, y, z, zeros, zeros)
windy = shape.solve(conductor, span, wind_y, wind_z, ns)

# 10 mm of radial ice (density 900 kg/m3)
radius = 0.5 * conductor.diameter
ice = -900.0 * 9.81 * np.pi * ((radius + 0.01) ** 2 - radius**2)
iced = shape.solve(conductor, span, 0.0, ice, ns)

fig, (side, top) = plt.subplots(1, 2, figsize=(11, 4))
side.plot(x, z, "k", label="weight only (catenary)")
side.plot(windy[0], windy[2], label="25 m/s wind")
side.plot(iced[0], iced[2], label=f"10 mm ice ({-ice:.1f} N/m)")
top.plot(x, y, "k")
top.plot(windy[0], windy[1], label="25 m/s wind")
side.set(title="side view", xlabel="x (m)", ylabel="z (m)")
top.set(title="plan view: blowout", xlabel="x (m)", ylabel="y (m)")
for ax in (side, top):
    ax.grid(True)
side.legend()
fig.suptitle("400 m span, tension 37 kN")
fig.tight_layout()
plt.show()
