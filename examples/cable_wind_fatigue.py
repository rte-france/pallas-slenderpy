"""Cable in a turbulent wind, and the rainflow count 89 mm from a support.

The run starts from the static shape under the wind at t = 0. ``count_cycles``
reads the out-of-plane position at the Poffenberger-Swart point, stored in
``los``, and returns one row per cycle or half cycle.
"""

import matplotlib.pyplot as plt

from slenderpy.future import fatigue, simulation
from slenderpy.future.cable import dynamic
from slenderpy.future.components import Conductor, Span
from slenderpy.future.force.wind import UniformTurbulentWind, WindDrag

conductor = Conductor(mass=1.571, diameter=0.0313, axial_stiffness=3.76e07)
span = Span(length=400.0, tension=3.7e04)
near = 1.0 - 0.089 / span.length  # 89 mm from support 2, as a span fraction
parameters = simulation.Parameters(
    ns=101, tf=300.0, dt=2.0e-03, dr=1.0e-02, los=[0.5, near]
)

wind = UniformTurbulentWind(
    mean=10.0, std=2.0, length_scale=150.0, t_start=0.0, t_end=300.0, dt=0.1, seed=1
)
drag = WindDrag(diameter=conductor.diameter, wind=wind)
res = dynamic.solve(conductor, span, parameters, force=drag)
cycles = fatigue.count_cycles(res, "y", span, distance=0.089, support="right")

fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6))
top.plot(res.lot(), res["y"].values[:, 0], c="C0")
top.set(title="response to the wind", xlabel="time (s)", ylabel="mid-span y (m)")
speed = top.twinx()
speed.plot(wind.times, wind.speed, c="0.6", lw=0.8)
speed.set_ylabel("wind speed (m/s)", color="0.4")
top.set_zorder(speed.get_zorder() + 1)  # response drawn over the wind
top.patch.set_visible(False)
bottom.hist(
    1e3 * cycles["range"], bins=40, weights=cycles["count"], log=True, color="C1"
)
bottom.set(
    title=f"rainflow count at 89 mm from the support ({cycles['count'].sum():.0f} cycles)",
    xlabel="range of y (mm)",
    ylabel="cycles",
)
for ax in (top, bottom):
    ax.grid(True)
fig.suptitle("400 m span, tension 37 kN, turbulent wind 10 m/s +- 2 m/s")
fig.tight_layout()
plt.show()
