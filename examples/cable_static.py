"""Cable under its weight: catenary, parabolic and elastic (nleq) closed forms.

At a given tension the three shapes are visually the same; their differences
grow as the tension drops and the sag increases.
"""

import matplotlib.pyplot as plt
import numpy as np

from slenderpy.cable.static import catenary, nleq, parabolic

# ASTER 570 conductor on a 400 m span with a 20 m support level difference
linm, axs, rts = 1.571, 3.653e07, 1.853e05
lspan, sld = 400.0, 20.0
x = np.linspace(0.0, lspan, 401)

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
for k, ratio in enumerate([0.05, 0.1, 0.2]):
    tension = ratio * rts
    z_cat = catenary.shape(x, lspan, tension, sld, linm)
    z_par = parabolic.shape(x, lspan, tension, sld, linm)
    x_nl, z_nl = nleq.shape(2001, lspan, tension, sld, linm, axs)
    z_nl = np.interp(x, x_nl, z_nl)
    left.plot(x, z_cat, c=f"C{k}", label=f"T = {ratio:.0%} RTS")
    right.plot(x, 1e2 * (z_par - z_cat), c=f"C{k}", ls="--")
    right.plot(x, 1e2 * (z_nl - z_cat), c=f"C{k}", ls="-")
right.plot([], [], "k--", label="parabolic - catenary")
right.plot([], [], "k-", label="nleq - catenary")
left.set(title="catenary shape", xlabel="x (m)", ylabel="z (m)")
right.set(title="difference to the catenary", xlabel="x (m)", ylabel="dz (cm)")
for ax in (left, right):
    ax.grid(True)
    ax.legend()
fig.suptitle("400 m span, 20 m level difference, ASTER 570")
fig.tight_layout()
plt.show()
