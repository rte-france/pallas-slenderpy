"""Energy balance of the dynamic cable solve.

Counterpart of the beam one, on the non-dimensional Lee-Perkins equations:
each row of the Crank-Nicolson step multiplied by ``du = u_new - u_old`` and
summed over the interior nodes is the work of that term over the step. The
kinetic, tension and axial energies are functions of the state, the external
and damping works are accumulated step by step, and

    kinetic_energy + tension_energy + axial_energy + damping_work - external_work

is constant up to the Newton residual with the implicit strain. With the
explicit (lagged) strain it drifts: the axial work over a step is
``vl2 * e_old * de`` rather than the increment of ``vl2/2 * e**2``. The ends
are pinned, so the supports do no work. The definitions are detailed in the
cable page of the documentation, section "Energy balance".
"""

from __future__ import annotations

import numpy as np

from slenderpy.cable._model import _Geometry, _strain

NAMES = (
    "kinetic_energy",
    "tension_energy",
    "axial_energy",
    "external_work",
    "damping_work",
)


class EnergyBalance:
    """Energies and cumulated works of a dynamic cable solve.

    All inputs are non-dimensional (lengths by the cable length, times by
    ``sqrt(L/g)``, forces by the weight); the values are returned in J.

    Parameters
    ----------
    geom : _Geometry
        Equilibrium geometry of the solve.
    vt2 : float
        Non-dimensional tension.
    vl2 : float
        Non-dimensional axial stiffness.
    damping : float
        Damping factor of the scheme, ``(1 - damping) v_new = (1 + damping)
        v_old + ...``; zero or negative.
    scale : float
        Energy scale ``mass * g * L**2`` (J).
    """

    def __init__(
        self,
        geom: _Geometry,
        vt2: float,
        vl2: float,
        damping: float,
        scale: float,
    ) -> None:
        self.geom = geom
        self.vt2 = vt2
        self.vl2 = vl2
        self.damping = damping
        self.scale = scale
        # the grid of the solver's operators, uniform in arc fraction
        self.h = 1.0 / (len(geom.s) - 1)
        self.energies = dict.fromkeys(NAMES, 0.0)

    def _state(self, un, ub, vn, vb) -> None:
        """Energies that are functions of the state."""
        h = self.h
        slope = (np.diff(un) / h) ** 2 + (np.diff(ub) / h) ** 2
        strain = _strain(un, ub, self.geom, self.vt2)
        self.energies["kinetic_energy"] = (
            0.5 * h * np.sum(vn[1:-1] ** 2 + vb[1:-1] ** 2)
        )
        self.energies["tension_energy"] = 0.5 * self.vt2 * h * np.sum(slope)
        self.energies["axial_energy"] = 0.5 * self.vl2 * strain**2

    def start(self, un, ub, vn, vb) -> None:
        """Energies of the initial state, works set to zero.

        Parameters
        ----------
        un, ub : np.ndarray
            Normal and binormal displacement at the nodes.
        vn, vb : np.ndarray
            Normal and binormal velocity at the nodes.
        """
        self.energies = dict.fromkeys(NAMES, 0.0)
        self._state(un, ub, vn, vb)

    def step(self, un_old, ub_old, un_new, ub_new, vn_new, vb_new, fn, fb, dt) -> None:
        """Add the works of one time step and update the state energies.

        Parameters
        ----------
        un_old, ub_old : np.ndarray
            Displacement at the start of the step.
        un_new, ub_new : np.ndarray
            Displacement at the end of the step.
        vn_new, vb_new : np.ndarray
            Velocity at the end of the step.
        fn, fb : np.ndarray
            Force the step uses, the mean of its two evaluations.
        dt : float
            Time step.
        """
        h = self.h
        dun = (un_new - un_old)[1:-1]
        dub = (ub_new - ub_old)[1:-1]
        self.energies["external_work"] += h * np.sum(fn[1:-1] * dun + fb[1:-1] * dub)
        # the scheme's damping term is damping * (v_old + v_new), and
        # (v_old + v_new) / 2 = du / dt
        self.energies["damping_work"] += (
            -2.0 * self.damping * h * np.sum(dun**2 + dub**2) / dt**2
        )
        self._state(un_new, ub_new, vn_new, vb_new)

    def values(self) -> list[float]:
        """Current energies and works in J, in the order of ``NAMES``."""
        return [self.scale * self.energies[name] for name in NAMES]
