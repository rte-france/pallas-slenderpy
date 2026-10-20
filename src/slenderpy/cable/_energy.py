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
        self.works = dict.fromkeys(NAMES[3:], 0.0)

    def start(self) -> None:
        """Set the works to zero."""
        self.works = dict.fromkeys(NAMES[3:], 0.0)

    def step(self, dun, dub, fn, fb, dt) -> None:
        """Add the works of one time step.

        Only the works are accumulated here: the state energies are computed
        by ``values`` when needed.

        Parameters
        ----------
        dun, dub : np.ndarray
            Normal and binormal displacement increments of the step, on the
            interior nodes.
        fn, fb : np.ndarray
            Force the step uses, the mean of its two evaluations, on all
            nodes.
        dt : float
            Time step.
        """
        h = self.h
        self.works["external_work"] += h * np.sum(fn[1:-1] * dun + fb[1:-1] * dub)
        # the scheme's damping term is damping * (v_old + v_new), and
        # (v_old + v_new) / 2 = du / dt
        if self.damping:
            self.works["damping_work"] += (
                -2.0 * self.damping * h * np.sum(dun**2 + dub**2) / dt**2
            )

    def values(self, un, ub, vn, vb) -> list[float]:
        """Energies of the state and works so far.

        Parameters
        ----------
        un, ub : np.ndarray
            Normal and binormal displacement at the nodes.
        vn, vb : np.ndarray
            Normal and binormal velocity at the nodes.

        Returns
        -------
        list of float
            The values in the order of ``NAMES`` (J).
        """
        h = self.h
        slope = (np.diff(un) / h) ** 2 + (np.diff(ub) / h) ** 2
        strain = _strain(un, ub, self.geom, self.vt2)
        energies = [
            0.5 * h * np.sum(vn[1:-1] ** 2 + vb[1:-1] ** 2),
            0.5 * self.vt2 * h * np.sum(slope),
            0.5 * self.vl2 * strain**2,
        ]
        return [self.scale * value for value in energies] + [
            self.scale * self.works[name] for name in NAMES[3:]
        ]
