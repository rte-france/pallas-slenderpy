"""Energy balance of the dynamic beam solve.

The balance is the beam equation multiplied by the velocity and integrated
over the span and over time, taken on the discrete equation the solver solves.
A Crank-Nicolson step gives ``y_new - y_old = dt * v_mid``, so each row of the
step multiplied by ``dy = y_new - y_old`` and summed over the rows the scheme
solves is the work of that term over the step. The kinetic, tension and
bending energies are functions of the state; the other terms are works,
accumulated step by step. With

    kinetic_energy + tension_energy + bending_energy
        + damping_work + hysteresis_work + curvature_work
        - external_work - boundary_work

constant up to the step residual. The definitions are detailed in the beam
page of the documentation, section "Energy balance".

Indices: the ``n`` nodes; ``J``, the nodes 1 to n-2, where the centered second
difference is defined; ``I``, the nodes 2 to n-3, the rows the scheme solves
(the others carry the order-4 boundary conditions).
"""

from __future__ import annotations

import numpy as np

from slenderpy.beam.bending import Bending

NAMES = (
    "kinetic_energy",
    "tension_energy",
    "bending_energy",
    "external_work",
    "damping_work",
    "hysteresis_work",
    "curvature_work",
    "boundary_work",
)


class EnergyBalance:
    """Energies and cumulated works of a dynamic beam solve.

    Parameters
    ----------
    mass : float
        Mass per unit length (kg/m).
    tension : float
        Tension (N).
    damp : float
        Damping coefficient of the scheme (kg/m/s).
    law : Bending
        Bending law of the solve.
    ds : float
        Space step (m).
    dt : float
        Time step (s).
    """

    def __init__(
        self,
        mass: float,
        tension: float,
        damp: float,
        law: Bending,
        ds: float,
        dt: float,
    ) -> None:
        self.mass = mass
        self.tension = tension
        self.damp = damp
        self.ei_linear = law.ei_linear
        self.plateau = law.plateau if law.hysteretic else 0.0
        self.ds = ds
        self.dt = dt
        self.energies = dict.fromkeys(NAMES, 0.0)

    def _second_difference(self, a: np.ndarray) -> np.ndarray:
        """Centered second difference of ``a`` on ``J``."""
        return (a[:-2] - 2.0 * a[1:-1] + a[2:]) / self.ds**2

    def _state(self, y: np.ndarray, v: np.ndarray, chi: np.ndarray) -> None:
        """Energies that are functions of the state."""
        ds = self.ds
        slope = np.diff(y) / ds
        self.energies["kinetic_energy"] = 0.5 * self.mass * ds * np.sum(v[2:-2] ** 2)
        self.energies["tension_energy"] = 0.5 * self.tension * ds * np.sum(slope**2)
        self.energies["bending_energy"] = (
            0.5 * self.ei_linear * ds * np.sum(chi[1:-1] ** 2)
        )

    def start(self, y: np.ndarray, v: np.ndarray, chi: np.ndarray) -> None:
        """Energies of the initial state, works set to zero.

        Parameters
        ----------
        y : np.ndarray
            Initial position at the nodes.
        v : np.ndarray
            Initial velocity at the nodes.
        chi : np.ndarray
            Initial curvature at the nodes.
        """
        self.energies = dict.fromkeys(NAMES, 0.0)
        self._state(y, v, chi)

    def step(
        self,
        y_old: np.ndarray,
        y_new: np.ndarray,
        v_new: np.ndarray,
        chi_old: np.ndarray,
        chi_new: np.ndarray,
        moment_old: np.ndarray,
        moment_new: np.ndarray,
        eta_old: np.ndarray,
        eta_new: np.ndarray,
        load: np.ndarray,
    ) -> None:
        """Add the works of one time step and update the state energies.

        Parameters
        ----------
        y_old, y_new : np.ndarray
            Position at the start and at the end of the step.
        v_new : np.ndarray
            Velocity at the end of the step.
        chi_old, chi_new : np.ndarray
            Curvature at the start and at the end of the step.
        moment_old, moment_new : np.ndarray
            Bending moment at the start and at the end of the step.
        eta_old, eta_new : np.ndarray
            Hysteresis variable at the start and at the end of the step.
        load : np.ndarray
            Sum of the forces at the start and at the end of the step, as the
            solver assembles it.
        """
        ds = self.ds
        dy = y_new - y_old
        dchi = (chi_new - chi_old)[1:-1]
        moment = 0.5 * (moment_old + moment_new)
        d2_dy = self._second_difference(dy)
        tension_old = self.energies["tension_energy"]

        works = self.energies
        works["external_work"] += 0.5 * ds * np.sum(load[2:-2] * dy[2:-2])
        works["damping_work"] += self.damp / self.dt * ds * np.sum(dy[2:-2] ** 2)
        works["hysteresis_work"] += (
            self.plateau * ds * np.sum(0.5 * (eta_old + eta_new)[1:-1] * dchi)
        )
        works["curvature_work"] += ds * np.sum(moment[1:-1] * (d2_dy - dchi))

        # what summation by parts leaves at the ends, for the bending and the
        # tension terms of the solved rows
        self._state(y_new, v_new, chi_new)
        bending = ds * np.sum(moment[1:-1] * d2_dy) - ds * np.sum(
            self._second_difference(moment)[1:-1] * dy[2:-2]
        )
        tension = self.tension * ds * np.sum(
            self._second_difference(0.5 * (y_old + y_new))[1:-1] * dy[2:-2]
        ) + (self.energies["tension_energy"] - tension_old)
        works["boundary_work"] += bending + tension

    def values(self) -> list[float]:
        """Current energies and works, in the order of ``NAMES``."""
        return [self.energies[name] for name in NAMES]
