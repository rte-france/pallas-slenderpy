"""Energy balance of the dynamic beam solve.

The balance is the beam equation multiplied by the velocity and integrated
over the span and over time, taken on the discrete equation the solver solves:
each row of the step multiplied by ``dy = y_new - y_old`` and summed over the
rows the scheme solves is the work of that term over the step. The physical
works take the forces at the middle of the step; ``numerical_work`` is the work
of what the HHT step adds to them, the inertia not being the kinetic energy
increment and the forces at n+1 and n being weighted by ``1 + alpha`` and
``-alpha`` rather than 1/2. It is the energy the scheme dissipates, and is zero
up to the step residual for ``alpha = 0`` (Crank-Nicolson). The kinetic,
tension and bending energies are functions of the state; the other terms are
works, accumulated step by step. With

    kinetic_energy + tension_energy + bending_energy
        + damping_work + hysteresis_work + curvature_work + numerical_work
        - external_work - boundary_work

constant up to the step residual. The definitions are detailed in the beam
page of the documentation, section "Energy balance".

Indices, on the extended grid of the solver (a ghost node beyond each
support, see [](`slenderpy.boundary_condition`)): the ``n`` nodes; ``J``, the
nodes 1 to n-2, the span nodes, where the centered second difference is
defined; ``I``, the nodes 2 to n-3, the rows the scheme solves (the others
carry the order-4 boundary conditions). The sums over ``J`` take half the
weight at the supports, its end nodes: summation by parts of the solved rows
then leaves no boundary term at a fixed clamped end, ``dy`` being zero at the
support and equal on both sides of it, nor at a fixed hinged end with no
moment.
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
    "numerical_work",
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
    alpha : float
        Alpha of the HHT scheme of the solve.
    """

    def __init__(
        self,
        mass: float,
        tension: float,
        damp: float,
        law: Bending,
        ds: float,
        dt: float,
        alpha: float,
    ) -> None:
        self.mass = mass
        self.tension = tension
        self.damp = damp
        self.ei_linear = law.ei_linear
        self.plateau = law.plateau if law.hysteretic else 0.0
        self.ds = ds
        self.dt = dt
        self.alpha = alpha
        self.works = dict.fromkeys(NAMES[3:], 0.0)
        self.tension_start = 0.0

    def _second_difference(self, a: np.ndarray) -> np.ndarray:
        """Centered second difference of ``a`` on ``J``."""
        return (a[:-2] - 2.0 * a[1:-1] + a[2:]) / self.ds**2

    @staticmethod
    def _over_span(a: np.ndarray) -> float:
        """Sum of ``a``, given on ``J``, with half the weight at the supports."""
        return np.sum(a) - 0.5 * (a[0] + a[-1])

    def _kinetic_energy(self, v: np.ndarray) -> float:
        return 0.5 * self.mass * self.ds * np.sum(v[2:-2] ** 2)

    def _tension_energy(self, y: np.ndarray) -> float:
        return 0.5 * self.tension * self.ds * np.sum((np.diff(y[1:-1]) / self.ds) ** 2)

    def start(self, y: np.ndarray) -> None:
        """Set the works to zero at the initial position ``y``."""
        self.works = dict.fromkeys(NAMES[3:], 0.0)
        self.tension_start = self._tension_energy(y)

    def step(
        self,
        y_old: np.ndarray,
        y_new: np.ndarray,
        v_old: np.ndarray,
        v_new: np.ndarray,
        a_new: np.ndarray,
        chi_old: np.ndarray,
        chi_new: np.ndarray,
        moment_old: np.ndarray,
        moment_new: np.ndarray,
        eta_old: np.ndarray,
        eta_new: np.ndarray,
        load_old: np.ndarray,
        load_new: np.ndarray,
    ) -> None:
        """Add the works of one time step.

        Only the works are accumulated here: the state energies are computed
        by ``values`` when needed.

        Parameters
        ----------
        y_old, y_new : np.ndarray
            Position at the start and at the end of the step.
        v_old, v_new : np.ndarray
            Velocity at the start and at the end of the step.
        a_new : np.ndarray
            Acceleration at the end of the step.
        chi_old, chi_new : np.ndarray
            Curvature at the start and at the end of the step.
        moment_old, moment_new : np.ndarray
            Bending moment at the start and at the end of the step.
        eta_old, eta_new : np.ndarray
            Hysteresis variable at the start and at the end of the step.
        load_old, load_new : np.ndarray
            Force at the start and at the end of the step, as the solver
            evaluates it.
        """
        ds = self.ds
        dy = y_new - y_old
        dy_solved = dy[2:-2]
        dchi = (chi_new - chi_old)[1:-1]
        moment = 0.5 * (moment_old + moment_new)
        moment_d2_dy = moment[1:-1] * self._second_difference(dy)

        works = self.works
        load = load_old + load_new
        works["external_work"] += 0.5 * ds * np.sum(load[2:-2] * dy_solved)
        works["damping_work"] += self.damp / self.dt * ds * np.sum(dy_solved**2)
        if self.plateau:
            works["hysteresis_work"] += (
                self.plateau
                * ds
                * self._over_span(0.5 * (eta_old + eta_new)[1:-1] * dchi)
            )
        works["curvature_work"] += ds * self._over_span(
            moment_d2_dy - moment[1:-1] * dchi
        )

        # what summation by parts leaves at the ends, for the bending and the
        # tension terms of the solved rows; the increments of the tension
        # energy that complete the tension part telescope, values() adds them
        bending = ds * self._over_span(moment_d2_dy) - ds * np.sum(
            self._second_difference(moment)[1:-1] * dy_solved
        )
        tension = (
            self.tension
            * ds
            * np.sum(self._second_difference(0.5 * (y_old + y_new))[1:-1] * dy_solved)
        )
        works["boundary_work"] += bending + tension

        # what the hht step adds to the works above: the inertia work beyond the
        # kinetic energy increment, the forces at n+1 and n weighted by
        # 1 + alpha and -alpha instead of 1/2, and the damping force at those
        # weights instead of at the mean velocity dy/dt
        alpha = self.alpha
        weight = 0.5 + alpha
        force_change = (
            self._second_difference(moment_new - moment_old)[1:-1]
            - self.tension * self._second_difference(dy)[1:-1]
        )
        damping_excess = self.damp * (
            (1.0 + alpha) * v_new - alpha * v_old - dy / self.dt
        )
        works["numerical_work"] += (
            self.mass * ds * np.sum(a_new[2:-2] * dy_solved)
            - (self._kinetic_energy(v_new) - self._kinetic_energy(v_old))
            + ds
            * np.sum(
                (
                    weight * (force_change - (load_new - load_old)[2:-2])
                    + damping_excess[2:-2]
                )
                * dy_solved
            )
        )

    def values(self, y: np.ndarray, v: np.ndarray, chi: np.ndarray) -> list[float]:
        """Energies of the state ``(y, v, chi)`` and works so far.

        Parameters
        ----------
        y : np.ndarray
            Position at the nodes.
        v : np.ndarray
            Velocity at the nodes.
        chi : np.ndarray
            Curvature at the nodes.

        Returns
        -------
        list of float
            The values in the order of ``NAMES`` (J).
        """
        ds = self.ds
        tension = self._tension_energy(y)
        works = dict(self.works)
        works["boundary_work"] += tension - self.tension_start
        return [
            self._kinetic_energy(v),
            tension,
            0.5 * self.ei_linear * ds * self._over_span(chi[1:-1] ** 2),
        ] + [works[name] for name in NAMES[3:]]
