"""Bending-moment law of a beam, as an operator on the curvature.

Two models share the same interface: a constant bending stiffness, and the
Bouc-Wen envelope whose tangent stiffness falls from ``ei_max`` at rest to
``ei_min`` once the curvature exceeds a critical value. Both expose
:meth:`Bending.moment` and :meth:`Bending.tangent`, the latter being the
``dM/dcurvature`` the Newton iterations of the static solver need, plus the
:attr:`Bending.ei_linear` the linear part of the scheme is assembled with. Use
:func:`create` to resolve one from a ``(conductor, span, model)`` triplet.

:meth:`Bending.moment` and :meth:`Bending.tangent` are envelopes in the
curvature alone, which is what a static solve needs. A time-domain solve uses
the second group -- :meth:`Bending.dynamic_moment`,
:meth:`Bending.update_eta`, :meth:`Bending.dynamic_tangent` and
:meth:`Bending.initial_eta` -- where the Bouc-Wen hysteresis is carried by a
state variable ``eta`` bounded by 1 rather than by the curvature history. The
constant law implements that group trivially, with no state, so a solver can
drive either model through the same calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from slenderpy.future.components import Conductor, Span


class BendingModel(str, Enum):
    """Bending-stiffness model used by the solvers."""

    CONSTANT = "constant"
    VARYING = "varying"


class Bending(ABC):
    """Bending-moment law, as a function of the curvature.

    Subclasses implement :meth:`moment` and :meth:`tangent` for one model, and
    pass the stiffness of the linear part of the law to this constructor.

    Parameters
    ----------
    ei_linear : float
        Stiffness of the linear part of the law, i.e. the bending stiffness the
        ``D4`` term of the scheme is assembled with. The remainder of the law is
        carried by the nonlinear residual.

    Attributes
    ----------
    ei_linear : float
        The stiffness passed at construction.
    """

    # laws carrying an internal state override this; a solver uses it to tell
    # whether a step needs an iteration at all
    hysteretic = False

    def __init__(self, ei_linear: float) -> None:
        self.ei_linear = ei_linear

    @abstractmethod
    def moment(self, curvature: np.ndarray) -> np.ndarray:
        """Bending moment at the given curvature.

        Parameters
        ----------
        curvature : np.ndarray
            Curvature, of any shape.

        Returns
        -------
        np.ndarray
            Bending moment, with the shape of ``curvature``.
        """

    @abstractmethod
    def tangent(self, curvature: np.ndarray) -> np.ndarray:
        """Tangent stiffness ``dM/dcurvature`` at the given curvature.

        Parameters
        ----------
        curvature : np.ndarray
            Curvature, of any shape.

        Returns
        -------
        np.ndarray
            Tangent stiffness, with the shape of ``curvature``.
        """

    @abstractmethod
    def dynamic_moment(self, curvature: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Bending moment of the time-domain law.

        Parameters
        ----------
        curvature : np.ndarray
            Curvature at the end of the step.
        eta : np.ndarray
            Hysteresis variable at the end of the step, from :meth:`update_eta`.

        Returns
        -------
        np.ndarray
            Bending moment, with the shape of ``curvature``.
        """

    @abstractmethod
    def update_eta(self, eta_old: np.ndarray, dchi: np.ndarray) -> np.ndarray:
        """Advance the hysteresis variable over one time step.

        Parameters
        ----------
        eta_old : np.ndarray
            Hysteresis variable at the start of the step.
        dchi : np.ndarray
            Curvature increment over the step.

        Returns
        -------
        np.ndarray
            Hysteresis variable at the end of the step.
        """

    @abstractmethod
    def dynamic_tangent(self, eta: np.ndarray, dchi: np.ndarray) -> np.ndarray:
        """Tangent stiffness of :meth:`dynamic_moment` over the step.

        Parameters
        ----------
        eta : np.ndarray
            Hysteresis variable at the end of the step.
        dchi : np.ndarray
            Curvature increment over the step.

        Returns
        -------
        np.ndarray
            ``dM/dcurvature`` over the step, with the shape of ``dchi``.
        """

    @abstractmethod
    def initial_eta(self, moment: np.ndarray, curvature: np.ndarray) -> np.ndarray:
        """Hysteresis variable matching an initial moment and curvature.

        Inverse of :meth:`dynamic_moment` in ``eta``, used to start a run from a
        known bending moment.

        Parameters
        ----------
        moment : np.ndarray
            Initial bending moment.
        curvature : np.ndarray
            Initial curvature.

        Returns
        -------
        np.ndarray
            Initial hysteresis variable.
        """


class ConstantBending(Bending):
    """Constant-stiffness law ``M = ei * curvature``, linear in the curvature.

    Parameters
    ----------
    ei : float
        Bending stiffness (N.m^2), strictly positive.

    Raises
    ------
    ValueError
        If ``ei`` is not strictly positive.
    """

    def __init__(self, ei: float) -> None:
        if not ei > 0.0:
            raise ValueError(f"ei must be > 0, got {ei}")

        super().__init__(ei)

    def moment(self, curvature: np.ndarray) -> np.ndarray:
        """Bending moment at the given curvature."""
        return self.ei_linear * curvature

    def tangent(self, curvature: np.ndarray) -> np.ndarray:
        """Tangent stiffness, ``ei`` everywhere."""
        return self.ei_linear * np.ones_like(curvature)

    def dynamic_moment(self, curvature: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Bending moment of the time-domain law, ignoring ``eta``."""
        return self.ei_linear * curvature

    def update_eta(self, eta_old: np.ndarray, dchi: np.ndarray) -> np.ndarray:
        """Hysteresis variable, left unchanged: this law has no internal state."""
        return eta_old

    def dynamic_tangent(self, eta: np.ndarray, dchi: np.ndarray) -> np.ndarray:
        """Tangent stiffness over the step, ``ei`` everywhere."""
        return self.ei_linear * np.ones_like(dchi)

    def initial_eta(self, moment: np.ndarray, curvature: np.ndarray) -> np.ndarray:
        """Hysteresis variable at rest, null for a law with no state."""
        return np.zeros_like(curvature)


class VaryingBending(Bending):
    """Bouc-Wen envelope, stiff at rest and softening with the curvature.

    The moment is ``(ei_max * chi_bar + ei_min * |c|) * (1 - exp(-|c| / chi_bar))``
    signed like the curvature, with the critical curvature
    ``chi_bar = (1 - ei_min / ei_max) * chi0``. Its tangent equals ``ei_max`` at
    zero curvature and tends to ``ei_min`` well beyond ``chi_bar``.

    Parameters
    ----------
    ei_min : float
        Bending stiffness of the fully softened conductor (N.m^2).
    ei_max : float
        Bending stiffness at rest (N.m^2), strictly greater than ``ei_min``.
    chi0 : float
        Reference curvature (1/m), strictly positive. Derived from the span as
        ``beta_flexion * tension``.

    Attributes
    ----------
    chi_bar : float
        Critical curvature separating the two stiffness regimes.
    plateau : float
        ``(ei_max - ei_min) * chi0``, the saturated hysteretic moment, i.e. the
        largest value the ``eta`` term of :meth:`dynamic_moment` can reach.

    Raises
    ------
    ValueError
        If the stiffnesses are not ordered ``0 < ei_min < ei_max``, or if
        ``chi0`` is not strictly positive. Equal stiffnesses would collapse
        ``chi_bar`` to zero and the law to ``0/0`` at zero curvature.
    """

    hysteretic = True

    def __init__(self, ei_min: float, ei_max: float, chi0: float) -> None:
        if not 0.0 < ei_min < ei_max:
            raise ValueError(
                f"varying model requires 0 < ei_min < ei_max, got "
                f"ei_min={ei_min} and ei_max={ei_max}"
            )
        if not chi0 > 0.0:
            raise ValueError(f"chi0 must be > 0, got {chi0}")

        super().__init__(ei_min)
        self.ei_min = ei_min
        self.ei_max = ei_max
        self.chi0 = chi0
        self.chi_bar = (1.0 - ei_min / ei_max) * chi0
        self.plateau = (ei_max - ei_min) * chi0

    def moment(self, curvature: np.ndarray) -> np.ndarray:
        """Bending moment at the given curvature."""
        c = np.abs(curvature)
        return (
            (self.ei_max * self.chi_bar + self.ei_min * c)
            * (1.0 - np.exp(-c / self.chi_bar))
            * np.sign(curvature)
        )

    def tangent(self, curvature: np.ndarray) -> np.ndarray:
        """Tangent stiffness ``dM/dcurvature``, an even function of the curvature."""
        c = np.abs(curvature)
        return self.ei_min + (
            self.ei_max - self.ei_min * (1.0 - c / self.chi_bar)
        ) * np.exp(-c / self.chi_bar)

    def dynamic_moment(self, curvature: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Bending moment of the time-domain law.

        Unlike :meth:`moment`, which is an envelope in the curvature alone, this
        is linear in the curvature and carries the whole hysteresis in ``eta``.
        """
        return self.ei_min * curvature + self.plateau * eta

    def update_eta(self, eta_old: np.ndarray, dchi: np.ndarray) -> np.ndarray:
        """Advance the Bouc-Wen hysteresis variable over one time step.

        Fully implicit discretisation of ``chi0 * d(eta)/dt = d(chi)/dt - 1/2 *
        (d(chi)/dt * abs(eta) + abs(d(chi)/dt) * eta)``, with ``d(chi)/dt``
        replaced by ``dchi/dt`` over the step::

            chi0*(eta - eta_old) = dchi - 1/2*(dchi*abs(eta) + abs(dchi)*eta)

        Both absolute values are known once the sign of ``eta`` is, and that sign
        is the sign of ``chi0*eta_old + dchi``, so the equation is solved in
        closed form -- no sub-iteration and no lagged ``abs(eta)``. The
        denominator is never below ``chi0``, so ``eta`` stays bounded by 1
        whatever the step size, and a reversal of ``dchi`` correctly leaves the
        hysteresis on the stiff branch.
        """
        numerator = self.chi0 * eta_old + dchi
        return numerator / (
            self.chi0 + 0.5 * dchi * (np.sign(dchi) + np.sign(numerator))
        )

    def dynamic_tangent(self, eta: np.ndarray, dchi: np.ndarray) -> np.ndarray:
        """Tangent stiffness of :meth:`dynamic_moment` over the step.

        Built from the derivative of :meth:`update_eta` with respect to ``dchi``,
        which equals ``1/chi0`` at rest -- making the tangent stiffness
        ``ei_max`` -- and decays as the hysteresis saturates, down to ``ei_min``.
        """
        branch = np.sign(dchi) + np.sign(eta)
        eta_tangent = (1.0 - 0.5 * branch * eta) / (self.chi0 + 0.5 * dchi * branch)
        return self.ei_min + self.plateau * eta_tangent

    def initial_eta(self, moment: np.ndarray, curvature: np.ndarray) -> np.ndarray:
        """Hysteresis variable matching an initial moment and curvature."""
        return (np.asarray(moment) - self.ei_min * curvature) / self.plateau


def create(
    conductor: Conductor,
    span: Span,
    model: BendingModel = BendingModel.CONSTANT,
    ei: float | None = None,
) -> Bending:
    """Resolve the bending law of a ``(conductor, span, model)`` triplet.

    Parameters
    ----------
    conductor : Conductor
        Conductor properties. The bending-stiffness fields are read according to
        ``model``.
    span : Span
        Span geometry; ``tension`` sets the reference curvature of the varying
        model, ``chi0 = beta_flexion * tension``.
    model : BendingModel, optional
        ``CONSTANT`` or ``VARYING``; accepts the enum member or its string value.
        Default ``CONSTANT``.
    ei : float or None, optional
        Constant model only: overrides the bending stiffness. When ``None``
        (default), ``conductor.ei_max`` is used.

    Returns
    -------
    Bending
        The selected law.

    Raises
    ------
    ValueError
        If the stiffness parameters required by ``model`` are not set, or if
        ``ei`` is given for the varying model, where it has no meaning.
    """
    model = BendingModel(model)

    if model is BendingModel.CONSTANT:
        ei_used = ei if ei is not None else conductor.ei_max
        if ei_used is None:
            raise ValueError(
                "constant model requires `ei` or `conductor.ei_max` to be set"
            )
        return ConstantBending(ei_used)

    if ei is not None:
        raise ValueError(
            "`ei` only applies to the constant model; the varying model reads "
            "ei_min and ei_max from the conductor"
        )
    if (
        conductor.ei_min is None
        or conductor.ei_max is None
        or conductor.beta_flexion is None
    ):
        raise ValueError(
            "varying model requires conductor.ei_min, ei_max and beta_flexion to be set"
        )

    # the reference curvature depends on the span tension
    chi0 = conductor.beta_flexion * span.tension

    return VaryingBending(conductor.ei_min, conductor.ei_max, chi0)
