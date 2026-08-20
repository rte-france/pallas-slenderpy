"""Bending-moment law of a beam, as an operator on the curvature.

Two models share the same interface: a constant bending stiffness, and the
Bouc-Wen envelope whose tangent stiffness falls from ``ei_max`` at rest to
``ei_min`` once the curvature exceeds a critical value. Both expose
:meth:`Bending.moment` and :meth:`Bending.tangent`, the latter being the
``dM/dcurvature`` the Newton iterations of the static solver need, plus the
:attr:`Bending.ei_linear` the linear part of the scheme is assembled with. Use
:func:`create` to resolve one from a ``(conductor, span, model)`` triplet.

The laws here are envelopes in the curvature alone, which is what a static
solve needs. The dynamic Bouc-Wen law carries its hysteresis in a state
variable ``eta`` instead, so it belongs on :class:`VaryingBending` as a second
pair of methods sharing the same ``chi0``, not in a separate module.
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

    Raises
    ------
    ValueError
        If the stiffnesses are not ordered ``0 < ei_min < ei_max``, or if
        ``chi0`` is not strictly positive. Equal stiffnesses would collapse
        ``chi_bar`` to zero and the law to ``0/0`` at zero curvature.
    """

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
