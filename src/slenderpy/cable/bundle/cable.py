"""Single-conductor dynamic cable model.

This module provides ``Cable``, a self-contained data container for the
parameters of a single suspended conductor under tension. It exposes the
non-dimensional wave-speed parameters used by the bundle solver and a
single-cable convenience entry point ``Cable.solve`` that delegates to a
degenerate :class:`~slenderpy.cable.bundle.bundle.Bundle` with one
sub-conductor and no spacers.

The physics and the Newmark beta=1/4 integration scheme match the legacy
solver in :mod:`slenderpy.cable`, but the implementation is independent:
this module does not import any private helpers from :mod:`slenderpy.cable`
or :mod:`slenderpy._cable_utils`. The public ``fdm_utils`` finite-difference
operators are shared because they are stable.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, TYPE_CHECKING

import numpy as np

from slenderpy import floatArray
from slenderpy._constant import _GRAVITY
from slenderpy.cable.static import catenary

if TYPE_CHECKING:
    from slenderpy import simulation


class Cable:
    """Suspended conductor under axial tension.

    Parameters
    ----------
    mass : float
        Linear mass (kg/m).
    diameter : float
        Conductor diameter (m). Kept for completeness; aerodynamic models
        provided by the user may depend on it.
    EA : float
        Axial stiffness (N).
    length : float
        Span length (m).
    tension : float
        Mechanical tension at the support (N).
    h : float, optional
        Height difference between supports (m). Default 0.
    g : float, optional
        Gravitational acceleration (m/s**2). Default 9.81.
    """

    def __init__(
        self,
        mass: float,
        diameter: float,
        EA: float,
        length: float,
        tension: float,
        h: float = 0.0,
        g: float = _GRAVITY,
    ) -> None:
        if mass <= 0.0:
            raise ValueError("mass must be positive")
        if diameter <= 0.0:
            raise ValueError("diameter must be positive")
        if EA <= 0.0:
            raise ValueError("EA must be positive")
        if length <= 0.0:
            raise ValueError("length must be positive")
        if tension <= 0.0:
            raise ValueError("tension must be positive")
        if g <= 0.0:
            raise ValueError("g must be positive")

        self.mass = mass
        self.diameter = diameter
        self.EA = EA
        self.length = length
        self.tension = tension
        self.h = h
        self.g = g

    @property
    def tAd(self) -> float:
        """Characteristic time scale used for non-dimensionalisation."""
        return float(np.sqrt(self.length / self.g))

    @property
    def uAd(self) -> float:
        """Characteristic velocity scale used for non-dimensionalisation."""
        return self.length / self.tAd

    @property
    def vt2(self) -> float:
        """Non-dimensional transverse wave-speed squared (vt2 = H/(m*g*L))."""
        return self.tension / (self.mass * self.g * self.length)

    @property
    def vl2(self) -> float:
        """Non-dimensional longitudinal wave-speed squared (vl2 = EA/(m*g*L))."""
        return self.EA / (self.mass * self.g * self.length)






    ######## AJOUTER MASSE ENTRETOISE ????? ################

    def static_shape(self, s: floatArray) -> floatArray:
        """Catenary rest shape at curvilinear span abscissae ``s`` in [0, 1].

        Convenience wrapper over :func:`slenderpy.cable.static.catenary.shape`.

        Parameters
        ----------
        s : numpy.ndarray
            Normalized curvilinear positions in [0, 1].

        Returns
        -------
        numpy.ndarray
            Static vertical displacement (m) at each ``s``, relative to the
            straight chord between supports.
        """
        x = s * self.length
        return catenary.shape(x, self.length, self.tension, self.h, self.mass, g=self.g)







    def solve(
        self,
        parameters: "simulation.Parameters",
        force: Optional[Callable] = None,
        zt: float = 0.0,
        un0: Optional[floatArray] = None,
        ub0: Optional[floatArray] = None,
        vn0: Optional[floatArray] = None,
        vb0: Optional[floatArray] = None,
    ) -> "simulation.Results":
        """Run a single-conductor dynamic simulation.

        Internally builds a degenerate :class:`Bundle` with this cable as
        the only sub-conductor and no spacers, then delegates to its
        ``solve`` method. Provided as a convenience so that single-cable
        users do not need to construct a bundle by hand; the legacy
        :func:`slenderpy.cable.solve` remains the canonical legacy API.

        Parameters
        ----------
        parameters : slenderpy.simulation.Parameters
            Simulation parameters (time stepping, output grid).
        force : callable, optional
            External force callback with the same signature as the legacy
            cable solver: ``force(s, t, un, ub, vn, vb) -> (fn, fb)`` where
            ``s`` is the spatial grid in [0, 1] and the returned arrays
            have the same shape as ``s``. Default: zero force.
        zt : float, optional
            Modal viscous damping ratio. Default 0.
        un0, ub0, vn0, vb0 : numpy.ndarray, optional
            Initial conditions for normal/binormal displacements and
            velocities. Default: zero (cable at rest at the static
            catenary equilibrium).

        Returns
        -------
        slenderpy.simulation.Results
            Results object with per-cable displacement and force traces.
        """
        from slenderpy.cable.bundle.bundle import Bundle, InitialConditions

        ic: Optional[InitialConditions]
        if un0 is None and ub0 is None and vn0 is None and vb0 is None:
            ic = None
        else:
            ic = InitialConditions(
                un=None if un0 is None else [un0],
                ub=None if ub0 is None else [ub0],
                vn=None if vn0 is None else [vn0],
                vb=None if vb0 is None else [vb0],
            )

        bundle = Bundle(cables=[self], spacers=[])
        return bundle.solve(
            parameters=parameters,
            force=_wrap_single_force(force) if force is not None else None,
            zt=zt,
            initial_conditions=ic,
        )


def _wrap_single_force(force: Callable) -> Callable:
    """Adapt a single-cable force callback to the bundle force signature.

    The bundle force callback takes per-cable state lists and returns
    per-cable force tuples; for the single-cable case we wrap a scalar
    callback so users don't need to think about lists.
    """

    def wrapped(
        s: floatArray,
        t: float,
        un_list: list,
        ub_list: list,
        vn_list: list,
        vb_list: list,
    ) -> Tuple[list, list]:
        fn, fb = force(s, t, un_list[0], ub_list[0], vn_list[0], vb_list[0])
        return [fn], [fb]

    return wrapped
