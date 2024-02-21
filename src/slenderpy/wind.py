"""Forces from wind."""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Union, Optional
import numpy as np


def air_volumic_mass(T: float = 293.15,
                     p: float = 1.013E+05,
                     phi: float = 0.) -> float:
    """Compute air volumic mass.

    Parameters
    ----------
    T : float, optional
        Temperature (K). The default is 293.15.
    p : float, optional
        Pressure (Pa). The default is 1.013E+05.
    phi : float, optional
        Relative humidity (in [0, 1] range). The default is 0.

    Returns
    -------
    float
        Air volumic mass (kg/m**3).

    """
    return 1. / (287.06 * T) * (
        p - 230.617 * phi * np.exp(17.5043 * (T - 273.15) / (T - 31.95)))


def air_density(T: float = 293.15,
                p: float = 1.013E+05,
                phi: float = 0.) -> float:
    """Compute air density.

    Density is the ratio of a given volumic mass over the volumic mass of
    water.

    Parameters
    ----------
    T : float, optional
        Temperature (K). The default is 293.15.
    p : float, optional
        Pressure (Pa). The default is 1.013E+05.
    phi : float, optional
        Relative humidity (in [0, 1] range). The default is 0.

    Returns
    -------
    float
        Air density.

    """
    return 1.0E-03 * air_volumic_mass(T, p, phi)


def air_dynamic_viscosity(T: Union[float, np.ndarray] = 293.15)\
        -> Union[float, np.ndarray]:
    r"""Compute air dynamic viscosity.

    Parameters
    ----------
    T : float or numpy.ndarray
        Air temperature (K)

    Returns
    -------
    float or numpy.ndarray
         Dynamic viscosity in kg.m\ :sup:`-1`\ .s\ :sup:`-1`\ .

    """
    return 8.8848E-15 * T**3 - 3.2398E-11 * T**2 + 6.2657E-08 * T + 2.3543E-06


def air_kinematic_viscosity(T: Union[float, np.ndarray] = 293.15,
                            p: float = 1.013E+05,
                            phi: float = 0.) -> Union[float, np.ndarray]:
    r"""Compute air kinematic viscosity.

    Parameters
    ----------
    T : float or numpy.ndarray
        Air temperature (in Celsius)

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    float or numpy.ndarray
         Kinematic viscosity in m\ :sup:`2`\ .s\ :sup:`-1`\ .
    """
    return air_dynamic_viscosity(T) / air_volumic_mass(T, p, phi)


def cylinder_drag(re: float) -> float:
    """Get cylinder drag coefficient.

    From https://kdusling.github.io/teaching/Applied-Fluids/DragCoefficient.html

    Parameters
    ----------
    re : float
        Reynolds number. Valid for re < 2E+05.

    Returns
    -------
    float
        Drag coefficient.
    """
    return (11. * np.power(re, -0.75) + 0.9 * (1.0 - np.exp(-1000. / re)) +
            1.2 * (1.0 - np.exp(-np.power(re / 4500., 0.7))))


class BishopHassanBase:
    """Base class for Bishop & Hassan models.

    Models for wind force over a circular cylinder (constant wind speed) based
    on wind tunnel experiments.
    """

    def __init__(self,
                 u: float,
                 st: float,
                 cd: float,
                 cd0: float,
                 cl: float,
                 cl0: float,
                 d: float,
                 rho: Optional[float] = None) -> None:
        """Init with args.

        Parameters
        ----------
        u : float
            Wind speed (m/s).
        st : float
            Strouhal number.
        cd : float
            Mean drag coefficient.
        cd0 : float
            Drag amplitudes.
        cl : float
            Mean lift coefficient.
        cl0 : float
            Lift amplitudes.
        d : float
            Cylinder diameter.
        rho : float, optional
            Air volumic mass. The default is None.

        Returns
        -------
        None.
        """
        self.u = u
        self.st = st
        self.cd = cd
        self.cd0 = cd0
        self.cl = cl
        self.cl0 = cl0
        if rho is None:
            rho = air_volumic_mass()
        self.wfc = 0.5 * rho * d * self.u * np.abs(self.u)
        fl = self.st / d * u
        fd = 2. * fl
        self.omd = 2. * np.pi * fd
        self.oml = 2. * np.pi * fl

    def _force_cst(self, s, t):
        wfl = self.wfc * (self.cl + self.cl0 * np.sin(self.oml * t)) * np.ones_like(s)
        wfd = self.wfc * (self.cd + self.cd0 * np.sin(self.omd * t)) * np.ones_like(s)
        return wfl, wfd

    def _force_rel(self, s, t, vn, vb):
        al = self.wfc / (self.u * np.abs(self.u))
        sq = np.sqrt((0. - vn)**2 + (self.u - vb)**2)
        cl = self.cl + self.cl0 * np.sin(self.oml * t)
        cd = self.cd + self.cd0 * np.sin(self.omd * t)
        fn = al * sq * ((0. - vn) * cd + (self.u - vb) * cl)
        fb = al * sq * ((self.u - vb) * cd - (0. - vn) * cl)
        return fn, fb


class BHCst(BishopHassanBase):
    """Bishop and Hassan force for a fixed object."""

    def __init__(self,
                 u: Optional[float] = None,
                 st: Optional[float] = None,
                 cd: Optional[float] = None,
                 cd0: Optional[float] = None,
                 cl: Optional[float] = None,
                 cl0: Optional[float] = None,
                 d: Optional[float] = None,
                 rho: Optional[float] = None) -> None:
        """Init with args.

        Parameters
        ----------
        u : float
            Wind speed (m/s).
        st : float
            Strouhal number.
        cd : float
            Mean drag coefficient.
        cd0 : float
            Drag amplitudes.
        cl : float
            Mean lift coefficient.
        cl0 : float
            Lift amplitudes.
        d : float
            Cylinder diameter.
        rho : float, optional
            Air volumic mass. The default is None.

        Returns
        -------
        None.
        """
        super().__init__(u, st, cd, cd0, cl, cl0, d, rho=rho)

    def __call__(self, s, t, un, ub, vn, vb):
        """Compute force."""
        return super()._force_cst(s, t)


class BHRel(BishopHassanBase):
    """Bishop and Hassan force with relative speed."""

    def __init__(self,
                 u: Optional[float] = None,
                 st: Optional[float] = None,
                 cd: Optional[float] = None,
                 cd0: Optional[float] = None,
                 cl: Optional[float] = None,
                 cl0: Optional[float] = None,
                 d: Optional[float] = None,
                 rho: Optional[float] = None) -> None:
        """Init with args.

        Parameters
        ----------
        u : float
            Wind speed (m/s).
        st : float
            Strouhal number.
        cd : float
            Mean drag coefficient.
        cd0 : float
            Drag amplitudes.
        cl : float
            Mean lift coefficient.
        cl0 : float
            Lift amplitudes.
        d : float
            Cylinder diameter.
        rho : float, optional
            Air volumic mass. The default is None.

        Returns
        -------
        None.
        """
        super().__init__(u, st, cd, cd0, cl, cl0, d, rho=rho)

    def __call__(self, s, t, un, ub, vn, vb):
        """Compute force."""
        return super()._force_rel(s, t, vn, vb)


class AutoDrag:
    def __init__(self,
                 u: float,
                 d: float,
                 T: float = 293.15,
                 p: float = 1.013E+05,
                 phi: float = 0.) -> None:
        """Init with args.

        Parameters
        ----------
        u : float
            Wind speed (m/s).
        d : float
            Cylinder diameter.
        T : float, optional
            Temperature (K). The default is 293.15.
        p : float, optional
            Pressure (Pa). The default is 1.013E+05.
        phi : float, optional
            Relative humidity (in [0, 1] range). The default is 0.

        Returns
        -------
        None.
        """
        self.u = u
        vm = air_volumic_mass(T, p, phi)
        self.al = d / air_kinematic_viscosity(T, p, phi)
        self.bt = 0.5 * vm * d

    def __call__(self, s, t, un, ub, vn, vb):
        rn = 0. - vn
        rb = self.u - vb
        ws = np.sqrt(rn**2 + rb**2)
        re = ws * self.al
        gm = self.bt * cylinder_drag(re) * ws
        fn = gm * rn
        fb = gm * rb
        return fn, fb
