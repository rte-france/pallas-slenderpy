"""Cable object and associated solvers."""

import io
import os
from typing import Tuple, Union, Callable, Optional

import numpy as np
import scipy as sp
from scipy.linalg import solve_banded
from slenderpy import _cable_utils as cbu
from slenderpy import _progress_bar as spb
from slenderpy import simtools


def _newt2dv(f1, f2, x0, y0, epsilon=1.0E-12, maxiter=64, dx=1.0E-03, dy=1.0E-03):
    """2D quasi-Newton with arrays."""
    q = 1.
    c = 0
    x = x0
    y = y0
    while q > epsilon and c < maxiter:
        F1 = f1(x, y)
        F2 = f2(x, y)
        Ja = 0.5 * (f1(x + dx, y) - f1(x - dx, y)) / dx
        Jb = 0.5 * (f1(x, y + dy) - f1(x, y - dy)) / dy
        Jc = 0.5 * (f2(x + dx, y) - f2(x - dx, y)) / dx
        Jd = 0.5 * (f2(x, y + dy) - f2(x, y - dy)) / dy
        di = 1. / (Ja * Jd - Jb * Jc)
        ex = di * (Jd * F1 - Jb * F2)
        ey = di * (Ja * F2 - Jc * F1)
        x -= ex
        y -= ey
        q = max(np.nanmax(np.abs(ex / x)), np.nanmax(np.abs(ey / y)))
        c += 1
    return x, y, c, np.maximum(np.abs(ex / x), np.abs(ey / y))


def _solve_p3(c1, c2, c3):
    """Compute real solution of : x**3 + c1 x**2 + c2 x + c3 = 0."""
    p = c2 - c1**2 / 3.
    q = c3 - c1 * c2 / 3. + 2. * c1**3 / 27.
    D = 4. * p**3 + 27. * q**2
    tp1 = (-q + np.sqrt(D / 27. + 0J)) / 2.
    tp2 = (-q - np.sqrt(D / 27. + 0J)) / 2.
    return np.real(np.power(tp1, 1 / 3) + np.power(tp2, 1 / 3) - c1 / 3.)


def __alts(Lp, a, x, s):
    return 2. * a * np.sinh(0.5 * (Lp * s + x) / a) * np.sinh(0.5 * Lp * s / a)


def _q_factor(L, h):
    return np.log((L + h) / (L - h))


def catenary_length(Lp: Union[float, np.ndarray],
                    a: Union[float, np.ndarray],
                    h: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute cable length.

    Parameters
    ----------
    Lp : float or numpy.ndarray
        Span length (m).
    a : float or numpy.ndarray
        Mechanical parameter or catenary constant (m).
    h : float or numpy.ndarray
        Height difference between anchors (m).

    Returns
    -------
    float or numpy.ndarray (depending on input)
        Cable length when supporting its own weight (m).

    """
    return np.sqrt(h**2 + (2.0 * a * np.sinh(0.5 * Lp / a))**2)


def irvine_number(Lp: Union[float, np.ndarray],
                  a: Union[float, np.ndarray],
                  h: Union[float, np.ndarray],
                  m: float,
                  EA: float,
                  g: float = 9.81) -> Union[float, np.ndarray]:
    """Compute Irvine number.

    Parameters
    ----------
    Lp : float or numpy.ndarray
        Span length (m).
    a : float or numpy.ndarray
        Mechanical parameter or catenary constant (m).
    h : float or numpy.ndarray
        Height difference between anchors (m).
    m : float
        Mass per unit length (kg/m).
    EA : float
        Axial stiffness (N).
    g : float, optional
        Gravitational acceleration. The default is 9.81.

    Returns
    -------
    float or numpy.ndarray (depending on input)
        Irvine number.

    """
    L = catenary_length(Lp, a, h)
    return np.sqrt((Lp / a)**2 * (Lp * EA / (L * a * m * g)))


def natural_frequency(L: Union[float, np.ndarray],
                      a: Union[float, np.ndarray],
                      g: float = 9.81) -> Union[float, np.ndarray]:
    """Compute string natural frequency.

    Parameters
    ----------
    L : float or numpy.ndarray
        Cable length (m).
    a : float or numpy.ndarray
        Mechanical parameter or catenary constant (m).
    g : float, optional
        Gravitational acceleration. The default is 9.81.

    Returns
    -------
    float or numpy.ndarray (depending on input)
        Natural frequency (Hz).

    """
    return 0.5 * np.sqrt(a * g) / L


def _alts(s, Lp, a, h):
    L = catenary_length(Lp, a, h)
    q = _q_factor(L, h)
    x = a * q - Lp
    return __alts(Lp, a, x, s)


def argsag(Lp: Union[float, np.ndarray],
        a: Union[float, np.ndarray],
        h: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Find curvilinear abscissa where sag occurs.

    Parameters
    ----------
    Lp : float or numpy.ndarray
        Span length (m).
    a : float or numpy.ndarray
        Mechanical parameter or catenary constant (m).
    h : float or numpy.ndarray
        Height difference between anchors (m).

    Returns
    -------
    float or numpy.ndarray (depending on input)
        Sag (m).

    """
    L = catenary_length(Lp, a, h)
    q = _q_factor(L, h)
    x = 0.5 * (a * q - Lp)
    return (a * np.arcsinh(h / Lp) - x) / Lp


def sag(Lp: Union[float, np.ndarray],
        a: Union[float, np.ndarray],
        h: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute suspended cable sag.

    Parameters
    ----------
    Lp : float or numpy.ndarray
        Span length (m).
    a : float or numpy.ndarray
        Mechanical parameter or catenary constant (m).
    h : float or numpy.ndarray
        Height difference between anchors (m).

    Returns
    -------
    float or numpy.ndarray (depending on input)
        Sag (m).

    """
    L = catenary_length(Lp, a, h)
    q = _q_factor(L, h)
    x = 0.5 * (a * q - Lp)
    s = argsag(Lp, a, h)
    return s * h - __alts(Lp, a, 2.0 * x, s)


def _blondel(m, EA, al, L, H, dT, g=9.81):
    c1 = +EA * (al * dT + (m * g * L)**2 / (24. * H**2)) - H
    c3 = -EA * (m * g * L)**2 / 24.
    return _solve_p3(c1, 0., c3)


def tension_corr_temperature(m: Union[float, np.ndarray],
                             EA: Union[float, np.ndarray],
                             al: Union[float, np.ndarray],
                             Lp: Union[float, np.ndarray],
                             H: Union[float, np.ndarray],
                             h: Union[float, np.ndarray],
                             dT: Union[float, np.ndarray],
                             g: float = 9.81,
                             epsilon: float = 1.0E-12,
                             maxiter: int = 16) \
        -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Evaluate variation of tension with temperature shift.

    Parameters
    ----------
    m : float or numpy.ndarray
        Mass per unit length (kg/m).
    EA : float or numpy.ndarray
        Axial stiffness (N).
    al : float or numpy.ndarray
        Thermal dilatation coefficient (K**-1)
    Lp : float or numpy.ndarray
        Span length (m).
    H : float or numpy.ndarray
        Tension at reference temperature (N).
    h : float or numpy.ndarray
        Height difference (m) between anchors.
    dT : float or numpy.ndarray
        Difference with reference temperature (K).
    g : float, optional
        Gravitational acceleration. The default is 9.81.
    epsilon : float, optional
        Relative error used in iterative schemes. The default is 1.0E-12
    maxiter : int, optional
        Maximum number of iterations in iterative schemes. The default is 16

    Returns
    -------
    float or numpy.ndarray (depending on input)
        New catenary constant at shifted temperature (m).
    float or numpy.ndarray (depending on input)
        Maximum error on iterative schemes.
    """
    w = m * g
    Lr = np.sqrt(Lp**2 + h**2)  # straight length
    cv = Lp / Lr  # cos(vv)
    sv = -h / Lr  # sin(vv)

    def phi1(s, v, h, dT):
        return h * (s / EA + (1. + al * dT) * (np.arcsinh(v / h) - np.arcsinh((v - w * s) / h)) / w)

    def phi2(s, v, h, dT):
        return (v * s - 0.5 * w * s**2) / EA + h / w * (1. + al * dT) * (
                np.sqrt(1. + (v / h)**2) - np.sqrt(1. + ((v - w * s) / h)**2))

    # (1) find length

    l_ = Lr
    v_ = 0.5 * w * l_

    def f1(v, l):
        return phi1(l, v, H, 0.) - Lp

    def f2(v, l):
        return phi2(l, v, H, 0.) - l * sv / cv

    v_, l_, _, qa = _newt2dv(f1, f2, v_, l_, epsilon=epsilon, maxiter=maxiter)

    # (2) find tension

    L0 = l_  # initial length

    def g1(v, h):
        return phi1(L0, v, h, dT) - Lp

    def g2(v, h):
        return phi2(L0, v, h, dT) - L0 * sv / cv

    h_ = _blondel(m, EA, al, L0, H, dT, g=g)
    v_, h_, _, qb = _newt2dv(g1, g2, v_, h_, epsilon=epsilon, maxiter=maxiter)

    return h_, np.maximum(qa, qb)


def catconst_corr_temperature(m: Union[float, np.ndarray],
                              EA: Union[float, np.ndarray],
                              al: Union[float, np.ndarray],
                              Lp: Union[float, np.ndarray],
                              A: Union[float, np.ndarray],
                              h: Union[float, np.ndarray],
                              dT: Union[float, np.ndarray],
                              g: float = 9.81,
                              epsilon: float = 1.0E-12,
                              maxiter: int = 16) \
        -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Evaluate variation of catenary constant with temperature shift.

    Parameters
    ----------
    m : float or numpy.ndarray
        Mass per unit length (kg/m).
    EA : float or numpy.ndarray
        Axial stiffness (N).
    al : float or numpy.ndarray
        Thermal dilatation coefficient (K**-1)
    Lp : float or numpy.ndarray
        Span length (m).
    A : float or numpy.ndarray
        Catenary constant or mechanical parameter at reference temperature (m).
    h : float or numpy.ndarray
        Height difference (m) between anchors.
    dT : float or numpy.ndarray
        Difference with reference temperature (K).
    g : float, optional
        Gravitational acceleration. The default is 9.81.
    epsilon : float, optional
        Relative error used in iterative schemes. The default is 1.0E-12
    maxiter : int, optional
        Maximum number of iterations in iterative schemes. The default is 16

    Returns
    -------
    float or numpy.ndarray (depending on input)
        New catenary constant at shifted temperature (m).
    float or numpy.ndarray (depending on input)
        Maximum error on iterative schemes.
    """
    H = m * g * A
    h_, e = tension_corr_temperature(m, EA, al, Lp, H, h, dT, g=g, epsilon=epsilon, maxiter=maxiter)
    return h_ / (m * g), e


class SCable:
    """A suspended cable."""

    def __init__(self,
                 mass: Optional[float] = None,
                 diameter: Optional[float] = None,
                 EA: Optional[float] = None,
                 length: Optional[float] = None,
                 tension: Optional[float] = None,
                 h: float = 0.,
                 g: float = 9.81) -> None:
        """Init with args.

        Parameters
        ----------
        mass : float
            Mass per unit length (kg/m).
        diameter : float
            Cable diameter (m). The default is None.
        EA : float
            Axial stiffness (N). The default is None.
        length : float
            Span length (m).
        tension : float
            Span tension (N).
        h : float, optional
            Height difference (m) between anchors. The default is 0.
        g : float, optional
            Gravitational acceleration. The default is 9.81.

        Returns
        -------
        None.
        """
        vrl = [mass, diameter, EA, length, tension, h, g]
        vrn = ['mass', 'diameter', 'EA', 'length', 'tension', 'h', 'g']
        pos = [True, True, True, True, True, False, False]
        for i in range(len(vrl)):
            if not isinstance(vrl[i], float):
                raise TypeError(f'input {vrn[i]} must be a float')
            if vrl[i] < 0. and pos[i]:
                raise ValueError(f'input {vrn[i]} must be positive')

        self.g = g  # gravity (m/s**2)
        self.m = mass  # mass per length unit (kg/m)
        self.d = diameter  # conductor diameter (m)
        self.EA = EA  # axial stiffness (N)
        self.Lp = length  # span length (m)
        self.H = tension  # tension (N)
        self.h = h  # pole altitude difference (2nd minus 1st, m)

        # mechanical parameter (m)
        self.a = self.H / (self.m * self.g)
        # cable length at rest (gravity only) (m)
        self.L = catenary_length(self.Lp, self.a, self.h)
        # Irvine number (no unit)
        self.lm = irvine_number(self.Lp, self.a, self.h, self.m, self.EA, g=self.g)
        # useful value (no unit)
        self.q = _q_factor(self.L, self.h)

    def natural_frequencies(self, n: int = 1) -> np.ndarray:
        """Compute the n first modes of the vibrating string.

        Parameters
        ----------
        n : int, optional
            Number of modes to compute. The default is 1.

        Returns
        -------
        numpy.ndarray
            Array of frequencies (in Hz).
        """
        return np.linspace(1, n, n) * natural_frequency(self.L, self.a, self.g)

    def natural_frequency(self):
        """Compute the natural frequency of the vibrating string."""
        return self.natural_frequencies(n=1)[0]

    def irvine_frequencies(self,
                           n: int = 10,
                           epsilon: float = 1.0E-09,
                           maxiter: int = 64) -> np.ndarray:
        """Compute Irvine frequencies.

        Solve transcendental equation in [Irvine1974].

        Parameters
        ----------
        n : int, optional
            Number of frequencies to compute. The default is 10.
        epsilon : float, optional
            Absolute error on frequency. The default is 1.0E-09.
        maxiter : int, optional
            Max number of iterations. The default is 64.

        Returns
        -------
        numpy.ndarray
            Array of frequencies (in Hz).
        """
        omega = np.zeros((n,))
        err = np.zeros((n,))
        for k in range(n):
            xm = (2 * k + 1) * np.pi
            xM = (2 * k + 3) * np.pi
            count = 1
            while (xM - xm) > epsilon and count <= maxiter:
                x = 0.5 * (xm + xM)
                y = np.tan(0.5 * x) - 0.5 * x + 0.5 * x**3 / self.lm**2
                if y > 0:
                    xM = x
                elif y < 0:
                    xm = x
                count = count + 1
            omega[k] = 0.5 * (xm + xM)
            err[k] = xM - xm
        return self.natural_frequency() * omega / np.pi

    def normal_frequencies(self, n: int = 10) -> np.ndarray:
        """Compute in-plane natural frequencies (normal direction).

        Parameters
        ----------
        n : int, optional
            Number of frequencies to compute. The default is 10.

        Returns
        -------
        fq : numpy.ndarray
            Array of frequencies (in Hz).
        """
        fq = self.natural_frequencies(n=n)
        tf = self.irvine_frequencies((1 + n) // 2)
        fq[::2] = tf
        return fq

    def binormal_frequencies(self, n: int = 10) -> np.ndarray:
        """Compute out-of-plane natural frequencies (binormal direction).

        Parameters
        ----------
        n : int, optional
            Number of frequencies to compute. The default is 10.

        Returns
        -------
        numpy.ndarray
            Array of frequencies (in Hz).
        """
        return self.natural_frequencies(n=n)

    def altitude_1s(self, s: np.ndarray) -> np.ndarray:
        """Compute cable altitude regarding 1st pole (catenary equation).

        Parameters
        ----------
        s : numpy.ndarray
            Positions on span (curvilinear abscissa: 0 is first pole and 1 the
            second).

        Returns
        -------
        numpy.ndarray
            Array of altidudes (in meter); same shape as input s.
        """
        return _alts(s, self.Lp, self.a, self.h)

    def altitude_2s(self, s):
        """Same as altitude_1s but regarding 2nd pole."""
        return self.altitude_1s(s) - self.h

    def altitude_1c(self, s):
        """Same as altitude_1s but with curvilinear abcissa along the cable."""
        xm = 0.5 * (self.a * self.q - self.Lp)
        return self.a * (np.sqrt(1. + (self.L / self.a * s + np.sinh(xm / self.a))**2
                                 ) - np.sqrt(1. + np.sinh(xm / self.a)**2))

    def altitude_2c(self, s):
        """Same as altitude_2c but regarding 2nd pole."""
        return self.altitude_1c(s) - self.h

    def cable2span(self, s):
        """Convert curvilinear abscissa along cable to curvilinear abscissa along span."""
        xm = 0.5 * (self.a * self.q - self.Lp)
        return (self.a * np.arcsinh(self.L / self.a * s + np.sinh(xm / self.a))
                - xm) / self.Lp

    def span2cable(self, s):
        """Convert curvilinear abscissa along span to curvilinear abscissa along cable."""
        return (2. * self.a / self.L * np.sinh(0.5 * self.Lp / self.a * s) *
                np.cosh(0.5 * self.Lp / self.a * (s - 1.) + 0.5 * self.q))

    def sag(self):
        """Get sag in meters"""
        return sag(self.Lp, self.a, self.h)


def solve(cb: SCable,
          pm: simtools.Parameters,
          force: Optional[Callable[[np.ndarray, float, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray],
          Tuple[np.ndarray, np.ndarray]]] = None,
          zt: float = 0.,
          un0: Optional[np.ndarray] = None,
          ub0: Optional[np.ndarray] = None,
          vn0: Optional[np.ndarray] = None,
          vb0: Optional[np.ndarray] = None,
          remove_cat: bool = False) -> simtools.Results:
    """EOM solver for a suspended cable and an external force.

    Results are offsets regarding the equilibrium position (catenary equations)
    in a local triad. In order to have absolute offsets, outputs results must
    be projected in a fixed triad and the equilibrium position must be added.

    Parameters
    ----------
    cb : slenderpy.cable.SCable
        A cable object.
    pm : slenderpy.simtools.Parameters
        Simulation parameters.
    force : TYPE, optional
        A force object. The default is None, which will lead to no force applied.
        This object can be any object with a __call__ method and 6 arguments: s
        (array of struct. elements), t (time), yn (normal offset), yb (binormal
        offset), vn (normal speed) and vb (binormal speed).
    zt : float, optional
        Viscous damping.
    un0 : numpy.ndarray, optional
        Initial normal offset. The default is None.
    ub0 : numpy.ndarray, optional
        Initial binormal offset. The default is None.
    vn0 : numpy.ndarray, optional
        Initial normal velocity. The default is None.
    vb0 : numpy.ndarray, optional
        Initial binormal velocity. The default is None.
    remove_cat : bool, optional
        Whether or not remove catenary equation for normal offset initialization.
        The default is False.

    Returns
    -------
    slenderpy.simtools.Results
        Simulation output with offsets (tangential, normal and binormal) in
        meters and axial force in newtons for the positions and times specified
        in input parameters.
    """
    ns, s, ds, N, n = cbu.spacediscr(pm.ns)
    vt2, vl2 = cbu.vtvl(cb)
    C, A, _, J = cbu.matrix(ds, n)
    tAd, uAd = cbu.adim(cb)
    t, tf, dt, ht, ht2 = cbu.times(pm, tAd)
    tau = -1. * ht2
    un, ub, vn, vb = cbu.init_vars(cb, s, un0, ub0, vn0, vb0, uAd, remove_cat)
    ut, ef = cbu.utef(un, ub, C, s, ds, vt2)

    z = -2. * np.pi * np.sqrt(vt2) * zt * ht

    lov = ['ut', 'un', 'ub', 'ef']
    res = simtools.Results(lot=pm.time_vector_output().tolist(), lov=lov, los=pm.los)
    res.update(0, s, lov, [ut, un, ub, ef])

    if force is None:
        force = cbu.ZeroForce()

    Db = np.zeros((3, n))

    # loop
    res.start_timer()
    pb = spb.generate(pm.pp, pm.nt, desc=__name__)
    for k in range(pm.nt):

        h = -1. / vt2 * un + 0.5 * ((C * un)**2 + (C * ub)**2)
        e = 0.5 * np.sum((h[:-1] + h[1:]) * ds)
        b = vt2 + vl2 * e

        fn1, fn2, fb1, fb2 = cbu.adim_force(force, s, t, dt, un, ub, vn, vb,
                                            tAd, cb.L, uAd, cb.m, cb.g)

        Rvn = (dt * b) * A * (un[1:-1] + 0.5 * ht * vn[1:-1]) + (1 + z) * vn[1:-1] + \
              dt * (0.5 * (fn1[1:-1] + fn2[1:-1]) + vl2 / vt2 * e)
        Rvb = (dt * b) * A * (ub[1:-1] + 0.5 * ht * vb[1:-1]) + (1 + z) * vb[1:-1] + \
              ht * (fb1[1:-1] + fb2[1:-1])

        Db[0, +1:] = tau * b * A.diagonal(k=1)
        Db[1, :] = 1. - z + tau * b * A.diagonal(k=0)
        Db[2, :-1] = tau * b * A.diagonal(k=-1)

        Rhs = np.column_stack((Rvn, Rvb))
        vnb = solve_banded((1, 1), Db, Rhs)

        un[1:-1] += ht * (vn[1:-1] + vnb[:, 0])
        ub[1:-1] += ht * (vb[1:-1] + vnb[:, 1])
        vn[1:-1] = vnb[:, 0]
        vb[1:-1] = vnb[:, 1]

        t += dt

        ut, ef = cbu.utef(un, ub, C, s, ds, vt2)

        if (k + 1) % pm.rr == 0:
            res.update((k // pm.rr) + 1, s, lov, [ut, un, ub, ef])
            pb.update(pm.rr)

    # END FOR
    pb.close()
    res.stop_timer()
    res.set_state({"un": un * cb.L,
                   "ub": ub * cb.L,
                   "vn": vn * uAd,
                   "vb": vb * uAd})

    # add dim
    res.data.assign_coords({simtools.__stime__: res.data[simtools.__stime__].values * tAd})
    for v in ['ut', 'un', 'ub']:
        res.data[v] *= cb.L
    res.data['ef'] *= cb.EA

    return res


def _solve_added_mass(cb, pm, force=None, am=None, zt=0., un0=None, ub0=None,
                      vn0=None, vb0=None, remove_cat=False):
    if am is None:
        return solve(cb, pm, force=force, zt=0., un0=un0, ub0=ub0, vn0=vn0,
                     vb0=vb0, remove_cat=remove_cat)

    ns, s, ds, N, n = cbu.spacediscr(pm.ns)

    # total mass
    tm = cbu.interp_init(s, am) + cb.m
    vt2 = cb.H / (tm * cb.g * cb.L)
    vl2 = cb.EA / (tm * cb.g * cb.L)

    C, A, _, J = cbu.matrix(ds, n)
    tAd, uAd = cbu.adim(cb)
    t, tf, dt, ht, ht2 = cbu.times(pm, tAd)
    tau = -1. * ht2
    un, ub, vn, vb = cbu.init_vars(cb, s, un0, ub0, vn0, vb0, uAd, remove_cat)
    ut, ef = cbu.utef(un, ub, C, s, ds, vt2)

    z = -2. * np.pi * np.sqrt(vt2[1:-1]) * zt * ht

    lov = ['ut', 'un', 'ub', 'ef']
    res = simtools.Results(lot=pm.time_vector_output().tolist(), lov=lov, los=pm.los)
    res.update(0, s, lov, [ut, un, ub, ef])

    if force is None:
        force = cbu.ZeroForce()

    Db = np.zeros((3, n))

    # loop
    res.start_timer()
    pb = spb.generate(pm.pp, pm.nt, desc=__name__)
    for k in range(pm.nt):

        h = -1. / vt2 * un + 0.5 * ((C * un)**2 + (C * ub)**2)
        e = 0.5 * np.sum((h[:-1] + h[1:]) * ds)
        b = vt2 + vl2 * e
        c = vl2 / vt2 * e
        bA = sp.sparse.diags([b[1:-1]], [0]) * A

        fn1, fn2, fb1, fb2 = cbu.adim_force(force, s, t, dt, un, ub, vn, vb,
                                            tAd, cb.L, uAd, cb.m, cb.g)

        Rvn = (dt * bA * (un[1:-1] + 0.5 * ht * vn[1:-1]) + (1 + z) * vn[1:-1]
               + dt * (0.5 * (fn1[1:-1] + fn2[1:-1]) + c[1:-1]))
        Rvb = (dt * bA * (ub[1:-1] + 0.5 * ht * vb[1:-1]) + (1 + z) * vb[1:-1]
               + ht * (fb1[1:-1] + fb2[1:-1]))

        Db[0, +1:] = tau * bA.diagonal(k=1)
        Db[1, :] = 1. - z + tau * bA.diagonal(k=0)
        Db[2, :-1] = tau * bA.diagonal(k=-1)

        Rhs = np.column_stack((Rvn, Rvb))
        vnb = solve_banded((1, 1), Db, Rhs)

        un[1:-1] += ht * (vn[1:-1] + vnb[:, 0])
        ub[1:-1] += ht * (vb[1:-1] + vnb[:, 1])
        vn[1:-1] = vnb[:, 0]
        vb[1:-1] = vnb[:, 1]

        t += dt

        ut, ef = cbu.utef(un, ub, C, s, ds, vt2)

        if (k + 1) % pm.rr == 0:
            res.update((k // pm.rr) + 1, s, lov, [ut, un, ub, ef])
            pb.update(pm.rr)

    # END FOR
    pb.close()
    res.stop_timer()
    res.set_state({"un": un * cb.L,
                   "ub": ub * cb.L,
                   "vn": vn * uAd,
                   "vb": vb * uAd})

    # add dim
    res.data.assign_coords({simtools.__stime__: res.data[simtools.__stime__].values * tAd})
    for v in ['ut', 'un', 'ub']:
        res.data[v] *= cb.L
    res.data['ef'] *= cb.EA

    return res


def tnb2xyz(res: simtools.Results,
            cb: SCable,
            mix_curv: bool = False) -> simtools.Results:
    """Project raw outputs from a cable solver to classi fixed triad.

    Projection from local (tnb) triad to classic (xyz): ex, ez is the plane
    defined by the poles and the span at rest, ex is horizontal and ez vertical;
    ey is such that exyz is direct (thus ey = -eb).

    Parameters
    ----------
    res : slenderpy.simtools.Results
        Input results from a cable simulation. The absence of one or more keys
        among the following list will lead to undefined behaviour: [ut, un, ub].
    cb : slenderpy.cable.SCable
        The cable object used to generate the results.
    mix_curv : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    prj : TYPE
        DESCRIPTION.
    """

    s = res.data[simtools.__scabs__].values
    if not mix_curv:
        ps = cb.cable2span(s)
    else:
        ps = s

    # data out
    prj = simtools.Results(lot=res.data[simtools.__stime__].values.tolist(),
                           lov=['ux', 'uy', 'uz', 'ef'], los=ps)

    for i, st in enumerate(s):
        # et in uxyz
        tx = 1.
        tz = cb.Lp / cb.L * np.sinh(cb.Lp / cb.a * (st - 0.5) + 0.5 * cb.q)
        tn = np.sqrt(tx**2 + tz**2)
        tx /= tn
        tz /= tn

        ut = res.data['ut'].values[:, i]
        un = res.data['un'].values[:, i]
        ub = res.data['ub'].values[:, i]

        prj.data['ux'].values[:, i] = tx * ut - tz * un + 0. * ub
        prj.data['uy'].values[:, i] = 0. * ut + 0. * un - 1. * ub
        prj.data['uz'].values[:, i] = tz * ut + tx * un + 0. * ub

    prj.data['ef'].values = res.data['ef'].values

    return prj


def export_vtk(cb: SCable,
               res: simtools.Results,
               rep: str,
               file: str = 'snapshot',
               fmt: str = '%+10.3E') -> None:
    """Export results from a simulation to VTK files.

    Input results must have been converted to the (xyz) triad.

    Parameters
    ----------
    cb : slenderpy.cable.SCable
        The cable object used to generate the results
    res : slenderpy.simulation.Results
        Results to export.
    rep : str
        Directory where to write the vtk files.
    file : str, optional
        Prefix name for generated files. The default is 'snapshot'.
    fmt : str, optional
        float format in generated files. The default is '%+10.3E'.

    Raises
    ------
    ValueError
        DESCRIPTION.
    """
    if not os.path.isdir(rep):
        raise ValueError('input rep must be a directory')

    sep = ' '
    enc = 'UTF-8'
    aos = res.data[simtools.__scabs__]

    def nda2str(dat, delimiter=sep, fmt=fmt, encoding=enc):
        s = io.BytesIO()
        np.savetxt(s, dat, delimiter=delimiter, fmt=fmt, encoding=encoding)
        return s.getvalue().decode(encoding=encoding)

    header = '# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n'
    tmp = np.zeros((len(aos), 3))
    tmp[:, 0] = cb.Lp * aos.values
    pos = nda2str(tmp)

    tmp = np.concatenate(([len(aos)], range(len(aos))))
    lin = nda2str(tmp.reshape((1, len(tmp))), fmt='%d')
    hdv = 'vectors offset float\n'
    hdc = 'scalars cat float\nLOOKUP_TABLE default\n'
    tmp = cb.altitude_1s(aos.values)
    cat = nda2str(tmp)

    hde = 'scalars ef float\nLOOKUP_TABLE default\n'

    for i, t in enumerate(res.data[simtools.__stime__]):
        with open(os.path.join(rep, file + f'_{i:06d}.vtk'),
                  mode='w', encoding="utf-8") as f:
            f.write(header)

            f.write(f'POINTS {len(aos)} float\n')
            f.write(pos)

            f.write(f'LINES 1 {1 + len(aos)}\n')
            f.write(lin)

            f.write(f'POINT_DATA {len(aos)}\n')

            f.write(hdv)
            tmp = nda2str(np.column_stack((res.data['ux'].values[i, :],
                                           res.data['uy'].values[i, :],
                                           res.data['uz'].values[i, :])))
            f.write(tmp)

            f.write(hdc)
            f.write(cat)

            f.write(hde)
            f.write(nda2str(res.data['ef'].values[i, :]))
