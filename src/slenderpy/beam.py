"""Beam object and associated solvers."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, List, Union, Optional, Callable
import numpy as np
import scipy as sp
from scipy.optimize import newton
from structvib import _cable_utils as cbu
from structvib import _progress_bar as spb
from structvib import fdm_utils as fdu
from structvib import simtools


def _bmom(k, cmin, cmax,  kc):
    kb = (1. - cmin / cmax) * kc
    return (cmin * k + cmax * kb) * (1. - np.exp(-k / kb))


def _bstf(k, cmin, cmax, kc):
    kb = (1. - cmin / cmax) * kc
    return cmin + (cmin * k + (cmax - cmin) * kb) * np.exp(-k / kb) / kb


class MASModel:
    """Bending moment and bending stiffness model."""

    @staticmethod
    def _eval_m(k, kp, ei, mi):
        """Evaluate bending moment."""
        m = np.zeros_like(k)
        for j in range(len(ei)):
            if j < len(ei) - 1:
                ix = np.logical_and(k >= kp[j], k < kp[j + 1])
            else:
                ix = k >= kp[-1]
            m[ix] = ei[j] * (k[ix] - kp[j]) + mi[j]
        return m

    @staticmethod
    def _eval_c(k, kp, ei):
        """Evaluate bending stiffness."""
        c = np.zeros_like(k)
        for j in range(len(ei)):
            if j < len(ei) - 1:
                ix = np.logical_and(k >= kp[j], k < kp[j + 1])
            else:
                ix = k >= kp[-1]
            c[ix] = ei[j]
        return c

    def __init__(self,
                 ei: Union[List[float], np.ndarray],
                 kp: Union[List[float], np.ndarray]) -> None:
        """Init with args.

        Arg ei must have one element more than kp arg.

        Parameters
        ----------
        ei : list or numpy.ndarray of floats
            List of bending stiffnesses (from largest to smallest).
        kp : list or numpy.ndarray of floats
            List of transition curvatures (from smallest to largest).

        Returns
        -------
        None.
        """
        if isinstance(ei, list):
            ei = np.array(ei)
        if isinstance(kp, list):
            kp = np.array(kp)

        v_ = [ei, kp]
        s_ = ['ei', 'kp']
        m_ = [2, 1]
        for i in range(2):
            v = v_[i]
            s = s_[i]
            m = m_[i]
            if not isinstance(v, np.ndarray):
                raise TypeError(f'input {s} must be a numpy.ndarray')
            if len(v.shape) != 1:
                raise ValueError(f'array {s} must be 1D')
            if len(v) < m:
                raise ValueError(f'array {s} must contain at least {m} elements')
            if np.any(v <= 0.):
                raise ValueError(f'all elements in {s} must be strictly positive')
        if np.any(np.diff(ei) >= 0.):
            raise ValueError('all elements in ei must be in descending, '
                             'no duplicates are allowed')
        if np.any(np.diff(kp) <= 0.):
            raise ValueError('all elements in kp must be in ascending order, '
                             'no duplicates are allowed')
        if len(ei) != len(kp) + 1:
            raise ValueError('array ei should have one element more than kp')

        self.kp = np.concatenate(([0.], kp))
        self.ei = np.array(ei)
        self.mi = np.concatenate(([0.],
                                  np.cumsum(self.ei[:-1] * np.diff(self.kp))))
        self.kc = ((self.mi[-1] - self.ei[-1] * self.kp[-1])
                   / (self.ei[0] - self.ei[-1]))

    def _c_sup(self):
        """Coefficients for simplification with overestimation."""
        kp = [self.kp[0], self.kc]
        ei = self.ei[[0, -1]]
        mi = [self.mi[0], self.ei[0] * self.kc]
        return kp, ei, mi

    def _c_inf(self):
        """Coefficients for simplification with underestimation."""
        if len(self.ei) <= 2:
            return self.kp, self.ei, self.mi
        kp = self.kp[[0, 1, -1]]
        ei = [self.ei[0],
              (self.mi[-1] - self.mi[1]) / (self.kp[-1] - self.kp[1]),
              self.ei[-1]]
        mi = self.mi[[0, 1, -1]]
        return kp, ei, mi

    def eval_m_sup(self, k):
        """Evaluate bending moment (simplification with overestimation)."""
        kp, ei, mi = self._c_sup()
        return MASModel._eval_m(k, kp, ei, mi)

    def eval_m(self, k):
        """Evaluate bending moment."""
        return MASModel._eval_m(k, self.kp, self.ei, self.mi)

    def eval_m_inf(self, k):
        """Evaluate bending moment (simplification with underestimation)."""
        kp, ei, mi = self._c_inf()
        return MASModel._eval_m(k, kp, ei, mi)

    def eval_c_sup(self, k):
        """Evaluate bending stiffness (simplification with overestimation)."""
        kp, ei, _ = self._c_sup()
        return MASModel._eval_c(k, kp, ei)

    def eval_c(self, k):
        """Evaluate bending stiffness."""
        return MASModel._eval_c(k, self.kp, self.ei)

    def eval_c_inf(self, k):
        """Evaluate bending stiffness (simplification with underestimation)."""
        kp, ei, _ = self._c_inf()
        return MASModel._eval_c(k, kp, ei)

    def eval_m_smooth(self, k):
        """Evaluate bending moment (Cinf version)."""
        return _bmom(k, self.ei[-1], self.ei[0], self.kc)

    def eval_c_smooth(self, k):
        """Evaluate bending stiffness (Cinf version)."""
        return _bstf(k, self.ei[-1], self.ei[0], self.kc)

    def eval_c_cont(self, k):
        """Evaluate bending stiffness (C0 version)."""
        if len(self.ei) <= 2:
            raise ValueError()

        kn = np.concatenate((self.kp[[1]],
                             0.5 * (self.kp[1:-1] + self.kp[2:]),
                             self.kp[[-1]]))

        c = np.zeros_like(k)
        c[k <= kn[0]] = self.ei[0]
        for j in range(len(kn) - 1):
            if j < len(kn) - 1:
                ix = np.logical_and(k >= kn[j], k < kn[j + 1])
            c[ix] = ((self.ei[j + 1] - self.ei[j]) / (kn[j + 1] - kn[j])
                     * (k[ix] - kn[j]) + self.ei[j])
        c[k >= kn[-1]] = self.ei[-1]
        return c


class Beam:
    """A Beam object."""

    def __init__(self,
                 mass: Optional[float] = None,
                 ei: Optional[Union[List[float], np.ndarray]] = None,
                 kp: Optional[Union[List[float], np.ndarray]] = None,
                 length: Optional[float] = None,
                 tension: Optional[float] = None) -> None:
        """Init with args.

        Parameters
        ----------
        mass : float
            Mass per unit length.
        ei : list or numpy.ndarray of floats
            List of bending stiffnesses (from largest to smallest).
        kp : list or numpy.ndarray of floats
            List of transition curvatures (from smallest to largest).
        length : float
            Span length (m).
        tension : float
            Span tension (N).

        Args ei and kp are passed to build a MASModel. Mass, length and tension
        must be positive.

        Returns
        -------
        None.
        """
        vrl = [mass, length, tension]
        vrn = ['mass', 'length', 'tension']
        for i in range(len(vrl)):
            if not isinstance(vrl[i], float):
                raise TypeError(f'input {vrn[i]} must be a float')
            if vrl[i] <= 0.:
                raise ValueError(f'input {vrn[i]} must be strictly positive')

        self.m = mass  # mass per length unit (kg/m)
        self.Lp = length  # span length (m)
        self.H = tension  # tension (N)
        self.mdl = MASModel(ei, kp)  # ...

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
        return 0.5 * np.linspace(1, n, n) / self.Lp * np.sqrt(self.H / self.m)

    def natural_frequency(self):
        """Compute the natural frequency of the vibrating string."""
        return self.natural_frequencies(n=1)[0]

    def natural_frequencies_rot_free(self,
                                     n: int = 10,
                                     c: float = 0) -> np.ndarray:
        """Compute natural frequencies for pinned-beam.

        Parameters
        ----------
        n : int, optional
            Number of frequencies to compute. The default is 10.
        c : float, optional
            Input curvature to evaluate bending stiffness. The default is 0.

        The bending stiffness is constant. The input Curvature must be positive.

        Returns
        -------
        numpy.ndarray
            Array of frequencies (in Hz).
        """
        ep = self.mdl.eval_c(c) / (self.H * self.Lp**2)
        nn = np.linspace(1, n, n)
        Wn = nn * np.sqrt(1. + ep * (np.pi * nn)**2)
        return Wn * self.natural_frequency()

    def natural_frequencies_rot_none(self,
                                     n: int = 10,
                                     c: float = 0) -> np.ndarray:
        """Compute natural frequencies for clamped-beam.

        Parameters
        ----------
        n : int, optional
            Number of frequencies to compute. The default is 10.
        c : float, optional
            Input curvature to evaluate bending stiffness. The default is 0.

        The bending stiffness is constant. The input Curvature must be positive.

        Returns
        -------
        numpy.ndarray
            Array of frequencies (in Hz).
        """
        ep = self.mdl.eval_c(c) / (self.H * self.Lp**2)
        f0 = self.natural_frequency()
        nn = np.linspace(1, n, n)

        Wg = 1. + np.sqrt(ep) + (1. + 0.5 * (np.pi * nn)**2) * ep
        rs = np.zeros_like(nn)

        def sqe(x):
            return np.sqrt(.25 / ep**2 + np.pi**2 * x**2 / ep)

        def k1L(x):
            return np.sqrt(sqe(x) + .5 / ep)

        def k2L(x):
            return np.sqrt(sqe(x) - .5 / ep)

        def fun(x):
            return np.tan(k2L(x)) - k2L(x) / k1L(x)

        def dk1(x):
            return x * np.pi**2 / (ep * sqe(x) * 2 * k1L(x))

        def dk2(x):
            return x * np.pi**2 / (ep * sqe(x) * 2 * k2L(x))

        def dfn(x):
            return dk2(x) / np.cos(k2L(x))**2 + (dk2(x) * k1L(x) - dk1(x) * k2L(x)) / k1L(x)**2

        for k in range(n):
            rs[k] = newton(fun, Wg[k], fprime=dfn)

        return f0 * nn * rs

    def EImin(self):
        """Get minimal bending stiffness."""
        return self.mdl.ei[-1]

    def EImax(self):
        """Get maximal bending stiffness."""
        return self.mdl.ei[0]


def __solve_cst__(bm, pm, force=None, am=None, y0=None, v0=None, c0=None,
                  bcl=None, bcr=None, zt=0.):
    """EOM solver for beam with constant bending stiffness."""
    ns = pm.ns
    s = np.linspace(0., bm.Lp, ns)
    ds = (s[-1] - s[0]) / (ns - 1)
    N = len(s)
    n = N - 2

    # time
    t = 0.
    tf = pm.tf
    dt = tf / pm.nt
    ht = 0.5 * dt
    ht2 = ht**2

    # matrices
    A, Ba = fdu.d2M_cst(n, ds, bcl, bcr)
    D, Bd = fdu.d4M_cst(n, ds, bcl, bcr)

    # total mass
    tm = cbu.interp_init(s, am) + bm.m
    tm = tm[1:-1]
    itm = sp.sparse.diags([1. / tm], [0])

    # init
    y = cbu.interp_init(s, y0)
    v = cbu.interp_init(s, v0)
    c = np.zeros_like(y)
    y, c = cbu.adjust(y, c, bcl, bcr, A, Ba, ds)

    if c0 is None:
        c0 = 0.

    z = 4. * np.pi * bm.natural_frequency() * zt
    b = bm.mdl.eval_c(np.abs(c0))

    lov = ['y', 'c', 'M']
    res = simtools.Results(lot=pm.time_vector_output().tolist(), lov=lov, los=pm.los)
    res.update(0, s / bm.Lp, lov, [y, c, b * c])

    if force is None:
        force = cbu.ZeroForce()

    Q = itm * (bm.H * A - b * D)
    q = itm * (bm.H * Ba - b * Bd)

    Qb = np.zeros((5, n))
    Qb[0, +2:] = -ht2 * Q.diagonal(k=2)
    Qb[1, +1:] = -ht2 * Q.diagonal(k=1)
    Qb[2, :] = 1. + ht * z - ht2 * Q.diagonal(k=0)
    Qb[3, :-1] = -ht2 * Q.diagonal(k=-1)
    Qb[4, :-2] = -ht2 * Q.diagonal(k=-2)

    # loop
    res.start_timer()
    pb = spb.generate(pm.pp, pm.nt, desc=__name__)
    for k in range(pm.nt):
        f1, _ = force(s, t, y, None, v, None)
        f2, _ = force(s, t + dt, y, None, v, None)

        r1 = y[1:-1] + ht * v[1:-1]
        r2 = v[1:-1] + ht * (Q * y[1:-1] + 2. * q - z * v[1:-1] + itm * (f1[1:-1] + f2[1:-1]))

        vn = sp.linalg.solve_banded((2, 2), Qb, r2 + ht * (Q * r1))
        yn = r1 + ht * vn

        y[1:-1] = yn
        v[1:-1] = vn
        y, c = cbu.adjust(y, c, bcl, bcr, A, Ba, ds)

        t += dt
        if (k + 1) % pm.rr == 0:
            res.update((k // pm.rr) + 1, s / bm.Lp, lov, [y, c, b * c])
            pb.update(pm.rr)
    # end loop
    pb.close()
    res.stop_timer()
    res.set_state({"y": y, "v": v, "c0": c0})

    return res


def solve_cst(bm: Beam,
              pm: simtools.Parameters,
              force: Optional[Callable[[np.ndarray, float, np.ndarray,
                                        np.ndarray, np.ndarray, np.ndarray],
                                        Tuple[np.ndarray, np.ndarray]]] = None,
              am: Optional[np.ndarray] = None,
              y0: Optional[np.ndarray] = None,
              v0: Optional[np.ndarray] = None,
              c0: Optional[float] = None,
              bcl: Optional[fdu.BoundaryCondition] = None,
              bcr: Optional[fdu.BoundaryCondition] = None,
              zt: float = 0.) -> simtools.Results:
    """EOM solver for beam with constant bending stiffness.

    Parameters
    ----------
    bm : structvib.beam.Beam
        A beam object.
    pm : structvib.simtools.Parameters
        Simulation parameters.
    force : TYPE, optional
        A force object. The default is None, which will lead to no force applied.
        This object can be any object with a __call__ method and 6 arguments: s
        (array of struct. elements), t (time), yn (normal offset), yb (binormal
        offset), vn (normal speed) and vb (binormal speed).
    am: numpy.ndarray, optional
    y0 : numpy.ndarray, optional
        Initial position. The default is None. If set to None a zero position
        will be used.
    v0 : numpy.ndarray, optional
        Initial velocity. The default is None. If set to None a zero velocity
        will be used.
    c0 : float, optional
        Curvature to compute bending stiffness. The default is None. If set to
        None a zero value will be used.
    bcl : structvib.fdm_utils.BoundaryCondition
        Left boundary condition.
    bcr : structvib.fdm_utils.BoundaryCondition
        Right boundary condition.
    zt : float, optional
        Fluid-friction coefficient. The default is 0.

    Returns
    -------
    structvib.simtools.Results
        Simulation output with offset, curvature and bending moment for
        the positions and times specified in input parameters.
    """
    return __solve_cst__(bm, pm, force=force, am=am, y0=y0, v0=v0, c0=c0, bcl=bcl, bcr=bcr, zt=zt)


def solve_ft(bm: Beam,
             pm: simtools.Parameters,
             force: Optional[Callable[[np.ndarray, float, np.ndarray,
                                       np.ndarray, np.ndarray, np.ndarray],
                                      Tuple[np.ndarray, np.ndarray]]] = None,
             am: Optional[np.ndarray] = None,
             y0: Optional[np.ndarray] = None,
             v0: Optional[np.ndarray] = None,
             m0: Optional[np.ndarray] = None,
             bcl: Optional[fdu.BoundaryCondition] = None,
             bcr: Optional[fdu.BoundaryCondition] = None,
             zt: float = 0.) -> simtools.Results:
    """EOM solver for beam with hysteretic behaviour (foti model).

    Parameters
    ----------
    bm : structvib.beam.Beam
        A beam object.
    pm : structvib.simtools.Parameters
        Simulation parameters.
    force : TYPE, optional
        A force object. The default is None, which will lead to no force applied.
        This object can be any object with a __call__ method and 6 arguments: s
        (array of struct. elements), t (time), yn (normal offset), yb (binormal
        offset), vn (normal speed) and vb (binormal speed).
    am: numpy.ndarray, optional
    y0 : numpy.ndarray, optional
        Initial position. The default is None. If set to None a zero position
        will be used.
    v0 : numpy.ndarray, optional
        Initial velocity. The default is None. If set to None a zero velocity
        will be used.
    m0 : numpy.ndarray, optional
        Initial moment. The default is None. If set to None a zero vmomentelocity
        will be used.
    bcl : structvib.fdm_utils.BoundaryCondition
        Left boundary condition.
    bcr : structvib.fdm_utils.BoundaryCondition
        Right boundary condition.
    zt : float, optional
        Fluid-friction coefficient. The default is 0.

    Returns
    -------
    structvib.simtools.Results
        Simulation output with offset, curvature, internal hysteresis variable
        and bending moment for the positions and times specified in input
        parameters.
    """
    vrl = [bcl, bcr]
    vrn = ['bcl', 'bcr']
    for i in range(len(vrl)):
        if not isinstance(vrl[i], fdu.BoundaryCondition):
            raise TypeError(f'input {vrn[i]} must be a structvib.fdm_utils.'
                            'BoundaryCondition')

    # space
    ns = pm.ns
    s = np.linspace(0., bm.Lp, ns)
    ds = (s[-1] - s[0]) / (ns - 1)
    n = ns - 2

    # time
    t = 0.
    tf = pm.tf
    dt = tf / pm.nt
    ht = 0.5 * dt
    ht2 = ht**2

    #
    cm = bm.EImin()
    cM = bm.EImax()
    x0 = bm.mdl.kc
    sg = 0.5
    al = (cM - cm) * x0

    # matrices
    D2, B2 = fdu.d2M_cst(n, ds, bcl, bcr)
    D4, B4 = fdu.d4M_cst(n, ds, bcl, bcr)

    # total mass
    tm = cbu.interp_init(s, am) + bm.m
    tm = tm[1:-1]
    itm = sp.sparse.diags([1. / tm], [0])

    # init
    y = cbu.interp_init(s, y0)
    v = cbu.interp_init(s, v0)
    m = cbu.interp_init(s, m0)
    c = np.zeros_like(y)
    y, c = cbu.adjust(y, c, bcl, bcr, D2, B2, ds)
    e = (m - cm * c) / al

    z = 4. * np.pi * bm.natural_frequency() * zt

    lov = ['y', 'c', 'e', 'M']
    res = simtools.Results(lot=pm.time_vector_output().tolist(), lov=lov, los=pm.los)
    res.update(0, s / bm.Lp, lov, [y, c, e, m])

    if force is None:
        force = cbu.ZeroForce()

    P21 = itm * (bm.H * D2 - cm * D4)
    P23 = (-1. * al) * (itm * D2)
    B = 2. * (bm.H * B2 - cm * B4 - np.zeros((n,)))
    G = sp.sparse.diags([np.zeros((n - 1,)), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [-n - 1, -n, -n + 1, -2, -1, 0, +1, +2, n - 1, n, n + 1])

    # loop
    res.start_timer()
    pb = spb.generate(pm.pp, pm.nt, desc=__name__)
    for k in range(pm.nt):

        f1, _ = force(s, t, y, None, v, None)
        f2, _ = force(s, t + dt, y, None, v, None)

        P32 = sp.sparse.diags((1. + (sg - 1.) * np.abs(e[1:-1])) / x0) * D2
        P33 = sp.sparse.diags(-1. * sg * np.abs(D2 * v[1:-1]) / x0)

        R2 = itm * (B + f1[1:-1] + f2[1:-1])

        Q1 = y[1:-1] + ht * v[1:-1]
        Q2 = v[1:-1] + ht * (P21 * y[1:-1] - z * v[1:-1] + P23 * e[1:-1] + R2)
        Q3 = e[1:-1] + ht * (P32 * v[1:-1] + P33 * e[1:-1])

        G.data[0, :n - 1] = -ht * P32.diagonal(k=-1)
        G.data[1, :n] = - ht * P32.diagonal(k=0)
        G.data[2, 1:n] = -ht * P32.diagonal(k=+1)

        G.data[3, :n - 2] = -ht2 * P21.diagonal(k=-2)
        G.data[4, :n - 1] = -ht2 * P21.diagonal(k=-1)
        G.data[5, :n] = (1. + ht * z) - ht2 * P21.diagonal(k=0)
        G.data[6, 1:n] = -ht2 * P21.diagonal(k=+1)
        G.data[7, 2:n] = -ht2 * P21.diagonal(k=+2)

        G.data[4, n:-1] = -ht * P33.diagonal(k=-1)
        G.data[5, n:] = 1. - ht * P33.diagonal(k=0)
        G.data[6, n + 1:] = -ht * P33.diagonal(k=+1)

        G.data[8, n:-1] = -ht * P23.diagonal(k=-1)
        G.data[9, n:] = -ht * P23.diagonal(k=0)
        G.data[10, n + 1:] = -ht * P23.diagonal(k=+1)

        Q = np.concatenate((Q2 + ht * P21 * Q1, Q3))
        X = sp.sparse.linalg.spsolve(G.tocsr(), Q)

        v[1:-1] = X[0 * n: 1 * n]
        e[1:-1] = X[1 * n: 2 * n]
        y[1:-1] = ht * v[1:-1] + Q1

        y, c = cbu.adjust(y, c, bcl, bcr, D2, B2, ds)
        m[1:-1] = cm * c[1:-1] + al * e[1:-1]
        t += dt

        if (k + 1) % pm.rr == 0:
            res.update((k // pm.rr) + 1, s / bm.Lp, lov, [y, c, e, m])
            pb.update(pm.rr)
    # END FOR
    pb.close()
    res.stop_timer()
    res.set_state({"y": y, "v": v, "m": m})

    return res


def static_gravity_var(bm: Beam,
                       bl: fdu.BoundaryCondition,
                       br: fdu.BoundaryCondition,
                       c0: float = 0.,
                       ns: int = 1001,
                       tol: float = 1.0E-09,
                       mxi: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static solver with gravity and varying bending stiffness.

    Parameters
    ----------
    bm : structvib.beam.Beam
        A beam object.
    bl : structvib.fdm_utils.BoundaryCondition
        Left boundary condition.
    br : structvib.fdm_utils.BoundaryCondition
        Right boundary condition.
    c0 : float, optional
        Initial curvature to compute bending stiffness. The default is 0.
    ns : int, optional
        Number of discretization points. The default is 1001.
    tol : float, optional
        Tolerance for convergence. The default is 1.0E-09.
    mxi : int, optional
        Maximum number of iterations. The default is 32.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]:
        Tuple with three numpy.ndarray of type float and size (ns,); the first
        one is the point discretization used on [0, bm.Lp], the second is the
        vertical offset (in meters), and the last one is the associated
        curvature (in m**-1).
    """
    s = np.linspace(0., bm.Lp, ns)
    f = -1. * bm.m * 9.81 * np.ones_like(s)
    y = np.zeros_like(s)
    c = c0 * np.ones_like(s)
    ds = (s[-1] - s[0]) / (ns - 1)
    D2, B2 = fdu.d2M_cst(ns - 2, ds, bl, br)
    D4, B4 = fdu.d4M_cst(ns - 2, ds, bl, br)

    err = 1
    cnt = 0
    yn = np.zeros_like(y)
    while err > tol and cnt <= mxi:
        b = bm.mdl.eval_c_smooth(np.abs(c[1:-1]))
        q = bm.H / b
        A = D4 - sp.sparse.diags(q) * D2
        b = f[1:-1] / b - (B4 - q * B2)
        yn[1:-1] = sp.sparse.linalg.spsolve(A, b)
        yn, c = cbu.adjust(yn, c, bl, br, D2, B2, ds)
        err = np.max(np.abs((yn - y)))
        y[:] = yn[:]
        cnt += 1
    # print('Convergence log err is %.3E in %d iterations' % (np.log10(err), cnt))
    return s, y, c


def static_gravity_cst(bm: Beam,
                       bl: fdu.BoundaryCondition,
                       br: fdu.BoundaryCondition,
                       c0: float = 0.,
                       ns: int = 1001) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static solver with gravity and constant bending stiffness.

    Parameters
    ----------
    bm : structvib.beam.Beam
        A beam object.
    bl : structvib.fdm_utils.BoundaryCondition
        Left boundary condition.
    br : structvib.fdm_utils.BoundaryCondition
        Right boundary condition.
    c0 : float, optional
        Initial curvature to compute bending stiffness. The default is 0.
    ns : int, optional
        Number of discretization points. The default is 1001.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]:
        Tuple with three numpy.ndarray of type float and size (ns,); the first
        one is the point discretization used on [0, bm.Lp], the second is the
        vertical offset (in meters), and the last one is the associated
        curvature (in m**-1).
    """
    return static_gravity_var(bm, bl, br, c0=c0, ns=ns, mxi=0)


def static_gravity_ft(bm: Beam,
                      bl: fdu.BoundaryCondition,
                      br: fdu.BoundaryCondition,
                      ns: int = 1001)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Static solver with gravity and hysteretic behaviour (foti model).

    Parameters
    ----------
    bm : structvib.beam.Beam
        A beam object.
    bl : structvib.fdm_utils.BoundaryCondition
        Left boundary condition.
    br : structvib.fdm_utils.BoundaryCondition
        Right boundary condition.
    ns : int, optional
        Number of discretization points. The default is 1001.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Tuple with five numpy.ndarray of type float and size (ns,):
          - point discretization used on [0, bm.Lp] (m)
          - vertical offset (m)
          - curvature (m**-1)
          - bending stiffness (N.m**2)
          - moment (N.m)
    """
    from structvib import force
    tf = 5. / bm.natural_frequency()
    dt = 0.02 / bm.natural_frequency()
    pm = simtools.Parameters(ns=ns, t0=0., tf=tf, dt=dt, dr=10 * dt, los=ns, pp=False)
    _, y0, c0 = static_gravity_var(bm, bl, br, c0=0., ns=ns)
    m0 = c0 * bm.mdl.eval_c_smooth(np.abs(c0))
    gv = force.Excitation(f=1., a=0., s=0.5 * bm.Lp, m=bm.m, L=bm.Lp, gravity=True)
    rs = solve_ft(bm, pm, force=gv, y0=y0, v0=None, m0=m0, bcl=bl, bcr=br, zt=1.)

    y = rs.data['y'][-1, :].values
    c = rs.data['c'][-1, :].values
    b = bm.mdl.eval_c_smooth(np.abs(c))
    m = rs.data['M'][-1, :].values

    return np.linspace(0., bm.Lp, ns), y, c, b, m


def _test_bending_stiffness(bm, kk):
    cmin = bm.EImin()
    cmax = bm.EImax()

    _, ax = mpl.subplots(nrows=1, ncols=2)

    ax[0].plot(kk[[0, -1]], [cmin, cmin], '--', c='gray')
    ax[0].plot(kk[[0, -1]], [cmax, cmax], '--', c='gray')
    ax[0].plot(kk, bm.mdl.eval_c(kk), '-', c='C0')
    ax[0].plot(kk, bm.mdl.eval_c_cont(kk), '-', c='C1')
    ax[0].plot(kk, bm.mdl.eval_c_smooth(kk), '-', c='C2')
    ax[0].grid(True)
    ax[0].set_xlabel('curvature (m$^{-1}$)')
    ax[0].set_ylabel('bending stiffness (Nm$^2$)')

    ax[1].plot(kk, bm.mdl.eval_m_sup(kk), '--', c='gray')
    ax[1].plot(kk, bm.mdl.eval_m_inf(kk), '--', c='gray')
    ax[1].plot(kk, bm.mdl.eval_m(kk), '-', c='C0')
    ax[1].plot(kk, bm.mdl.eval_m_smooth(kk), '-', c='C2')
    ax[1].grid(True)
    ax[1].set_xlabel('curvature (m$^{-1}$)')
    ax[1].set_ylabel('bending moment (Nm)')


def _test_frequencies(bm, n, d):
    fn = bm.natural_frequencies(n)
    xn = range(1, n + 1)
    cb = mpl.cm.Blues(np.linspace(0.2, 0.8, len(d)))
    co = mpl.cm.Oranges(np.linspace(0.2, 0.8, len(d)))

    mpl.figure()

    for i, k in enumerate(d):
        cf = bm.natural_frequencies_rot_none(n=n, c=k)
        mpl.plot(xn, cf / fn, '.-', c=cb[i],
                 label=f'rot none, c={k:.1E}')

    for i, k in enumerate(d):
        ff = bm.natural_frequencies_rot_free(n=n, c=k)
        mpl.plot(xn, ff / fn, '.-', c=co[i],
                 label=f'rot free, c={k:.1E}')

    mpl.xlabel('mode ($n$)')
    mpl.ylabel('normalized freq ($f_n/nf_0$, Hz)')
    mpl.legend()
    mpl.grid(True)


def _test_solve(bm):
    zt = 0.
    ex = Excitation(f=0.1, a=-1.0E+03, s=0.5 * bm.Lp, m=bm.m, L=bm.Lp, t0=0.,
                    tf=np.inf, gravity=False, g=9.81)
    pm = simtools.Parameters(ns=1001, t0=0., tf=10., dt=2.0E-03, dr=5.0E-02,
                             los=[0.1, 0.2, 0.3, 0.4, 0.5], pp=True)
    bl, br = fdu.rot_none('left', y=0., dy=0.), fdu.rot_none('right', y=0., dy=0.)

    rcm = solve_cst(bm, pm, force=ex, y0=None, v0=None, c0=bm.mdl.kp[-1],
                    bcl=bl, bcr=br, zt=zt)
    rcM = solve_cst(bm, pm, force=ex, y0=None, v0=None, c0=bm.mdl.kp[0],
                    bcl=bl, bcr=br, zt=zt)
    rft = solve_ft(bm, pm, force=ex, y0=None, v0=None, m0=None,
                   bcl=bl, bcr=br, zt=zt)

    rft.drop(lov=['e'])
    fg, ax = simtools.multiplot([rcM, rft, rcm],
                                lb=['cmax', 'foti', 'cmin'], Lref=bm.Lp)

    km = 0.1
    jj = -1

    mpl.figure()
    mpl.plot([0, km], [0, km * bm.EImax()], '--', c='gray')
    mpl.plot([0, km], [0, km * bm.EImin()], '--', c='gray')
    for r in [rft]:
        mpl.plot(r.data['c'][:, jj], r.data['M'][:, jj], '-',
                 label=str(pm.los[jj]))
    mpl.grid(True)

    kk = np.linspace(0, km, 101)
    mpl.plot(kk, bm.mdl.eval_m(kk), '--', c='red')

    mpl.xlabel('curvature (m$^{-1}$)')
    mpl.ylabel('bending moment (Nm)')


if __name__ == '__main__':
    import matplotlib.pyplot as mpl
    from matplotlib import cm

    from structvib.force import Excitation

    mpl.close('all')

    kk = np.linspace(0., 0.5, 501)
    bm = Beam(mass=1.57, ei=[2155., 797., 222., 50., 28.],
              kp=[0.0073, 0.024, 0.058, 0.19],
              length=50., tension=2.8E+04)

    _test_bending_stiffness(bm, kk)
    _test_frequencies(bm, 10, [0., 0.01, 0.04, 0.1, 0.4])
    _test_solve(bm)
