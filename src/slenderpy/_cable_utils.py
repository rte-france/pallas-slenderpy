"""Utility functions hidden to the user."""
from typing import Tuple, Optional

import numpy as np
import scipy.interpolate
import scipy.sparse
from scipy.optimize import brenth
from structvib import fdm_utils as fdmu


class ZeroForce:
    """Base class for a force to apply on a structure."""

    def __call__(self,
                 s: np.ndarray,
                 t: float,
                 un: Optional[np.ndarray] = None,
                 ub: Optional[np.ndarray] = None,
                 vn: Optional[np.ndarray] = None,
                 vb: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute force value.

        Here nothing is applied hence zero is returned. If args un, ub, vn and
        vb are set they must have the same shape as s.

        Parameters
        ----------
        s : numpy.ndarray
            Space discretization. Returned arrays must have the same shape.
        t : float
            Time.
        un : numpy.ndarray, optional
            Normal offset. The default is None.
        ub : numpy.ndarray, optional
            Binormal offset. The default is None.
        vn : numpy.ndarray, optional
            Normal velocity. The default is None.
        vb : numpy.ndarray, optional
            Binormal velocity. The default is None.

        Returns
        -------
        fl : numpy.ndarray
            Lift force.
        fd : numpy.ndarray
            Drag force.

        """
        fl = np.zeros_like(s)
        fd = np.zeros_like(s)
        return fl, fd


def geom_discretization(N: int = 501,
                        delta: float = 0.001) -> np.ndarray:
    """Geometric discretization of [0, 1] interval.

    With refinement towards both ends.

    Parameters
    ----------
    N : int, optional
        Number of discretization points; if not odd, N+1 is used. The default
        is 501.
    delta : float, optional
        Size of first segment; must be greater than 1/(N-1). The default is 0.001.

    Raises
    ------
    TypeError
        DESCRIPTION.
    ValueError
        DESCRIPTION.

    Returns
    -------
    numpy.ndarray
        Discretized interval.

    """
    if not isinstance(N, int):
        raise TypeError('input N must be an int')
    if N < 1:
        raise ValueError('input N must be positive')
    if N % 2 == 0:
        N += 1
    if not isinstance(delta, float):
        raise TypeError('input delta must be a float')
    if delta >= 1. / (N - 1):
        raise ValueError('input delta too large')
    n = (N - 1) // 2

    def fun(x):
        return 2. * delta * (x**n - 1.) + 1. - x

    q = brenth(fun, 1.0 + 1.0E-15, np.exp(np.log(0.5 / delta) / (n - 1)))
    d = delta * np.power(q, np.arange(0, n, 1))
    d = np.concatenate(([0.], d, np.flip(d)))
    return np.cumsum(d)


def spacediscr(ns):
    """Generate space discretization and other related values."""
    s = np.linspace(0., 1., ns)
    ds = np.diff(s)
    N = len(s)
    n = N - 2
    return ns, s, ds, N, n


def vtvl(cb):
    """Compute vt and vl velocities [Lee]."""
    vt2 = cb.H / (cb.m * cb.g * cb.L)
    vl2 = cb.EA / (cb.m * cb.g * cb.L)
    return vt2, vl2


def matrix(ds, n):
    """Generate finite differences matrixes."""
    C = fdmu.d1M(ds)
    A = fdmu.d2M(ds)
    I = scipy.sparse.eye(n)
    J = np.ones((n,))
    return C, A, I, J


def adim(cb):
    """Get time and velocity for adimensionnal vars."""
    tAd = np.sqrt(cb.L / cb.g)
    uAd = cb.L / tAd
    return tAd, uAd


def times(pm, tAd):
    """Get time-related variables."""
    t = pm.t0 / tAd
    tf = pm.tf / tAd
    dt = (tf - t) / pm.nt
    ht = 0.5 * dt
    ht2 = ht**2
    return t, tf, dt, ht, ht2


def utef(un, ub, C, s, ds, vt2):
    """Compute tangential offset and axial force [Lee]."""
    h = -1. / vt2 * un + 0.5 * ((C * un)**2 + (C * ub)**2)
    H = 0.5 * (h[:-1] + h[1:]) * ds
    ut = np.sum(H) * s - np.cumsum(np.concatenate(([0.], H)))
    e = (C * ut) + 0.5 * ((C * ut)**2 + (C * un)**2 + (C * ub)**2)
    ef = np.log(np.sqrt(1.0 + 2.0 * e))
    return ut, ef


def interp_init(s, x=None):
    """Interpolate input values for init."""
    if x is None:
        x = np.zeros_like(s)
    else:
        x = np.interp(s, np.linspace(s[0], s[-1], len(x)), x)
    return x


def init_vars(cb, s, un0, ub0, vn0, vb0, uAd, remove_cat):
    """Initialize cable variables."""
    un = interp_init(s, un0)
    ub = interp_init(s, ub0)
    vn = interp_init(s, vn0)
    vb = interp_init(s, vb0)

    if remove_cat:
        un -= cb.altitude_1s(s)
    un /= cb.L
    ub /= cb.L
    vn /= uAd
    vb /= uAd

    return un, ub, vn, vb


def adim_force(force, s, t, dt, un, ub, vn, vb, tAd, L, uAd, m, g):
    """From adim parameters, get dim force and re-adim all before return."""
    t_ = t * tAd
    tp_ = (t + dt) * tAd
    un_ = un * L
    ub_ = ub * L
    vn_ = vn * uAd
    vb_ = vb * uAd
    fn1, fb1 = force(s, t_, un_, ub_, vn_, vb_)
    fn2, fb2 = force(s, tp_, un_, ub_, vn_, vb_)
    fn1 /= (m * g)
    fb1 /= (m * g)
    fn2 /= (m * g)
    fb2 /= (m * g)

    return fn1, fn2, fb1, fb2


def adjust(y, c, bl, br, A, B, ds):
    """Adjust boundary conditions in beam solvers."""
    y[0] = bl.c1c + bl.c1q * y[1]
    y[-1] = br.c1c + br.c1q * y[-2]
    c[1:-1] = A * y[1:-1] + B
    c[0] = ((bl.c2c + bl.c2q * y[1]) - 2. * y[0] + y[1]) / ds**2
    c[-1] = (y[-2] - 2. * y[-1] + (br.c2c + br.c2q * y[-2])) / ds**2
    return y, c
