"""Different models to compute cable natural frequencies and modes."""

import numpy as np
from pyntb.optimize import bisect_v

from slenderpy.future._constant import _GRAVITY
from slenderpy.future.cable.static.nleq import _RTOL, _MAXITER, length as n_length
from slenderpy.future.cable.static.parabolic import length as p_length, sag


def _natural(length, tension, linm):
    """Get natural frequency of cable."""
    return 0.5 * np.sqrt(tension / (linm * length**2))


def _natural_taut(lspan, tension, sld, linm):
    """Get natural frequency using taut string formula."""
    length = np.sqrt(lspan**2 + sld**2)
    return _natural(length, tension, sld, linm)


def _natural_parabolic(lspan, tension, sld, linm, g=_GRAVITY):
    """Get natural frequency using taut string formula and length from parabolic model."""
    length = p_length(lspan, tension, sld, linm, g=g)
    return _natural(length, tension, sld, linm)


def _natural_nleq(
    lspan, tension, sld, linm, axs, g=_GRAVITY, rtol=_RTOL, maxiter=_MAXITER
):
    """Get natural frequency using taut string formula and length from nleq models."""
    length = n_length(lspan, tension, sld, linm, axs, g=g, rtol=rtol, maxiter=maxiter)
    return _natural(length, tension, sld, linm)


def natural(
    lspan,
    tension,
    sld,
    linm,
    axs=None,
    method="taut",
    g=_GRAVITY,
    rtol=_RTOL,
    maxiter=_MAXITER,
):
    """Get natural frequency using taut string formula with different lengths according to arg method."""
    if method == "taut":
        return _natural_taut(lspan, tension, sld, linm)
    elif method == "parabolic":
        return _natural_parabolic(lspan, tension, sld, linm, g=g)
    elif method == "nleq":
        return _natural_nleq(
            lspan, tension, sld, linm, axs, g=g, rtol=rtol, maxiter=maxiter
        )
    else:
        raise ValueError()


def irvine_number(lspan, tension, sld, linm, axs, g=_GRAVITY):
    """Compute Irvine number."""
    r = sag(lspan, tension, sld, linm, g=g) / lspan
    return np.sqrt(64 * r**2 * lspan / (1 + 8 * r**2) * axs / tension)


def _irvine_frequencies(
    lspan: float,
    tension: float,
    sld: float,
    linm: float,
    axs: float,
    g=_GRAVITY,
    n: int = 10,
    tol: float = 1.0e-09,
    maxiter: int = 64,
) -> np.ndarray:
    """Compute Irvine frequencies.

    Solve transcendental equation in [Irvine1974]. Float version.

    Parameters
    ----------

    Returns
    -------

    """
    lm = irvine_number(lspan, tension, sld, linm, axs, g=g)
    f0 = _natural_taut(lspan, tension, sld, linm)

    def fun(x):
        return np.tan(0.5 * x) - 0.5 * x + 0.5 * x**3 / lm**2

    xm = (2 * np.arange(n) + 1) * np.pi
    xM = (2 * np.arange(n) + 3) * np.pi
    x, e = bisect_v(fun, xm, xM, xm.shape, tol=tol, maxiter=maxiter)

    return f0 * x / np.pi


def _ip_frequencies(
    lspan: float,
    tension: float,
    sld: float,
    linm: float,
    axs: float,
    g=_GRAVITY,
    n: int = 10,
    tol: float = 1.0e-09,
    maxiter: int = 64,
) -> np.ndarray:
    """Compute in-plane natural frequencies (normal direction).

    Parameters
    ----------

    Returns
    -------

    """
    f0 = _natural_taut(lspan, tension, sld, linm)
    fq = f0 * np.arange(1, n + 1)
    ni = (1 + n) // 2
    tf = _irvine_frequencies(
        lspan, tension, sld, linm, axs, n=ni, tol=tol, maxiter=maxiter
    )
    fq[::2] = tf
    return fq


def _op_frequencies(
    lspan: float,
    tension: float,
    sld: float,
    linm: float,
    axs: float,
    g=_GRAVITY,
    n: int = 10,
) -> np.ndarray:
    """Compute out-of-plane natural frequencies (binormal direction).

    Parameters
    ----------

    Returns
    -------

    """
    f0 = _natural_taut(lspan, tension, sld, linm)
    return f0 * np.arange(1, n + 1)
