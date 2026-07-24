"""Different models to compute cable natural frequencies and modes."""

from enum import Enum

import numpy as np
from pyntb.optimize import bisect_v

from slenderpy.future import floatArrayLike
from slenderpy.future._constant import _GRAVITY
from slenderpy.future.cable.static.catenary import length as c_length
from slenderpy.future.cable.static.nleq import _MAXITER, _RTOL
from slenderpy.future.cable.static.nleq import length as n_length
from slenderpy.future.cable.static.parabolic import length as p_length
from slenderpy.future.cable.static.parabolic import sag


class FrequencyMethod(str, Enum):
    """Length model used in the taut-string natural-frequency formula."""

    TAUT = "taut"
    PARABOLIC = "parabolic"
    CATENARY = "catenary"
    NLEQ = "nleq"


def _natural(length: floatArrayLike, tension: floatArrayLike, linm: floatArrayLike):
    """Get natural frequency of cable."""
    return 0.5 * np.sqrt(tension / (linm * length**2))


def _natural_taut(lspan: floatArrayLike, tension: floatArrayLike, sld: floatArrayLike, linm: floatArrayLike):
    """Get natural frequency using taut string formula."""
    length = np.sqrt(lspan**2 + sld**2)
    return _natural(length, tension, linm)


def _natural_parabolic(lspan: floatArrayLike, tension: floatArrayLike, sld: floatArrayLike, linm: floatArrayLike, g:float=_GRAVITY):
    """Get natural frequency using taut string formula and length from parabolic model."""
    length = p_length(lspan, tension, sld, linm, g=g)
    return _natural(length, tension, linm)


def _natural_catenary(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    g: float = _GRAVITY,
):
    """Get natural frequency using taut string formula and length from catenary model."""
    length = c_length(lspan, tension, sld, linm, g=g)
    return _natural(length, tension, linm)


def _natural_nleq(
    lspan: floatArrayLike, tension: floatArrayLike, sld: floatArrayLike, linm: floatArrayLike, axs, g:float=_GRAVITY, rtol:float=_RTOL, maxiter:int=_MAXITER
):
    """Get natural frequency using taut string formula and length from nleq models."""
    length = n_length(lspan, tension, sld, linm, axs, g=g, rtol=rtol, maxiter=maxiter)
    return _natural(length, tension, linm)


def natural(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    axs: floatArrayLike | None=None,
    method:FrequencyMethod=FrequencyMethod.TAUT,
    g:float=_GRAVITY,
    rtol:float=_RTOL,
    maxiter:int=_MAXITER,
):
    """Get natural frequency using taut string formula with different lengths according to arg method.

    The ``method`` argument accepts a :class:`FrequencyMethod` member or its
    string value ("taut", "parabolic", "catenary" or "nleq"); an invalid value
    raises ``ValueError``.
    """
    method = FrequencyMethod(method)
    if method is FrequencyMethod.TAUT:
        return _natural_taut(lspan, tension, sld, linm)
    elif method is FrequencyMethod.PARABOLIC:
        return _natural_parabolic(lspan, tension, sld, linm, g=g)
    elif method is FrequencyMethod.CATENARY:
        return _natural_catenary(lspan, tension, sld, linm, g=g)
    else:
        return _natural_nleq(
            lspan, tension, sld, linm, axs, g=g, rtol=rtol, maxiter=maxiter
        )


def irvine_number(lspan: floatArrayLike, tension: floatArrayLike, sld: floatArrayLike, linm: floatArrayLike, axs, g=_GRAVITY):
    """Compute Irvine number."""
    r = sag(lspan, tension, sld, linm, g=g) / lspan
    return np.sqrt(64 * r**2 * lspan / (1 + 8 * r**2) * axs / tension)


def _irvine_frequencies(
    lspan: float,
    tension: float,
    sld: float,
    linm: float,
    axs: float,
    g:float=_GRAVITY,
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
