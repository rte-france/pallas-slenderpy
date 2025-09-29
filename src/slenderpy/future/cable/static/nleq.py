"""Resolution of an extensible cable model with axial stiffness."""

from typing import Union

import numpy as np
from pyntb.optimize import qnewt2d_v

from slenderpy.future._constant import _GRAVITY
from slenderpy.future.cable.static import blondel
from slenderpy.future.cable.static.parabolic import _g

_RTOL = 1.0e-12
_MAXITER = 16


def _msg(name, c, e):
    """Print msg when convergence fails."""
    print(
        f"{name} : max count is {np.max(c)}, "
        f"log10 max err is {np.log10(np.max(e)):.2f}"
    )
    return


def _xpos(s, tension, linw, axs, lve):
    """Horizontal position when moving along curvilinear abscissa.

    From Pierre Latteur, "Calculer une structure : de la théorie à l'exemple",
    Bruylant, 2006. Chap. 13, paragraph 7, equation [3]; see
    https://www.issd.be/PDF/13_Chap13_6Juillet2006.pdf.

    Parameters
    ----------
    s : curvilinear abscissa along cable (m)
    tension : mechanical tension (N)
    linw : linear weight (N.m**-1)
    axs : axial stiffness (N)
    lve : left vertical effort (N)

    Returns
    -------
    Horizontal position (m)

    """
    return (tension * s / axs) + (tension / linw) * (
        np.arcsinh(lve / tension) - np.arcsinh((lve - linw * s) / tension)
    )


def _ypos(s, tension, linw, axs, lve):
    """Vertical position when moving along curvilinear abscissa.

    From Pierre Latteur, "Calculer une structure : de la théorie à l'exemple",
    Bruylant, 2006. Chap. 13, paragraph 7, equation [4]; see
    https://www.issd.be/PDF/13_Chap13_6Juillet2006.pdf.

    Parameters
    ----------
    s : curvilinear abscissa along cable (m)
    tension : mechanical tension (N)
    linw : linear weight (N.m**-1)
    axs : axial stiffness (N)
    lve : left vertical effort (N)

    Returns
    -------
    Vertical position (m)

    """
    return (s / axs) * (lve - 0.5 * linw * s) + (tension / linw) * (
        np.sqrt(1 + (lve / tension) ** 2)
        - np.sqrt(1 + ((lve - linw * s) / tension) ** 2)
    )


def solve(
    lspan: Union[float, np.ndarray],
    tension: Union[float, np.ndarray],
    sld: Union[float, np.ndarray],
    linm: Union[float, np.ndarray],
    axs: Union[float, np.ndarray],
    g=_GRAVITY,
    rtol=_RTOL,
    maxiter=_MAXITER,
) -> Union[float, np.ndarray]:
    """Solve cable equilibrium with quasi-newton.

    From Pierre Latteur, "Calculer une structure : de la théorie à l'exemple",
    Bruylant, 2006. Chap. 13, paragraph 7. See
    https://www.issd.be/PDF/13_Chap13_6Juillet2006.pdf.

    Here we solve the equation system [1-2] with a quasi-newton algorithm.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    axs : axial stiffness (N)
    g : gravitational acceleration (m.s**-2)
    rtol : relative tolerance for quasi-newton algorithm
    maxiter : maximum number of iterations in quasi-newton algorithm

    Returns
    -------
    lcab: cable length before applying load (m)
    lve: left vertical effort (N)

    Return arrays have the same size as the given inputs.

    """

    # shortcut for linear weight
    linw = -linm * g

    # first guess for cable length and vertical effort
    lg = np.sqrt(lspan**2 + sld**2)
    rg = 0.5 * linw * lg

    # functions in quasi-newton (equilibrium equation to zero)
    def fun1(l_, r_):
        return lspan - _xpos(l_, tension, linw, axs, r_)

    def fun2(l_, r_):
        return sld - _ypos(l_, tension, linw, axs, r_)

    # solve
    lcab, lve, c, e = qnewt2d_v(fun1, fun2, lg, rg, rtol=rtol, maxiter=maxiter)

    # print if problem
    if np.max(c) >= maxiter or np.max(e) >= rtol:
        _msg("solve", c, e)

    return lcab, lve


def shape(
    s: Union[int, float, np.ndarray],
    lspan: Union[float, np.ndarray],
    tension: Union[float, np.ndarray],
    sld: Union[float, np.ndarray],
    linm: Union[float, np.ndarray],
    axs: Union[float, np.ndarray],
    lcab=None,
    lve=None,
    g=_GRAVITY,
    rtol=_RTOL,
    maxiter=_MAXITER,
) -> Union[float, np.ndarray]:
    """Cable position at equilibrium.

    If args lcab or lve is None, the cable equilibrium is recomputed (via the
    function solve). If arg s is a float or an array of floats, output values
    make sense only if s is in [0, lcab] interval. If arg s is an integer, we
    replace it with a np.linspace(0, lcab, n) array.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    s : curvilinear abscissa along cable (m)
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    axs : axial stiffness (N)
    lcab: cable length before applying load (m)
    lve: left vertical effort (N)
    g : gravitational acceleration (m.s**-2)
    rtol : relative tolerance for quasi-newton algorithm
    maxiter : maximum number of iterations in quasi-newton algorithm

    Returns
    -------
    x : Horizontal position (m)
    y : Vertical position (m)

    Return arrays have the same size as the given inputs.

    """
    if lcab is None or lve is None:
        lcab, lve = solve(
            lspan, tension, sld, linm, axs, g=g, rtol=rtol, maxiter=maxiter
        )
    linw = -linm * g
    if isinstance(s, int):
        s_ = np.linspace(0, lcab, s)
    else:
        s_ = s
    x = _xpos(s_, tension, linw, axs, lve)
    y = _ypos(s_, tension, linw, axs, lve)
    return x, y


def stress(
    s: Union[int, float, np.ndarray],
    lspan: Union[float, np.ndarray],
    tension: Union[float, np.ndarray],
    sld: Union[float, np.ndarray],
    linm: Union[float, np.ndarray],
    axs: Union[float, np.ndarray],
    lcab=None,
    lve=None,
    g=_GRAVITY,
    rtol=_RTOL,
    maxiter=_MAXITER,
):
    """Stress in cable when moving along curvilinear abscissa.

    From Pierre Latteur, "Calculer une structure : de la théorie à l'exemple",
    Bruylant, 2006. Chap. 13, paragraph 7, equation [5]; see
    https://www.issd.be/PDF/13_Chap13_6Juillet2006.pdf.

    If args lcab or lve is None, the cable equilibrium is recomputed (via the
    function solve). If arg s is a float or an array of floats, output values
    make sense only if s is in [0, lcab] interval. If arg s is an integer, we
    replace it with a np.linspace(0, lcab, n) array.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    s : curvilinear abscissa along cable (m)
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    axs : axial stiffness (N)
    lcab: cable length before applying load (m)
    lve: left vertical effort (N)
    g : gravitational acceleration (m.s**-2)
    rtol : relative tolerance for quasi-newton algorithm
    maxiter : maximum number of iterations in quasi-newton algorithm

    Returns
    -------
    Stress along cable (N). Return arrays have the same size as the given inputs.

    """
    if lcab is None or lve is None:
        lcab, lve = solve(
            lspan, tension, sld, linm, axs, g=g, rtol=rtol, maxiter=maxiter
        )
    linw = -linm * g
    if isinstance(s, int):
        s_ = np.linspace(0, lcab, s)
    else:
        s_ = s

    return np.sqrt(tension**2 + (lve - linw * s_) ** 2)


def mean_stress(
    lspan: Union[float, np.ndarray],
    tension: Union[float, np.ndarray],
    sld: Union[float, np.ndarray],
    linm: Union[float, np.ndarray],
    axs: Union[float, np.ndarray],
    lcab=None,
    lve=None,
    g=_GRAVITY,
    rtol=_RTOL,
    maxiter=_MAXITER,
):
    """Average stress in cable.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    axs : axial stiffness (N)
    lcab: cable length before applying load (m)
    lve: left vertical effort (N)
    g : gravitational acceleration (m.s**-2)
    rtol : relative tolerance for quasi-newton algorithm
    maxiter : maximum number of iterations in quasi-newton algorithm

    Returns
    -------
    Average stress (N). Return array has the same size as the given inputs.

    """
    if lcab is None or lve is None:
        lcab, lve = solve(
            lspan, tension, sld, linm, axs, g=g, rtol=rtol, maxiter=maxiter
        )
    linw = linm * g
    a_ = (tension / linw) ** 2
    b_ = lve / linw
    N = linw / lcab * (_g(lcab, a_, b_) - _g(0.0, a_, b_))
    return N


def length(
    lspan: Union[float, np.ndarray],
    tension: Union[float, np.ndarray],
    sld: Union[float, np.ndarray],
    linm: Union[float, np.ndarray],
    axs: Union[float, np.ndarray],
    lcab=None,
    lve=None,
    g=_GRAVITY,
    rtol=_RTOL,
    maxiter=_MAXITER,
):
    """Cable length (after applying load).

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    axs : axial stiffness (N)
    lcab: cable length before applying load (m)
    lve: left vertical effort (N)
    g : gravitational acceleration (m.s**-2)
    rtol : relative tolerance for quasi-newton algorithm
    maxiter : maximum number of iterations in quasi-newton algorithm

    Returns
    -------
    Cable length (m). Return array has the same size as the given inputs.

    """
    if lcab is None or lve is None:
        lcab, lve = solve(
            lspan, tension, sld, linm, axs, g=g, rtol=rtol, maxiter=maxiter
        )
    n = mean_stress(
        lspan,
        tension,
        sld,
        linm,
        axs,
        lcab=lcab,
        lve=lve,
        g=g,
        rtol=rtol,
        maxiter=maxiter,
    )
    return lcab * (1.0 + n / axs)


def argsag(
    lspan: Union[float, np.ndarray],
    tension: Union[float, np.ndarray],
    sld: Union[float, np.ndarray],
    linm: Union[float, np.ndarray],
    axs: Union[float, np.ndarray],
    lcab=None,
    lve=None,
    g=_GRAVITY,
    rtol=_RTOL,
    maxiter=_MAXITER,
) -> Union[float, np.ndarray]:
    """Find curvilinear abscissa where sag occurs.

    If args lcbab or lve is None, the cable equilibrium is recomputed (via the
    function solve).

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    axs : axial stiffness (N)
    lcab: cable length before applying load (m)
    lve: left vertical effort (N)
    g : gravitational acceleration (m.s**-2)
    rtol : relative tolerance for quasi-newton algorithm
    maxiter : maximum number of iterations in quasi-newton algorithm

    Returns
    -------
    Curvilinear abscissa where sag occurs (m)

    """
    if lcab is None or lve is None:
        lcab, lve = solve(
            lspan, tension, sld, linm, axs, g=g, rtol=rtol, maxiter=maxiter
        )
    ell = lve / (-linm * g)
    return np.minimum(np.maximum(ell, 0.0), lcab)


def sag(
    lspan: Union[float, np.ndarray],
    tension: Union[float, np.ndarray],
    sld: Union[float, np.ndarray],
    linm: Union[float, np.ndarray],
    axs: Union[float, np.ndarray],
    lcab=None,
    lve=None,
    g=_GRAVITY,
    rtol=_RTOL,
    maxiter=_MAXITER,
) -> Union[float, np.ndarray]:
    """Compute sag given a suspended cable characteristics.

    The sag is the vertical distance between the lowest point of the cable and
    the  line that joins the two suspensions points. When the support level
    difference is important, it can be equal to zero (the lowest point is one of
    the anchor points).

    If args lcbab or lve is None, the cable equilibrium is recomputed (via the
    function solve).

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    axs : axial stiffness (N)
    lcab: cable length before applying load (m)
    lve: left vertical effort (N)
    g : gravitational acceleration (m.s**-2)
    rtol : relative tolerance for quasi-newton algorithm
    maxiter : maximum number of iterations in quasi-newton algorithm

    Returns
    -------
    Cable sag at equilibrium (m). Return array has the same size as the given
    inputs.

    """

    # if incomplete input, compute cable equilibrium
    if lcab is None or lve is None:
        lcab, lve = solve(
            lspan, tension, sld, linm, axs, g=g, rtol=rtol, maxiter=maxiter
        )

    # compute curvilinear abscissa where sag occurs
    ell = argsag(
        lspan,
        tension,
        sld,
        linm,
        axs,
        lcab=lcab,
        lve=lve,
        g=g,
        rtol=rtol,
        maxiter=maxiter,
    )

    # compute actual sag
    linw = -linm * g
    sag_ = sld * _xpos(ell, tension, linw, axs, lve) / lspan - _ypos(
        ell, tension, linw, axs, lve
    )

    return sag_


def max_chord(
    lspan: Union[float, np.ndarray],
    tension: Union[float, np.ndarray],
    sld: Union[float, np.ndarray],
    linm: Union[float, np.ndarray],
    axs: Union[float, np.ndarray],
    lcab=None,
    lve=None,
    g=_GRAVITY,
    rtol=_RTOL,
    maxiter=_MAXITER,
) -> Union[float, np.ndarray]:
    """Maximum value taken by chord length.

    A chord is a vertical line between a point on the cable and the line that
    joins the two suspensions points. The maximum chord length is the largest
    chord possible. It is often used as an approximation for sag (and equal to
    sag if sld=0).

    If args lcbab or lve is None, the cable equilibrium is recomputed (via the
    function solve).

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension : mechanical tension (N)
    sld : support level difference (m)
    linm : linear mass (kg.m**-1)
    axs : axial stiffness (N)
    lcab: cable length before applying load (m)
    lve: left vertical effort (N)
    g : gravitational acceleration (m.s**-2)
    rtol : relative tolerance for quasi-newton algorithm
    maxiter : maximum number of iterations in quasi-newton algorithm

    Returns
    -------
    Max chord length (m). Return array has the same size as the given inputs.

    """

    # if incomplete input, compute cable equilibrium
    if lcab is None or lve is None:
        lcab, lve = solve(
            lspan, tension, sld, linm, axs, g=g, rtol=rtol, maxiter=maxiter
        )

    linw = -linm * g
    s0 = (lve - tension * sld / lspan) / linw
    ch = sld * _xpos(s0, tension, linw, axs, lve) / lspan - _ypos(
        s0, tension, linw, axs, lve
    )

    return ch


def thermexp_tension(
    lspan: Union[float, np.ndarray],
    tension_i: Union[float, np.ndarray],
    sld: Union[float, np.ndarray],
    temperature_i: Union[float, np.ndarray],
    temperature_f: Union[float, np.ndarray],
    linm_i: Union[float, np.ndarray],
    axs: Union[float, np.ndarray],
    alpha: Union[float, np.ndarray],
    g=_GRAVITY,
    rtol=_RTOL,
    maxiter=_MAXITER,
):
    """Compute new tension with temperature change.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension_i : initial mechanical tension (N)
    sld : support level difference (m)
    temperature_i : initial temperature of cable (K)
    temperature_f : final temperature of cable (K)
    linm_i : initial linear mass (kg.m**-1)
    axs : axial stiffness (N)
    alpha : thermal expansion coefficient (K**-1)
    g : gravitational acceleration (m.s**-2)
    rtol : relative tolerance for quasi-newton algorithm
    maxiter : maximum number of iterations in quasi-newton algorithm

    Returns
    -------
    Mechanical tension in final state (N). Return array has the same size as
    the given inputs.

    """

    # """Compute new tension with temperature change."""

    # first equilibrium
    lcab_i, lve_i = solve(
        lspan, tension_i, sld, linm_i, axs, g=g, rtol=rtol, maxiter=maxiter
    )

    # get new length and linear mass
    lcab_f = lcab_i * (1.0 + alpha * (temperature_f - temperature_i))
    linm_f = 1 / (1 / linm_i * (1.0 + alpha * (temperature_f - temperature_i)))

    # product shortcut
    linw = -linm_f * g

    # first guess for cable new tension and vertical effort
    tg = blondel.tension(
        linm_i * lcab_i * g, tension_i, temperature_i, temperature_f, axs, alpha
    )
    rg = lve_i

    # functions in quasi-newton (equilibrium equation to zero)
    def fun1(t_, r_):
        return lspan - _xpos(lcab_f, t_, linw, axs, r_)

    def fun2(t_, r_):
        return sld - _ypos(lcab_f, t_, linw, axs, r_)

    # solve
    tension_f, _, c, e = qnewt2d_v(fun1, fun2, tg, rg, rtol=rtol, maxiter=maxiter)
    if np.max(c) >= maxiter or np.max(e) >= rtol:
        _msg("thermexp_tension", c, e)
    tension_f[e > rtol] = np.nan

    return tension_f


def thermexp_temperature(
    lspan,
    tension_i,
    tension_f,
    sld,
    temperature_i,
    linm_i,
    axs,
    alpha,
    g=_GRAVITY,
    rtol=_RTOL,
    maxiter=_MAXITER,
):
    """Inverse of thermexp_tension, ie compute new temperature with tension change.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan : span length (m)
    tension_i : initial mechanical tension (N)
    tension_f : final mechanical tension (N)
    sld : support level difference (m)
    temperature_i : initial temperature of cable (K)
    linm_i : initial linear mass (kg.m**-1)
    axs : axial stiffness (N)
    alpha : thermal expansion coefficient (K**-1)
    g : gravitational acceleration (m.s**-2)
    rtol : relative tolerance for quasi-newton algorithm
    maxiter : maximum number of iterations in quasi-newton algorithm

    Returns
    -------
    Mechanical tension in final state (N). Return array has the same size as
    the given inputs.

    """

    # first equilibrium
    lcab_i, lve_i = solve(
        lspan, tension_i, sld, linm_i, axs, g=g, rtol=rtol, maxiter=maxiter
    )

    # first guess for cable new temperature and vertical effort
    tg = blondel.temperature(
        linm_i * lcab_i * g, tension_i, tension_f, temperature_i, axs, alpha
    )
    rg = lve_i

    # functions in quasi-newton (equilibrium equation to zero)
    def fun1(t_, r_):
        lcab = lcab_i * (1.0 + alpha * (t_ - temperature_i))
        linw = -g / (1 / linm_i * (1.0 + alpha * (t_ - temperature_i)))
        return lspan - _xpos(lcab, tension_f, linw, axs, r_)

    def fun2(t_, r_):
        lcab = lcab_i * (1.0 + alpha * (t_ - temperature_i))
        linw = -g / (1 / linm_i * (1.0 + alpha * (t_ - temperature_i)))
        return sld - _ypos(lcab, tension_f, linw, axs, r_)

    # solve
    temperature_f, _, c, e = qnewt2d_v(fun1, fun2, tg, rg, rtol=rtol, maxiter=maxiter)
    if np.max(c) >= maxiter or np.max(e) >= rtol:
        _msg("thermexp_temperature", c, e)

    return temperature_f
