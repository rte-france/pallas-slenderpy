"""Resolution of an extensible cable model with axial stiffness."""

from typing import Union

import numpy as np

from slenderpy import floatArrayLike
from slenderpy._constant import _GRAVITY
from slenderpy.cable.static import blondel
from slenderpy.cable.static.parabolic import _f

_RTOL = 1.0e-12
_MAXITER = 64

# relative step of the centered finite differences of the newton jacobian
_FD_STEP = 1.0e-06
# halvings of a newton step tried before giving up on a case
_MAX_HALVINGS = 30


def _msg(name, err, rtol):
    """Print msg when convergence fails."""
    print(
        f"{name} : {np.count_nonzero(~(err <= rtol))} case(s) did not "
        f"converge, log10 max residual is {np.log10(np.nanmax(err)):.2f}"
    )


def _newton2d(fun1, fun2, x, y, scale, valid, rtol, maxiter):
    """Damped newton on the system [fun1(x, y), fun2(x, y)] = [0, 0], per case.

    The jacobian is estimated with centered finite differences. Each step is
    halved until it decreases the residual norm and keeps ``valid(x)`` true; a
    case stops once ``max(|fun1|, |fun2|) <= rtol * scale``, so converged cases
    are left untouched while others iterate.

    Returns the solution and the final residual relative to ``scale``.
    """
    x, y, scale = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float), scale
    )
    x, y = x.copy(), y.copy()
    f1, f2 = fun1(x, y), fun2(x, y)
    err = np.maximum(np.abs(f1), np.abs(f2)) / scale

    for _ in range(maxiter):
        active = err > rtol
        if not active.any():
            break

        hx = _FD_STEP * np.maximum(np.abs(x), 1.0)
        hy = _FD_STEP * np.maximum(np.abs(y), 1.0)
        a = (fun1(x + hx, y) - fun1(x - hx, y)) / (2.0 * hx)
        b = (fun1(x, y + hy) - fun1(x, y - hy)) / (2.0 * hy)
        c = (fun2(x + hx, y) - fun2(x - hx, y)) / (2.0 * hx)
        d = (fun2(x, y + hy) - fun2(x, y - hy)) / (2.0 * hy)
        det = a * d - b * c
        dx = (d * f1 - b * f2) / det
        dy = (a * f2 - c * f1) / det

        # backtrack, case by case, until the step decreases the residual
        norm = np.hypot(f1, f2)
        relaxation = np.ones_like(x)
        for _ in range(_MAX_HALVINGS):
            x_new = x - relaxation * dx
            y_new = y - relaxation * dy
            g1, g2 = fun1(x_new, y_new), fun2(x_new, y_new)
            accepted = (np.hypot(g1, g2) < norm) & valid(x_new)
            if np.all(accepted | ~active):
                break
            relaxation = np.where(accepted, relaxation, 0.5 * relaxation)

        update = active & accepted
        x = np.where(update, x_new, x)
        y = np.where(update, y_new, y)
        f1 = np.where(update, g1, f1)
        f2 = np.where(update, g2, f2)
        err = np.maximum(np.abs(f1), np.abs(f2)) / scale

    return x, y, err


def _xpos(s, tension, linw, axs, lve):
    """Horizontal position when moving along curvilinear abscissa.

    From Pierre Latteur, "Calculer une structure : de la théorie à l'exemple",
    Bruylant, 2006. Chap. 13, paragraph 7, equation [3]; see
    https://www.issd.be/PDF/13_Chap13_6Juillet2006.pdf.

    Parameters
    ----------
    s
        Curvilinear abscissa along cable (m).
    tension
        Mechanical tension (N).
    linw
        Linear weight (N.m**-1).
    axs
        Axial stiffness (N).
    lve
        Left vertical effort (N).

    Returns
    -------
    float or array
        Horizontal position (m).

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
    s
        Curvilinear abscissa along cable (m).
    tension
        Mechanical tension (N).
    linw
        Linear weight (N.m**-1).
    axs
        Axial stiffness (N).
    lve
        Left vertical effort (N).

    Returns
    -------
    float or array
        Vertical position (m).

    """
    return (s / axs) * (lve - 0.5 * linw * s) + (tension / linw) * (
        np.sqrt(1 + (lve / tension) ** 2)
        - np.sqrt(1 + ((lve - linw * s) / tension) ** 2)
    )


def solve(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    axs: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
    rtol: float = _RTOL,
    maxiter: int = _MAXITER,
) -> floatArrayLike:
    """Solve cable equilibrium with a damped newton.

    From Pierre Latteur, "Calculer une structure : de la théorie à l'exemple",
    Bruylant, 2006. Chap. 13, paragraph 7. See
    https://www.issd.be/PDF/13_Chap13_6Juillet2006.pdf.

    Here we solve the equation system [1-2] with a damped newton algorithm.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    axs
        Axial stiffness (N).
    g
        Gravitational acceleration (m.s**-2).
    rtol
        Tolerance on the equilibrium residuals, relative to the span length.
    maxiter
        Maximum number of newton iterations.

    Returns
    -------
    lcab : float or array
        Cable length before applying load (m). Same shape as the broadcast
        inputs.
    lve : float or array
        Left vertical effort (N). Same shape as ``lcab``.

    """

    # shortcut for linear weight
    linw = -linm * g

    # first guess for cable length and vertical effort
    lg = np.sqrt(lspan**2 + sld**2)
    rg = 0.5 * linw * lg

    # functions in newton (equilibrium equation to zero)
    def fun1(l_, r_):
        return lspan - _xpos(l_, tension, linw, axs, r_)

    def fun2(l_, r_):
        return sld - _ypos(l_, tension, linw, axs, r_)

    # solve; the cable length stays positive
    lcab, lve, err = _newton2d(
        fun1, fun2, lg, rg, lspan, lambda l_: l_ > 0.0, rtol, maxiter
    )
    if not np.all(err <= rtol):
        _msg("solve", err, rtol)

    return lcab, lve


def shape(
    s: Union[int, floatArrayLike],
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    axs: floatArrayLike,
    lcab=None,
    lve=None,
    g: floatArrayLike = _GRAVITY,
    rtol: float = _RTOL,
    maxiter: int = _MAXITER,
) -> floatArrayLike:
    """Cable position at equilibrium.

    If args lcab or lve is None, the cable equilibrium is recomputed (via the
    function solve). If arg s is a float or an array of floats, output values
    make sense only if s is in [0, lcab] interval. If arg s is an integer, we
    replace it with a np.linspace(0, lcab, n) array.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    s
        Curvilinear abscissa along cable (m).
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    axs
        Axial stiffness (N).
    lcab
        Cable length before applying load (m), from `solve`.
    lve
        Left vertical effort (N), from `solve`.
    g
        Gravitational acceleration (m.s**-2).
    rtol
        Tolerance on the equilibrium residuals, relative to the span length.
    maxiter
        Maximum number of newton iterations.

    Returns
    -------
    x : float or array
        Horizontal position (m). Same shape as ``s`` broadcast with the other
        inputs.
    y : float or array
        Vertical position (m). Same shape as ``x``.

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
    s: Union[int, floatArrayLike],
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    axs: floatArrayLike,
    lcab=None,
    lve=None,
    g: floatArrayLike = _GRAVITY,
    rtol: float = _RTOL,
    maxiter: int = _MAXITER,
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
    s
        Curvilinear abscissa along cable (m).
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    axs
        Axial stiffness (N).
    lcab
        Cable length before applying load (m), from `solve`.
    lve
        Left vertical effort (N), from `solve`.
    g
        Gravitational acceleration (m.s**-2).
    rtol
        Tolerance on the equilibrium residuals, relative to the span length.
    maxiter
        Maximum number of newton iterations.

    Returns
    -------
    float or array
        Stress along cable (N). Same shape as ``s`` broadcast with the other
        inputs.

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
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    axs: floatArrayLike,
    lcab=None,
    lve=None,
    g: floatArrayLike = _GRAVITY,
    rtol: float = _RTOL,
    maxiter: int = _MAXITER,
):
    """Average stress in cable.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    axs
        Axial stiffness (N).
    lcab
        Cable length before applying load (m), from `solve`.
    lve
        Left vertical effort (N), from `solve`.
    g
        Gravitational acceleration (m.s**-2).
    rtol
        Tolerance on the equilibrium residuals, relative to the span length.
    maxiter
        Maximum number of newton iterations.

    Returns
    -------
    float or array
        Average stress (N). Same shape as the broadcast inputs.

    """
    if lcab is None or lve is None:
        lcab, lve = solve(
            lspan, tension, sld, linm, axs, g=g, rtol=rtol, maxiter=maxiter
        )
    a = tension / (linm * g)
    b = lve / tension
    N = tension * a / lcab * (_f(lcab / a + b) - _f(b))
    return N


def length(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    axs: floatArrayLike,
    lcab=None,
    lve=None,
    g: floatArrayLike = _GRAVITY,
    rtol: float = _RTOL,
    maxiter: int = _MAXITER,
):
    """Cable length (after applying load).

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    axs
        Axial stiffness (N).
    lcab
        Cable length before applying load (m), from `solve`.
    lve
        Left vertical effort (N), from `solve`.
    g
        Gravitational acceleration (m.s**-2).
    rtol
        Tolerance on the equilibrium residuals, relative to the span length.
    maxiter
        Maximum number of newton iterations.

    Returns
    -------
    float or array
        Cable length (m). Same shape as the broadcast inputs.

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
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    axs: floatArrayLike,
    lcab=None,
    lve=None,
    g: floatArrayLike = _GRAVITY,
    rtol: float = _RTOL,
    maxiter: int = _MAXITER,
) -> floatArrayLike:
    """Find curvilinear abscissa where sag occurs.

    If args lcbab or lve is None, the cable equilibrium is recomputed (via the
    function solve).

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    axs
        Axial stiffness (N).
    lcab
        Cable length before applying load (m), from `solve`.
    lve
        Left vertical effort (N), from `solve`.
    g
        Gravitational acceleration (m.s**-2).
    rtol
        Tolerance on the equilibrium residuals, relative to the span length.
    maxiter
        Maximum number of newton iterations.

    Returns
    -------
    float or array
        Curvilinear abscissa where sag occurs (m). Same shape as the broadcast inputs.

    """
    if lcab is None or lve is None:
        lcab, lve = solve(
            lspan, tension, sld, linm, axs, g=g, rtol=rtol, maxiter=maxiter
        )
    ell = lve / (-linm * g)
    return np.minimum(np.maximum(ell, 0.0), lcab)


def sag(
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    axs: floatArrayLike,
    lcab=None,
    lve=None,
    g: floatArrayLike = _GRAVITY,
    rtol: float = _RTOL,
    maxiter: int = _MAXITER,
) -> floatArrayLike:
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
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    axs
        Axial stiffness (N).
    lcab
        Cable length before applying load (m), from `solve`.
    lve
        Left vertical effort (N), from `solve`.
    g
        Gravitational acceleration (m.s**-2).
    rtol
        Tolerance on the equilibrium residuals, relative to the span length.
    maxiter
        Maximum number of newton iterations.

    Returns
    -------
    float or array
        Cable sag at equilibrium (m). Same shape as the broadcast inputs.

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
    lspan: floatArrayLike,
    tension: floatArrayLike,
    sld: floatArrayLike,
    linm: floatArrayLike,
    axs: floatArrayLike,
    lcab=None,
    lve=None,
    g: floatArrayLike = _GRAVITY,
    rtol: float = _RTOL,
    maxiter: int = _MAXITER,
) -> floatArrayLike:
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
    lspan
        Span length (m).
    tension
        Mechanical tension (N).
    sld
        Support level difference (m).
    linm
        Linear mass (kg.m**-1).
    axs
        Axial stiffness (N).
    lcab
        Cable length before applying load (m), from `solve`.
    lve
        Left vertical effort (N), from `solve`.
    g
        Gravitational acceleration (m.s**-2).
    rtol
        Tolerance on the equilibrium residuals, relative to the span length.
    maxiter
        Maximum number of newton iterations.

    Returns
    -------
    float or array
        Max chord length (m). Same shape as the broadcast inputs.

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


def thermal_expansion_tension(
    lspan: floatArrayLike,
    tension_i: floatArrayLike,
    sld: floatArrayLike,
    temperature_i: floatArrayLike,
    temperature_f: floatArrayLike,
    linm_i: floatArrayLike,
    axs: floatArrayLike,
    alpha: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
    rtol: float = _RTOL,
    maxiter: int = _MAXITER,
):
    """Compute new tension with temperature change.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan
        Span length (m).
    tension_i
        Initial mechanical tension (N).
    sld
        Support level difference (m).
    temperature_i
        Initial temperature of cable (K).
    temperature_f
        Final temperature of cable (K).
    linm_i
        Initial linear mass (kg.m**-1).
    axs
        Axial stiffness (N).
    alpha
        Thermal expansion coefficient (K**-1).
    g
        Gravitational acceleration (m.s**-2).
    rtol
        Tolerance on the equilibrium residuals, relative to the span length.
    maxiter
        Maximum number of newton iterations.

    Returns
    -------
    float or array
        Mechanical tension in final state (N), nan where the solver did not
        converge. Same shape as the broadcast inputs.

    """

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

    # functions in newton (equilibrium equation to zero)
    def fun1(t_, r_):
        return lspan - _xpos(lcab_f, t_, linw, axs, r_)

    def fun2(t_, r_):
        return sld - _ypos(lcab_f, t_, linw, axs, r_)

    # solve; the tension stays positive
    tension_f, _, err = _newton2d(
        fun1, fun2, tg, rg, lspan, lambda t_: t_ > 0.0, rtol, maxiter
    )
    if not np.all(err <= rtol):
        _msg("thermal_expansion_tension", err, rtol)
    tension_f = np.where(err <= rtol, tension_f, np.nan)

    return tension_f


def thermal_expansion_temperature(
    lspan: floatArrayLike,
    tension_i: floatArrayLike,
    tension_f: floatArrayLike,
    sld: floatArrayLike,
    temperature_i: floatArrayLike,
    linm_i: floatArrayLike,
    axs: floatArrayLike,
    alpha: floatArrayLike,
    g: floatArrayLike = _GRAVITY,
    rtol: float = _RTOL,
    maxiter: int = _MAXITER,
):
    """Inverse of thermal_expansion_tension: new temperature from a tension change.

    If more than one arg is an array, they must have the same size (no check).

    Parameters
    ----------
    lspan
        Span length (m).
    tension_i
        Initial mechanical tension (N).
    tension_f
        Final mechanical tension (N).
    sld
        Support level difference (m).
    temperature_i
        Initial temperature of cable (K).
    linm_i
        Initial linear mass (kg.m**-1).
    axs
        Axial stiffness (N).
    alpha
        Thermal expansion coefficient (K**-1).
    g
        Gravitational acceleration (m.s**-2).
    rtol
        Tolerance on the equilibrium residuals, relative to the span length.
    maxiter
        Maximum number of newton iterations.

    Returns
    -------
    float or array
        Temperature in final state (K), nan where the solver did not converge.
        Same shape as the broadcast inputs.

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

    # functions in newton (equilibrium equation to zero)
    def fun1(t_, r_):
        lcab = lcab_i * (1.0 + alpha * (t_ - temperature_i))
        linw = -g / (1 / linm_i * (1.0 + alpha * (t_ - temperature_i)))
        return lspan - _xpos(lcab, tension_f, linw, axs, r_)

    def fun2(t_, r_):
        lcab = lcab_i * (1.0 + alpha * (t_ - temperature_i))
        linw = -g / (1 / linm_i * (1.0 + alpha * (t_ - temperature_i)))
        return sld - _ypos(lcab, tension_f, linw, axs, r_)

    # solve; the cable length stays positive
    def valid(t_):
        return 1.0 + alpha * (t_ - temperature_i) > 0.0

    temperature_f, _, err = _newton2d(fun1, fun2, tg, rg, lspan, valid, rtol, maxiter)
    if not np.all(err <= rtol):
        _msg("thermal_expansion_temperature", err, rtol)
    temperature_f = np.where(err <= rtol, temperature_f, np.nan)

    return temperature_f
