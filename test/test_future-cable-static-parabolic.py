import numpy as np
from slenderpy.future.cable.static import parabolic


def test_parabolic_shape(ast570, random_spans):
    """Check that parabolic shape's ends matche span length and support level difference."""
    nx = 999
    tol = 1.0E-15

    linm, _, rts = ast570
    lspan, tratio, sld = random_spans
    tension = rts * tratio
    x = np.linspace(0, lspan, nx)
    y = parabolic.shape(x, lspan, tension, sld, linm)

    assert (
            np.all(np.isclose(x[0, :], 0., atol=tol)) and
            np.all(np.isclose(x[-1, :], lspan, atol=tol)) and
            np.all(np.isclose(y[0, :], 0., atol=tol)) and
            np.all(np.isclose(y[-1, :], sld, atol=tol))
    )


def test_parabolic_length(ast570, random_spans):
    """Check that the computed parabolic length matches a numerical one."""
    nx = 59999
    atol = 1.0E-06
    rtol = 1.0E-09

    linm, _, rts = ast570
    lspan, tratio, sld = random_spans
    tension = rts * tratio
    x = np.linspace(0, lspan, nx)
    y = parabolic.shape(x, lspan, tension, sld, linm)

    length_1 = parabolic.length(lspan, tension, sld, linm)
    length_2 = np.sqrt(np.diff(x, axis=0)**2 + np.diff(y, axis=0)**2).sum(axis=0)

    assert np.all(np.isclose(length_1, length_2, atol=atol, rtol=rtol))


def test_parabolic_sag(ast570, random_spans):
    """Check that the computed sag matches a numerical one."""
    nx = 9999
    rtol = 1.0E-09

    linm, _, rts = ast570
    lspan, tratio, sld = random_spans
    tension = rts * tratio

    x = np.linspace(0, lspan, nx)
    y = parabolic.shape(x, lspan, tension, sld, linm)

    x0 = parabolic.argsag(lspan, tension, sld, linm)
    s0 = parabolic.sag(lspan, tension, sld, linm)

    ix = np.argmin(y, axis=0)
    ns = len(lspan)
    x1 = x[ix, range(ns)]
    s1 = sld * x1 / lspan - y[ix, range(ns)]

    atolx = lspan / (nx - 1)
    x2 = np.vstack((x1 + atolx, x1, x1 - atolx))
    y2 = sld * x2 / lspan - parabolic.shape(x2, lspan, tension, sld, linm)
    atoly = 0.5 * np.maximum(np.abs(y2[0, :] - y2[1, :]), np.abs(y2[1, :] - y2[2, :]))

    assert (
            np.all(np.isclose(x1, x0, atol=atolx, rtol=rtol)) and
            np.all(np.isclose(s1, s0, atol=atoly, rtol=rtol))
    )


def test_parabolic_chord(ast570, random_spans):
    """Check that the computed chord length matches a numerical one."""
    nx = 9999
    rtol = 1.0E-09

    linm, _, rts = ast570
    lspan, tratio, sld = random_spans
    tension = rts * tratio

    x = np.linspace(0, lspan, nx)
    y = parabolic.shape(x, lspan, tension, sld, linm)

    x0 = 0.5 * lspan
    s0 = parabolic.max_chord(lspan, tension, sld, linm)

    ix = np.argmax(sld * x / lspan - y, axis=0)
    ns = len(lspan)
    x1 = x[ix, range(ns)]
    s1 = sld * x1 / lspan - y[ix, range(ns)]

    atolx = lspan / (nx - 1)
    x2 = np.vstack((x1 + atolx, x1, x1 - atolx))
    y2 = sld * x2 / lspan - parabolic.shape(x2, lspan, tension, sld, linm)
    atoly = 0.5 * np.maximum(np.abs(y2[0, :] - y2[1, :]), np.abs(y2[1, :] - y2[2, :]))

    assert (
            np.all(np.isclose(x1, x0, atol=atolx, rtol=rtol)) and
            np.all(np.isclose(s1, s0, atol=atoly, rtol=rtol))
    )
