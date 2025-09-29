import numpy as np

from slenderpy.future.cable.static import catenary


def test_shape(ast570, random_spans):
    """Check that shape's ends matche span length and support level difference."""
    nx = 999
    tol = 1.0e-15

    linm, _, rts = ast570
    lspan, tratio, sld = random_spans

    tension = rts * tratio
    x = np.linspace(0, lspan, nx)
    y = catenary.shape(x, lspan, tension, sld, linm)

    assert (
        np.allclose(x[0, :], 0.0, atol=tol)
        and np.allclose(x[-1, :], lspan, atol=tol)
        and np.allclose(y[0, :], 0.0, atol=tol)
        and np.allclose(y[-1, :], sld, atol=tol)
    )


def test_length(ast570, random_spans):
    """Check that the computed length matches a numerical one."""
    nx = 59999
    atol = 1.0e-06
    rtol = 1.0e-09

    linm, _, rts = ast570
    lspan, tratio, sld = random_spans

    tension = rts * tratio
    x = np.linspace(0, lspan, nx)
    y = catenary.shape(x, lspan, tension, sld, linm)

    length_1 = catenary.length(lspan, tension, sld, linm)
    length_2 = np.sqrt(np.diff(x, axis=0) ** 2 + np.diff(y, axis=0) ** 2).sum(axis=0)

    assert np.allclose(length_1, length_2, atol=atol, rtol=rtol)


def test_sag(ast570, random_spans):
    """Check that the computed sag matches a numerical one."""
    nx = 9999
    rtol = 1.0e-09

    linm, _, rts = ast570
    lspan, tratio, sld = random_spans

    tension = rts * tratio
    x = np.linspace(0, lspan, nx)
    y = catenary.shape(x, lspan, tension, sld, linm)

    x0 = catenary.argsag(lspan, tension, sld, linm)
    s0 = catenary.sag(lspan, tension, sld, linm)

    ix = np.argmin(y, axis=0)
    ns = len(lspan)
    x1 = x[ix, range(ns)]
    s1 = sld * x1 / lspan - y[ix, range(ns)]

    atolx = lspan / (nx - 1)
    x2 = np.vstack((x1 + atolx, x1, x1 - atolx))
    y2 = sld * x2 / lspan - catenary.shape(x2, lspan, tension, sld, linm)
    atoly = 0.5 * np.maximum(np.abs(y2[0, :] - y2[1, :]), np.abs(y2[1, :] - y2[2, :]))

    assert np.allclose(x1, x0, atol=atolx, rtol=rtol) and np.allclose(
        s1, s0, atol=atoly, rtol=rtol
    )


def test_chord(ast570, random_spans):
    """Check that the computed chord length matches a numerical one."""
    nx = 9999
    rtol = 1.0e-09

    linm, _, rts = ast570
    lspan, tratio, sld = random_spans
    tension = rts * tratio

    x = np.linspace(0, lspan, nx)
    y = catenary.shape(x, lspan, tension, sld, linm)

    _, a_, x0_ = catenary._lax0(lspan, tension, sld, linm)
    x0 = a_ * np.arcsinh(sld / lspan) - 0.5 * x0_
    s0 = catenary.max_chord(lspan, tension, sld, linm)

    ix = np.argmax(sld * x / lspan - y, axis=0)
    ns = len(lspan)
    x1 = x[ix, range(ns)]
    s1 = sld * x1 / lspan - y[ix, range(ns)]

    atolx = lspan / (nx - 1)
    x2 = np.vstack((x1 + atolx, x1, x1 - atolx))
    y2 = sld * x2 / lspan - catenary.shape(x2, lspan, tension, sld, linm)
    atoly = 0.5 * np.maximum(np.abs(y2[0, :] - y2[1, :]), np.abs(y2[1, :] - y2[2, :]))

    assert np.allclose(x1, x0, atol=atolx, rtol=rtol) and np.allclose(
        s1, s0, atol=atoly, rtol=rtol
    )


# def test_stress(ast570, random_spans):
#     """Check that the computed mean stress matches its integration."""
#     nx = 59999
#     atol = 1.0e-06
#     rtol = 1.0e-09
#
#     linm, _, rts = ast570
#     lspan, tratio, sld = random_spans
#     tension = rts * tratio
#     x = np.linspace(0, lspan, nx)
#     s = catenary.stress(x, lspan, tension, sld, linm)
#
#     sm1 = catenary.mean_stress(lspan, tension, sld, linm)
#     sm2 = np.sum(0.5 * (s[1:, :] + s[:-1, :]) * np.diff(x, axis=0), axis=0) / lspan
#
#     assert np.allclose(sm1, sm2, atol=atol, rtol=rtol)
