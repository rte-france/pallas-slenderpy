import numpy as np

from slenderpy.future.cable.static import nleq


def test_shape(ast570, random_spans):
    """Check that shape's ends matche span length and support level difference."""
    nx = 999
    tol = 1.0e-15

    linm, axs, rts = ast570
    lspan, tratio, sld = random_spans

    tension = rts * tratio
    x, y = nleq.shape(nx, lspan, tension, sld, linm, axs)

    assert (
        np.allclose(x[0, :], 0.0, atol=tol)
        and np.allclose(x[-1, :], lspan, atol=tol)
        and np.allclose(y[0, :], 0.0, atol=tol)
        and np.allclose(y[-1, :], sld, atol=tol)
    )


def test_stress(ast570, random_spans):
    """Check that the computed mean stress matches its integration."""
    nx = 59999
    atol = 1.0e-06
    rtol = 1.0e-09

    linm, axs, rts = ast570
    lspan, tratio, sld = random_spans

    tension = rts * tratio
    lcab, lve = nleq.solve(lspan, tension, sld, linm, axs)

    s = np.linspace(0.0, lcab, nx)
    n = nleq.stress(s, lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)

    sm1 = nleq.mean_stress(lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)
    sm2 = np.sum(0.5 * (n[1:] + n[:-1]) * np.diff(s, axis=0), axis=0) / lcab

    assert np.allclose(sm1, sm2, atol=atol, rtol=rtol)


def test_length(ast570, random_spans):
    """Check that the computed length matches a numerical one."""
    nx = 9999
    rtol = 1.0e-09

    linm, axs, rts = ast570
    lspan, tratio, sld = random_spans

    tension = rts * tratio
    lcab, lve = nleq.solve(lspan, tension, sld, linm, axs)

    l = np.linspace(0, lcab, nx)
    x, y = nleq.shape(l, lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)

    length_1 = nleq.length(lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)
    length_2 = np.sqrt(np.diff(x, axis=0) ** 2 + np.diff(y, axis=0) ** 2).sum(axis=0)

    assert np.max(np.abs(length_2 - length_1) / lspan) < rtol


def test_sag(ast570, random_spans):
    """Check that the computed sag matches a numerical one."""
    nx = 9999
    rtol = 1.0e-09

    linm, axs, rts = ast570
    lspan, tratio, sld = random_spans

    tension = rts * tratio
    lcab, lve = nleq.solve(lspan, tension, sld, linm, axs)

    l = np.linspace(0, lcab, nx)
    x, y = nleq.shape(l, lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)

    l0 = nleq.argsag(lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)
    x0, y0 = nleq.shape(l0, lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)
    s0 = nleq.sag(lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)

    ix = np.argmin(y, axis=0)
    ns = len(lspan)
    l1 = l[ix, range(ns)]
    x1 = x[ix, range(ns)]
    s1 = sld * x1 / lspan - y[ix, range(ns)]

    atoll = np.max(np.diff(l, axis=0), axis=0)
    l2 = np.vstack((l1 + atoll, l1, l1 - atoll))
    x2, y2 = nleq.shape(l2, lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)
    y2 = sld * x2 / lspan - y2
    atolx = 0.5 * np.abs(x2[2, :] - x2[0, :])
    atoly = 0.5 * np.maximum(np.abs(y2[0, :] - y2[1, :]), np.abs(y2[1, :] - y2[2, :]))

    assert np.allclose(x1, x0, atol=atolx, rtol=rtol) and np.allclose(
        s1, s0, atol=atoly, rtol=rtol
    )


def test_chord(ast570, random_spans):
    """Check that the computed chord length matches a numerical one."""
    nx = 9999
    rtol = 1.0e-09

    linm, axs, rts = ast570
    lspan, tratio, sld = random_spans

    tension = rts * tratio
    lcab, lve = nleq.solve(lspan, tension, sld, linm, axs)

    l = np.linspace(0, lcab, nx)
    x, y = nleq.shape(l, lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)

    l0 = (lve - tension * sld / lspan) / (-linm * nleq._GRAVITY)
    x0, _ = nleq.shape(l0, lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)
    s0 = nleq.max_chord(lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)

    ix = np.argmax(sld * x / lspan - y, axis=0)
    ns = len(lspan)
    l1 = l[ix, range(ns)]
    x1 = x[ix, range(ns)]
    s1 = sld * x1 / lspan - y[ix, range(ns)]

    atoll = np.max(np.diff(l, axis=0), axis=0)
    l2 = np.vstack((l1 + atoll, l1, l1 - atoll))
    x2, y2 = nleq.shape(l2, lspan, tension, sld, linm, axs, lcab=lcab, lve=lve)
    y2 = sld * x2 / lspan - y2
    atolx = 0.5 * np.abs(x2[2, :] - x2[0, :])
    atoly = 0.5 * np.maximum(np.abs(y2[0, :] - y2[1, :]), np.abs(y2[1, :] - y2[2, :]))

    assert np.allclose(x1, x0, atol=atolx, rtol=rtol) and np.allclose(
        s1, s0, atol=atoly, rtol=rtol
    )
