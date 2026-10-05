import numpy as np

from slenderpy.cable.static import parabolic


def test_shape(ast570, random_spans):
    """Check that shape's ends matche span length and support level difference."""
    nx = 999
    tol = 1.0e-15

    linm, _, rts, _ = ast570
    lspan, tratio, sld = random_spans

    tension = rts * tratio
    x = np.linspace(0, lspan, nx)
    y = parabolic.shape(x, lspan, tension, sld, linm)

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

    linm, _, rts, _ = ast570
    lspan, tratio, sld = random_spans

    tension = rts * tratio
    x = np.linspace(0, lspan, nx)
    y = parabolic.shape(x, lspan, tension, sld, linm)

    length_1 = parabolic.length(lspan, tension, sld, linm)
    length_2 = np.sqrt(np.diff(x, axis=0) ** 2 + np.diff(y, axis=0) ** 2).sum(axis=0)

    assert np.allclose(length_1, length_2, atol=atol, rtol=rtol)


def test_sag(ast570, random_spans):
    """Check that the computed sag matches a numerical one."""
    nx = 9999
    rtol = 1.0e-09

    linm, _, rts, _ = ast570
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

    assert np.allclose(x1, x0, atol=atolx, rtol=rtol) and np.allclose(
        s1, s0, atol=atoly, rtol=rtol
    )


def test_chord(ast570, random_spans):
    """Check that the computed chord length matches a numerical one."""
    nx = 9999
    rtol = 1.0e-09

    linm, _, rts, _ = ast570
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

    assert np.allclose(x1, x0, atol=atolx, rtol=rtol) and np.allclose(
        s1, s0, atol=atoly, rtol=rtol
    )


def test_stress(ast570, random_spans):
    """Check that the computed mean stress matches its integration."""
    nx = 59999
    atol = 1.0e-06
    rtol = 1.0e-09

    linm, _, rts, _ = ast570
    lspan, tratio, sld = random_spans
    tension = rts * tratio
    x = np.linspace(0, lspan, nx)
    s = parabolic.stress(x, lspan, tension, sld, linm)

    sm1 = parabolic.mean_stress(lspan, tension, sld, linm)
    sm2 = np.sum(0.5 * (s[1:, :] + s[:-1, :]) * np.diff(x, axis=0), axis=0) / lspan

    assert np.allclose(sm1, sm2, atol=atol, rtol=rtol)


def test_thermal_expansion_scalar(ast570):
    """Scalar input gives a scalar; the temperature inverse recovers the change."""
    linm, _, rts, _ = ast570
    lspan, sld, alpha = 400.0, 20.0, 2.3e-05
    tension_i = 0.2 * rts
    tension_f = parabolic.thermal_expansion_tension(
        lspan, tension_i, sld, 288.0, 338.0, linm, alpha
    )
    assert np.ndim(tension_f) == 0
    assert tension_f < tension_i
    # the cable length grows by the thermal expansion
    length_i = parabolic.length(lspan, tension_i, sld, linm)
    dl = 1.0 + alpha * 50.0
    length_f = parabolic.length(lspan, tension_f, sld, linm / dl)
    assert np.isclose(length_f, length_i * dl, rtol=1e-12)
    temperature_f = parabolic.thermal_expansion_temperature(
        lspan, tension_i, tension_f, sld, 288.0, linm, alpha
    )
    assert np.isclose(temperature_f, 338.0, rtol=1e-09)


def test_thermal_expansion_array(ast570):
    linm, _, rts, _ = ast570
    lspan = np.array([200.0, 400.0, 600.0])
    tension_i = 0.2 * rts * np.ones(3)
    tension_f = parabolic.thermal_expansion_tension(
        lspan, tension_i, 0.0, 288.0, 338.0, linm, 2.3e-05
    )
    assert tension_f.shape == (3,)
    assert np.all(tension_f < tension_i)
