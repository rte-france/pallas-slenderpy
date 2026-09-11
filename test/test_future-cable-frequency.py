import numpy as np
import pytest

from slenderpy.future.cable import frequency
from slenderpy.future.cable.frequency import FrequencyMethod, natural
from slenderpy.future.cable.static.catenary import length as catenary_length

# ASTER570-like conductor on a representative span.
LSPAN = 400.0
TENSION = 30000.0
SLD = 5.0
LINM = 1.571
AXS = 3.653e07


def _taut_f0(lspan, tension, sld, linm):
    """Reference taut-string fundamental frequency, computed from first principles."""
    length = np.sqrt(lspan**2 + sld**2)
    return 0.5 * np.sqrt(tension / (linm * length**2))


def test_natural_taut_matches_formula():
    expected = _taut_f0(LSPAN, TENSION, SLD, LINM)
    assert natural(LSPAN, TENSION, SLD, LINM, method="taut") == pytest.approx(expected)


def test_natural_default_is_taut():
    assert natural(LSPAN, TENSION, SLD, LINM) == pytest.approx(
        natural(LSPAN, TENSION, SLD, LINM, method=FrequencyMethod.TAUT)
    )


def test_natural_string_and_enum_equivalent():
    for m in ("taut", "parabolic", "catenary"):
        via_str = natural(LSPAN, TENSION, SLD, LINM, method=m)
        via_enum = natural(LSPAN, TENSION, SLD, LINM, method=FrequencyMethod(m))
        assert via_str == pytest.approx(via_enum)


def test_natural_catenary_matches_length_model():
    length = catenary_length(LSPAN, TENSION, SLD, LINM)
    expected = 0.5 * np.sqrt(TENSION / (LINM * length**2))
    assert natural(LSPAN, TENSION, SLD, LINM, method="catenary") == pytest.approx(
        expected
    )


def test_natural_catenary_below_taut():
    # The catenary length exceeds the straight chord used by the taut model, so
    # its natural frequency is lower.
    f_catenary = natural(LSPAN, TENSION, SLD, LINM, method="catenary")
    f_taut = natural(LSPAN, TENSION, SLD, LINM, method="taut")
    assert f_catenary < f_taut


def test_natural_invalid_method_raises():
    with pytest.raises(ValueError):
        natural(LSPAN, TENSION, SLD, LINM, method="bogus")


def test_natural_all_methods_positive():
    for m in (
        FrequencyMethod.TAUT,
        FrequencyMethod.PARABOLIC,
        FrequencyMethod.CATENARY,
        FrequencyMethod.NLEQ,
    ):
        f = natural(LSPAN, TENSION, SLD, LINM, axs=AXS, method=m)
        assert np.isfinite(f)
        assert f > 0


def test_op_frequencies_are_harmonics():
    n = 5
    fq = frequency._op_frequencies(LSPAN, TENSION, SLD, LINM, AXS, n=n)
    assert fq.shape == (n,)
    f0 = _taut_f0(LSPAN, TENSION, SLD, LINM)
    assert fq == pytest.approx(f0 * np.arange(1, n + 1))


def test_ip_frequencies_interleave_harmonics():
    # In-plane frequencies interleave the symmetric Irvine modes (even indices)
    # with the antisymmetric harmonics (odd indices), so they are not globally
    # monotonic. The odd-index entries are left as the taut harmonics 2*f0, 4*f0.
    n = 5
    fq = frequency._ip_frequencies(LSPAN, TENSION, SLD, LINM, AXS, n=n)
    assert fq.shape == (n,)
    assert np.all(np.isfinite(fq))
    assert np.all(fq > 0)
    f0 = _taut_f0(LSPAN, TENSION, SLD, LINM)
    assert fq[1] == pytest.approx(2 * f0)
    assert fq[3] == pytest.approx(4 * f0)


# --- irvine_number ---------------------------------------------------------


def _canonical_irvine2(lspan, tension, sld, linm, axs, g=9.81):
    """Irvine number squared from the virtual-length definition.

    lambda^2 = (axs/tension) * (lspan/a)^2 * lspan/Le with
    Le = integral_0^lspan (1 + z'^2)^(3/2) dx, integrated numerically. This is
    the reference the approximate closed form must reproduce.
    """
    from slenderpy.future.cable.static import catenary

    a = catenary._mechparam(tension, linm, g=g)
    length = catenary_length(lspan, tension, sld, linm, g=g)
    xm = 0.5 * (a * catenary._qfactor(length, sld) - lspan)
    x = np.linspace(0.0, lspan, 200001)
    slope = np.sinh((x + xm) / a)
    virtual = np.trapezoid((1.0 + slope**2) ** 1.5, x)
    return (axs / tension) * (lspan / a) ** 2 * lspan / virtual


@pytest.mark.parametrize("lspan", [100.0, 200.0, 400.0, 800.0])
@pytest.mark.parametrize("sld", [0.0, 5.0, 20.0])
def test_irvine_number_matches_the_virtual_length_definition(lspan, sld):
    """Guards the spurious lspan factor: with it, this is off by sqrt(lspan)."""
    if abs(sld) / lspan > 0.05:
        pytest.skip("outside the shallow-span range the closed form assumes")
    got = frequency.irvine_number(lspan, TENSION, sld, LINM, AXS) ** 2
    expected = _canonical_irvine2(lspan, TENSION, sld, LINM, AXS)
    assert got == pytest.approx(expected, rel=0.01)


# measured departures at lspan=400: +1.5%, +6.1%, +9.6%, +39.9%
@pytest.mark.parametrize(
    "ratio, tolerance", [(0.1, 0.02), (0.2, 0.07), (0.25, 0.10), (0.5, 0.40)]
)
def test_irvine_number_degrades_on_steep_spans(ratio, tolerance):
    """Document the error outside the shallow range instead of hiding it.

    The closed form loses its sld dependence, so it drifts from the virtual
    length definition as the span steepens. Irvine's theory does not hold there
    either; this pins the size of the departure so a future change is visible.
    """
    lspan = 400.0
    sld = ratio * lspan
    got = frequency.irvine_number(lspan, TENSION, sld, LINM, AXS) ** 2
    expected = _canonical_irvine2(lspan, TENSION, sld, LINM, AXS)
    assert got == pytest.approx(expected, rel=tolerance)
    assert got > expected  # the closed form always overestimates here


@pytest.mark.parametrize("sld", [-100.0, -30.0, 0.0, 30.0, 100.0])
def test_irvine_number_is_finite_on_inclined_spans(sld):
    """parabolic.sag collapses to 0 for an inclined span; max_chord does not."""
    lm = frequency.irvine_number(LSPAN, TENSION, sld, LINM, AXS)
    assert np.isfinite(lm)
    assert lm > 0.0


def test_irvine_number_is_dimensionless():
    """Scaling the span with the sag ratio held fixed must not change lambda."""
    lm1 = frequency.irvine_number(400.0, TENSION, 0.0, LINM, AXS)
    # halving the span and the linear mass keeps lspan/a, hence the sag ratio
    lm2 = frequency.irvine_number(200.0, TENSION, 0.0, 2.0 * LINM, AXS)
    assert lm1 == pytest.approx(lm2, rel=1.0e-12)


def test_irvine_frequencies_are_roots_of_the_transcendental_equation():
    """Each returned frequency must actually solve the Irvine equation."""
    n = 4
    lm2 = frequency.irvine_number(LSPAN, TENSION, SLD, LINM, AXS) ** 2
    f0 = _taut_f0(LSPAN, TENSION, SLD, LINM)
    fq = frequency._irvine_frequencies(LSPAN, TENSION, SLD, LINM, AXS, n=n)
    for f in fq:
        x = np.pi * f / f0
        residual = np.tan(0.5 * x) - 0.5 * x + 0.5 * x**3 / lm2
        assert abs(residual) < 1.0e-06


def test_irvine_frequencies_are_bracketed_and_increasing():
    n = 4
    fq = frequency._irvine_frequencies(LSPAN, TENSION, SLD, LINM, AXS, n=n)
    f0 = _taut_f0(LSPAN, TENSION, SLD, LINM)
    x = np.pi * fq / f0
    assert np.all(np.diff(x) > 0.0)
    for k in range(n):
        assert (2 * k + 1) * np.pi < x[k] < (2 * k + 3) * np.pi
