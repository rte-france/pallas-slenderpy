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
