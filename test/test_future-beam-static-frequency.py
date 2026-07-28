import numpy as np
import pytest

from slenderpy.future.beam.static import frequency

# Representative beam with a non-negligible bending contribution (ep ~ 0.1).
LENGTH = 10.0
TENSION = 1000.0
MASS = 1.0
EI = 10000.0
N = 5


def test_natural_frequencies_matches_string_formula():
    fq = frequency.natural_frequencies(LENGTH, TENSION, MASS, N)
    expected = 0.5 * np.arange(1, N + 1) / LENGTH * np.sqrt(TENSION / MASS)
    assert fq.shape == (N,)
    assert fq == pytest.approx(expected)


def test_natural_frequency_is_first_string_mode():
    f = frequency.natural_frequency(LENGTH, TENSION, MASS)
    first = frequency.natural_frequencies(LENGTH, TENSION, MASS, 1)[0]
    assert f == pytest.approx(first)


def test_hinged_reduces_to_string_when_ei_zero():
    string = frequency.natural_frequencies(LENGTH, TENSION, MASS, N)
    pinned = frequency.natural_frequencies_hinged(LENGTH, TENSION, MASS, 0.0, N)
    assert pinned == pytest.approx(string)


def test_hinged_above_string_for_positive_ei():
    string = frequency.natural_frequencies(LENGTH, TENSION, MASS, N)
    pinned = frequency.natural_frequencies_hinged(LENGTH, TENSION, MASS, EI, N)
    assert pinned.shape == (N,)
    assert np.all(pinned > string)


def test_clamped_shape_finite_positive():
    clamped = frequency.natural_frequencies_clamped(LENGTH, TENSION, MASS, EI, N)
    assert clamped.shape == (N,)
    assert np.all(np.isfinite(clamped))
    assert np.all(clamped > 0)


def test_stiffness_ordering_clamped_pinned_string():
    # Clamped ends are stiffer than pinned ends, which are stiffer than a string,
    # so the natural frequencies order accordingly for a positive bending stiffness.
    string = frequency.natural_frequencies(LENGTH, TENSION, MASS, N)
    pinned = frequency.natural_frequencies_hinged(LENGTH, TENSION, MASS, EI, N)
    clamped = frequency.natural_frequencies_clamped(LENGTH, TENSION, MASS, EI, N)
    assert np.all(pinned > string)
    assert np.all(clamped >= pinned)
