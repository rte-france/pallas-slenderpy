import numpy as np
import pytest

import slenderpy.future.beam.bending as BD
from slenderpy.future.components import Conductor, Span

_EI = 1500.0
_EI_MIN = 100.0
_EI_MAX = 4000.0
_BETA = 1.0e-06
_TENSION = 30000.0

_SPAN = Span(length=400.0, tension=_TENSION)
_CONSTANT_CONDUCTOR = Conductor(mass=1.6, diameter=0.03, ei_max=_EI)
_VARYING_CONDUCTOR = Conductor(
    mass=1.6, diameter=0.03, ei_min=_EI_MIN, ei_max=_EI_MAX, beta_flexion=_BETA
)


def _varying_law():
    """The varying law as :func:`create` would build it from the fixtures."""
    return BD.VaryingBending(_EI_MIN, _EI_MAX, _BETA * _TENSION)


@pytest.mark.parametrize("model", [BD.BendingModel.CONSTANT, "constant"])
def test_create_selects_the_constant_model(model):
    """Check the factory accepts the enum member and its string value."""
    law = BD.create(_CONSTANT_CONDUCTOR, _SPAN, model)

    assert type(law) is BD.ConstantBending
    assert isinstance(law, BD.Bending)
    assert law.ei_linear == _EI


@pytest.mark.parametrize("model", [BD.BendingModel.VARYING, "varying"])
def test_create_selects_the_varying_model(model):
    """Check the factory accepts the enum member and its string value."""
    law = BD.create(_VARYING_CONDUCTOR, _SPAN, model)

    assert type(law) is BD.VaryingBending
    assert isinstance(law, BD.Bending)
    assert law.ei_linear == _EI_MIN


def test_create_derives_chi0_from_the_span_tension():
    """Check the reference curvature is ``beta_flexion * tension``."""
    law = BD.create(_VARYING_CONDUCTOR, _SPAN, BD.BendingModel.VARYING)

    assert law.chi0 == _BETA * _TENSION
    assert law.chi_bar == (1.0 - _EI_MIN / _EI_MAX) * _BETA * _TENSION


def test_create_ei_overrides_the_conductor_stiffness():
    """Check the constant model prefers an explicit ``ei`` over the conductor."""
    law = BD.create(_CONSTANT_CONDUCTOR, _SPAN, BD.BendingModel.CONSTANT, ei=42.0)

    assert law.ei_linear == 42.0


def test_create_rejects_ei_with_the_varying_model():
    """Check an ``ei`` that would be silently ignored is refused instead."""
    with pytest.raises(ValueError, match="constant model"):
        BD.create(_VARYING_CONDUCTOR, _SPAN, BD.BendingModel.VARYING, ei=42.0)


def test_create_rejects_a_constant_model_without_stiffness():
    """Check a conductor with no ``ei_max`` and no override is refused."""
    with pytest.raises(ValueError, match="constant model"):
        BD.create(Conductor(mass=1.6), _SPAN, BD.BendingModel.CONSTANT)


@pytest.mark.parametrize("missing", ["ei_min", "ei_max", "beta_flexion"])
def test_create_rejects_a_varying_model_with_a_missing_field(missing):
    """Check each field the varying model needs is checked for."""
    fields = {"ei_min": _EI_MIN, "ei_max": _EI_MAX, "beta_flexion": _BETA}
    fields[missing] = None

    with pytest.raises(ValueError, match="varying model"):
        BD.create(Conductor(mass=1.6, **fields), _SPAN, BD.BendingModel.VARYING)


def test_create_rejects_an_unknown_model():
    """Check an unsupported model name does not fall through to a default."""
    with pytest.raises(ValueError):
        BD.create(_CONSTANT_CONDUCTOR, _SPAN, "quadratic")


def test_base_class_is_abstract():
    """Check the base class cannot be used as a law on its own."""
    with pytest.raises(TypeError):
        BD.Bending(_EI)


def test_constant_moment_is_linear():
    """Check the constant law is exactly ``ei * curvature``."""
    curvature = np.linspace(-1.0, 1.0, 101)
    law = BD.ConstantBending(_EI)

    assert np.array_equal(law.moment(curvature), _EI * curvature)


def test_constant_tangent_is_the_stiffness_everywhere():
    """Check the constant tangent is an array of ``ei``, whatever the curvature."""
    curvature = np.linspace(-1.0, 1.0, 101)
    law = BD.ConstantBending(_EI)
    tangent = law.tangent(curvature)

    assert tangent.shape == curvature.shape
    assert np.all(tangent == _EI)


def test_constant_rejects_a_non_positive_stiffness():
    """Check a null or negative stiffness is refused at construction."""
    for ei in (0.0, -1.0):
        with pytest.raises(ValueError, match="ei must be"):
            BD.ConstantBending(ei)


def test_varying_rejects_equal_stiffnesses():
    """Check ``ei_min == ei_max`` is refused rather than yielding ``nan``.

    ``Conductor`` allows it, but it collapses ``chi_bar`` to zero, and the law
    then evaluates ``0 * (1 - exp(-0/0))`` at zero curvature. Zero curvature is
    not an edge case here: the second-derivative scheme leaves its first and
    last rows empty, so it happens at both ends on every single call.
    """
    with pytest.raises(ValueError, match="ei_min < ei_max"):
        BD.VaryingBending(_EI_MAX, _EI_MAX, _BETA * _TENSION)


@pytest.mark.parametrize("chi0", [0.0, -1.0e-03])
def test_varying_rejects_a_non_positive_reference_curvature(chi0):
    """Check a null or negative ``chi0`` is refused at construction."""
    with pytest.raises(ValueError, match="chi0 must be"):
        BD.VaryingBending(_EI_MIN, _EI_MAX, chi0)


def test_varying_moment_is_odd():
    """Check the moment is an odd function of the curvature, and zero at zero."""
    curvature = np.geomspace(1.0e-08, 10.0, 500)
    law = _varying_law()

    assert np.array_equal(law.moment(curvature), -law.moment(-curvature))
    assert law.moment(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]


def test_varying_tangent_is_even():
    """Check the tangent is an even function, as the derivative of an odd law."""
    curvature = np.geomspace(1.0e-08, 10.0, 500)
    law = _varying_law()

    assert np.array_equal(law.tangent(curvature), law.tangent(-curvature))


def test_varying_tangent_limits():
    """Check the tangent runs from ``ei_max`` at rest down to ``ei_min``."""
    law = _varying_law()

    assert law.tangent(np.zeros(1))[0] == _EI_MAX
    assert law.tangent(np.array([1.0e04 * law.chi_bar]))[0] == pytest.approx(_EI_MIN)


def test_varying_moment_slope_at_rest():
    """Check the moment leaves the origin with the slope ``ei_max``."""
    law = _varying_law()
    curvature = 1.0e-04 * law.chi_bar

    assert law.moment(np.array([curvature]))[0] == pytest.approx(
        _EI_MAX * curvature, rel=1.0e-03
    )


def test_varying_moment_asymptote():
    """Check the softened branch is ``ei_min * c + ei_max * chi_bar``."""
    law = _varying_law()
    curvature = np.array([1.0e03, 1.0e04]) * law.chi_bar

    expected = _EI_MIN * curvature + _EI_MAX * law.chi_bar

    assert np.allclose(law.moment(curvature), expected, rtol=1.0e-12, atol=0.0)


def test_varying_tangent_against_finite_differences():
    """Check the tangent is the derivative of the moment it is paired with.

    The comparison is limited by the rounding of the difference quotient, about
    ``eps * max|M| / h`` relative to the tangent; a wrong derivative would miss
    by orders of magnitude more.
    """
    curvature = np.linspace(-1.0, 1.0, 2001)
    law = _varying_law()
    h = 1.0e-09

    numerical = (law.moment(curvature + h) - law.moment(curvature - h)) / (2.0 * h)

    assert np.allclose(numerical, law.tangent(curvature), rtol=1.0e-06, atol=0.0)


@pytest.mark.parametrize(
    "law, expected",
    [
        (BD.ConstantBending(_EI), _EI),
        (BD.VaryingBending(_EI_MIN, _EI_MAX, _BETA * _TENSION), _EI_MIN),
    ],
)
def test_ei_linear_is_the_stiffness_of_the_linear_part(law, expected):
    """Check the stiffness the ``D4`` term is assembled with."""
    assert law.ei_linear == expected
