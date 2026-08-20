import numpy as np
import pytest

import slenderpy.future.beam.curvature as CV
import slenderpy.future.fd_utils as fdu

_N = 201
_DS = 0.05
_X = np.arange(_N) * _DS


def _finite_difference_jacobian(operator, y, h):
    """Central-difference jacobian of ``operator.value`` at ``y``."""
    jac = np.zeros((y.size, y.size))

    for k in range(y.size):
        step = np.zeros(y.size)
        step[k] = h
        jac[:, k] = (operator.value(y + step) - operator.value(y - step)) / (2.0 * h)

    return jac


@pytest.mark.parametrize(
    "approx_curvature, expected",
    [(True, CV.ApproximateCurvature), (False, CV.ExactCurvature)],
)
def test_create_selects_the_model(approx_curvature, expected):
    """Check the factory maps the flag to the matching subclass."""
    operator = CV.create(_N, _DS, approx_curvature)

    assert type(operator) is expected
    assert isinstance(operator, CV.Curvature)
    assert operator.n == _N
    assert operator.ds == _DS


def test_base_class_is_abstract():
    """Check the base class cannot be used as an operator on its own."""
    with pytest.raises(TypeError):
        CV.Curvature(_N, _DS)


def test_size_guard():
    """Check the node-count guard of the underlying scheme is not bypassed."""
    for approx_curvature in (True, False):
        with pytest.raises(ValueError):
            CV.create(2, _DS, approx_curvature)


def test_approximate_value_is_the_second_derivative():
    """Check the approximate curvature is exactly ``D2 @ y``."""
    y = np.sin(_X)
    operator = CV.ApproximateCurvature(_N, _DS)

    assert np.array_equal(operator.value(y), fdu.second_derivative(_N, _DS) @ y)


def test_approximate_jacobian_does_not_depend_on_y():
    """Check the approximate operator is linear: one constant jacobian."""
    operator = CV.ApproximateCurvature(_N, _DS)
    expected = fdu.second_derivative(_N, _DS).toarray()

    for y in (np.zeros(_N), np.sin(_X), 100.0 * _X**2):
        assert np.array_equal(operator.jacobian(y).toarray(), expected)


def test_exact_value_on_a_parabola():
    """Check the exact curvature against its closed form on a parabola.

    Both centered schemes are exact for a quadratic, so the interior nodes carry
    no truncation error and must match ``y'' / (1 + y'**2)**(3/2)``. What is left
    is the rounding of the second difference amplified by ``1 / ds**2``, of the
    order of ``eps * max|y| / (2 a ds**2)``, i.e. a few 1e-12 here; a wrong
    formula would miss by orders of magnitude more.
    """
    a = 0.7
    y = a * _X**2

    numerical = CV.ExactCurvature(_N, _DS).value(y)
    analytical = 2.0 * a / (1.0 + (2.0 * a * _X) ** 2) ** 1.5

    assert np.allclose(numerical[1:-1], analytical[1:-1], rtol=1.0e-09, atol=0.0)


def test_exact_value_matches_the_float_power_form():
    """Check the ``metric * sqrt(metric)`` form is faithful to ``metric**1.5``.

    The two are algebraically equal; this pins the rounding difference of the
    faster form to a few epsilon.
    """
    y = 3.0 * np.sin(_X)
    d1 = fdu.first_derivative(_N, _DS)
    d2 = fdu.second_derivative(_N, _DS)

    value = CV.ExactCurvature(_N, _DS).value(y)
    reference = d2 @ y * (1.0 + (d1 @ y) ** 2) ** -1.5

    assert np.allclose(value, reference, rtol=1.0e-14, atol=0.0)


def test_exact_value_tends_to_the_approximate_one_for_small_slopes():
    """Check both models agree once the slope is negligible."""
    y = 1.0e-06 * np.sin(_X)

    exact = CV.ExactCurvature(_N, _DS).value(y)
    approximate = CV.ApproximateCurvature(_N, _DS).value(y)

    assert np.allclose(exact, approximate, rtol=1.0e-10, atol=0.0)
    assert not np.array_equal(exact, approximate)


@pytest.mark.parametrize("approx_curvature", [True, False])
def test_boundary_nodes_are_empty(approx_curvature):
    """Check the curvature vanishes at the end nodes, as ``D2`` does there."""
    operator = CV.create(_N, _DS, approx_curvature)
    value = operator.value(np.sin(_X))

    assert value[0] == 0.0
    assert value[-1] == 0.0


@pytest.mark.parametrize("approx_curvature", [True, False])
def test_jacobian_against_finite_differences(approx_curvature):
    """Check the analytic jacobian against a central difference of the value."""
    y = 0.5 * np.sin(_X)
    operator = CV.create(_N, _DS, approx_curvature)

    analytic = operator.jacobian(y).toarray()
    numerical = _finite_difference_jacobian(operator, y, 1.0e-07)

    assert np.allclose(
        numerical, analytic, rtol=0.0, atol=1.0e-06 * np.abs(analytic).max()
    )


@pytest.mark.parametrize("amplitude", [0.1, 1.0, 5.0])
def test_exact_jacobian_at_large_slopes(amplitude):
    """Check the jacobian stays accurate where the metric term dominates."""
    y = amplitude * np.sin(_X)
    operator = CV.ExactCurvature(_N, _DS)

    analytic = operator.jacobian(y).toarray()
    numerical = _finite_difference_jacobian(operator, y, 1.0e-08)

    assert np.allclose(
        numerical, analytic, rtol=0.0, atol=1.0e-05 * np.abs(analytic).max()
    )
