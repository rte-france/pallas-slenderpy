"""Tests for slenderpy.boundary_condition."""

import numpy as np
import pytest

from slenderpy.boundary_condition import BoundaryCondition, clamped, hinged


def test_defaults_are_dirichlet():
    assert BoundaryCondition(2).left == ((1.0, 0.0, 0.0, 0.0),)
    bc = BoundaryCondition(4)
    assert bc.right == ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))


def test_helpers():
    assert hinged(1.0, 2.0).left[0][3] == 1.0
    assert hinged().left[1] == (0.0, 0.0, 1.0, 0.0)
    assert clamped(1.0, 2.0).right[0][3] == 2.0
    assert clamped().right[1] == (0.0, 1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "order, left, right, error",
    [
        (3, None, None, ValueError),
        (2, "bad", None, TypeError),
        (2, ((1, 0, 0, 0), (0, 1, 0, 0)), None, ValueError),
        (2, ((1, 0, 0),), None, ValueError),
        (2, ((0, 0, 0, 0),), None, ValueError),
        (2, None, ((0, 0, 0, 0),), ValueError),
        (4, None, "bad", TypeError),
        (4, ((1, 0, 0, 0),), None, ValueError),
        (4, ((1, 0, 0, 0), (0, 1, 0)), None, ValueError),
        (4, ((1, 0, 0, 0), (2, 0, 0, 0)), None, ValueError),
        (4, None, ((0, 1, 0, 0), (0, 2, 0, 0)), ValueError),
    ],
)
def test_invalid_inputs_raise(order, left, right, error):
    with pytest.raises(error):
        BoundaryCondition(order, left=left, right=right)


def test_compute_fills_the_boundary_rows():
    n, ds = 6, 0.5
    matrix, rhs = hinged(1.0, 2.0).compute(n, ds)
    assert matrix[0, 0] == 1.0
    assert matrix[1, :4].toarray().ravel() == pytest.approx([8.0, -20.0, 16.0, -4.0])
    assert matrix[-1, -1] == 1.0
    assert rhs == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 2.0])


def test_update_rhs_order_and_positions():
    # each callable is evaluated at its bound; the order is
    # (left[0], left[1], right[1], right[0])
    def make(k):
        return lambda x, t: 10.0 * k + x + t

    bc = BoundaryCondition(4, dynamic_values=[make(k) for k in range(4)])
    x = np.linspace(0.0, 5.0, 6)
    rhs = bc.update_rhs(6, x, 0.5)
    assert rhs == pytest.approx([0.5, 10.5, 0.0, 0.0, 25.5, 35.5])


def test_update_rhs_order_2_uses_first_and_last():
    bc = BoundaryCondition(
        2, dynamic_values=[lambda x, t: 1.0, None, None, lambda x, t: 2.0]
    )
    rhs = bc.update_rhs(4, np.linspace(0.0, 1.0, 4), 0.0)
    assert rhs == pytest.approx([1.0, 0.0, 0.0, 2.0])


def test_hinged_curvature_rows_use_the_one_sided_stencil():
    # the curvature of the second relation is taken at the bound node, not
    # one node inside
    ds = 0.5
    dense = hinged().compute(7, ds)[0].toarray()
    np.testing.assert_allclose(dense[1], np.array([2, -5, 4, -1, 0, 0, 0]) / ds**2)
    np.testing.assert_allclose(dense[-2], np.array([0, 0, 0, -1, 4, -5, 2]) / ds**2)
    np.testing.assert_allclose(dense[0], [1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(dense[-1], [0, 0, 0, 0, 0, 0, 1])


def test_mixed_second_relation_keeps_its_slope_term():
    ds = 0.5
    bc = BoundaryCondition(
        4,
        left=((1, 0, 0, 0), (0, 1, 1, 0)),
        right=((1, 0, 0, 0), (0, 1, 1, 0)),
    )
    dense = bc.compute(7, ds)[0].toarray()
    np.testing.assert_allclose(
        dense[1],
        np.array([-1, 1, 0, 0, 0, 0, 0]) / ds
        + np.array([2, -5, 4, -1, 0, 0, 0]) / ds**2,
    )
    np.testing.assert_allclose(
        dense[-2],
        np.array([0, 0, 0, 0, 0, -1, 1]) / ds
        + np.array([0, 0, 0, -1, 4, -5, 2]) / ds**2,
    )
