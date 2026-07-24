"""Unit tests for the core domain model (no solvers involved)."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from slenderpy.future.stockbridge import (
    Clamp,
    ClampParameters,
    Mass,
    MassParameters,
    MessengerCableParameters,
    Side,
)

# --- Side -----------------------------------------------------------------


def test_side_left_epsilon_is_minus_one():
    assert Side.LEFT.epsilon == -1


def test_side_right_epsilon_is_plus_one():
    assert Side.RIGHT.epsilon == +1


def test_side_string_value():
    assert Side.LEFT.value == "left"
    assert Side.RIGHT.value == "right"


# --- Parameter dataclasses ------------------------------------------------


def test_mass_parameters_frozen():
    p = MassParameters(0.1, 0.01, 1.0, 0.001)
    with pytest.raises(FrozenInstanceError):
        p.mass = 2.0  # type: ignore[misc]


def test_messenger_cable_parameters_frozen():
    p = MessengerCableParameters(10, 0.1, 0.1, 40, 25, 5, 2.5, 0.15, 0.03)
    with pytest.raises(FrozenInstanceError):
        p.nb_space_points = 20  # type: ignore[misc]


def test_clamp_parameters_frozen():
    p = ClampParameters(0.5, 0.0025, 0.03)
    with pytest.raises(FrozenInstanceError):
        p.mass = 1.0  # type: ignore[misc]


# --- Clamp ----------------------------------------------------------------


def test_clamp_attributes_match_parameters(clamp_params):
    c = Clamp(clamp_params)
    assert c.mass == clamp_params.mass
    assert c.moment_of_inertia == clamp_params.moment_of_inertia
    assert c.half_length == clamp_params.half_length


def test_clamp_zero_input_zero_output(clamp):
    fc, mc = clamp.compute_forces_at_clamp(0, 0, 0, 0, 0, 0, 0, 0)
    assert fc == 0
    assert mc == 0


def test_clamp_compute_forces_known_values(clamp):
    # Newton on the clamp: Fc = m*a - (F1 + F2);
    # Mc = I*alpha + M2 - M1 + F2*(l2+hl) - F1*(l1+hl).
    F1, F2, M1, M2, l1, l2, a, alpha = 1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 5.0, 6.0
    fc, mc = clamp.compute_forces_at_clamp(F1, F2, M1, M2, l1, l2, a, alpha)
    assert fc == pytest.approx(clamp.mass * a - (F1 + F2))
    expected_mc = (
        clamp.moment_of_inertia * alpha
        + M2
        - M1
        + F2 * (l2 + clamp.half_length)
        - F1 * (l1 + clamp.half_length)
    )
    assert mc == pytest.approx(expected_mc)


# --- Mass -----------------------------------------------------------------


def test_mass_attributes_populated(mass_right, mass_params, cable_params):
    assert mass_right.mass == mass_params.mass
    assert mass_right.length_to_clamp == mass_params.length_to_clamp
    assert mass_right.length_to_centroid == mass_params.length_to_centroid
    assert mass_right.moment_of_inertia == mass_params.moment_of_inertia
    assert mass_right.nb_space_points == cable_params.nb_space_points
    assert mass_right.nb_unknowns == 6 + 2 * cable_params.nb_space_points
    assert mass_right.x.shape == (cable_params.nb_space_points,)


def test_mass_side_sets_epsilon(mass_right, mass_left):
    assert mass_right.epsilon == +1
    assert mass_left.epsilon == -1
    assert mass_right.side is Side.RIGHT
    assert mass_left.side is Side.LEFT


def test_mass_piecewise_cable_arrays_three_regions(mass_params):
    """The cable is split in three regions: boundary | cable | boundary."""
    cable = MessengerCableParameters(
        nb_space_points=20,
        ratio_boundary1=0.2,
        ratio_boundary2=0.2,
        ei_max_boundary=999.0,
        ei_max_cable=1.0,
        ei_min_boundary=999.0,
        ei_min_cable=1.0,
        chi0_boundary=0.9,
        chi0_cable=0.1,
    )
    m = Mass(mass_params, cable, Side.RIGHT)
    # First node and last node are in boundary region.
    assert m.ei_max[0] == 999.0
    assert m.ei_max[-1] == 999.0
    # Middle node is in cable region.
    assert m.ei_max[len(m.ei_max) // 2] == 1.0
    # All three arrays have shape (n,) and are piecewise.
    assert m.ei_max.shape == (20,)
    assert m.ei_min.shape == (20,)
    assert m.chi0.shape == (20,)
    # Three distinct regions in chi0.
    assert set(np.unique(m.chi0).tolist()) == {0.1, 0.9}


def test_mass_matrix_symmetric(mass_right):
    M = mass_right.mass_matrix
    assert M.shape == (2, 2)
    assert np.allclose(M, M.T)


def test_mass_matrix_inverse(mass_right):
    M = mass_right.mass_matrix
    assert np.allclose(M @ mass_right.mass_matrix_inv, np.eye(2))


def test_mass_matrix_analytic(mass_right, mass_params):
    """The 2x2 mass matrix follows the rigid-body-on-arm formula."""
    coef = mass_params.mass * mass_params.length_to_centroid
    expected = np.array(
        [
            [mass_params.mass, -coef],
            [
                -coef,
                mass_params.moment_of_inertia + mass_params.length_to_centroid * coef,
            ],
        ]
    )
    assert np.allclose(mass_right.mass_matrix, expected)


def test_mass_compute_exterior_forces_sign_flips_with_side(mass_right, mass_left):
    """The rotational term flips sign between left and right (epsilon)."""
    half_length, vert_acc, rot_acc = 0.03, 1.0, 2.0
    fr, mr_ = mass_right.compute_exterior_forces(half_length, vert_acc, rot_acc)
    fl, ml_ = mass_left.compute_exterior_forces(half_length, vert_acc, rot_acc)
    # Vertical-only contribution is identical (does not depend on epsilon).
    f_vert_only = -mass_right.mass * vert_acc
    m_vert_only = mass_right.mass * mass_right.length_to_centroid * vert_acc
    # The asymmetric piece reverses sign.
    asym_f = (fr - f_vert_only) + (fl - f_vert_only)
    asym_m = (mr_ - m_vert_only) + (ml_ - m_vert_only)
    assert asym_f == pytest.approx(0.0)
    assert asym_m == pytest.approx(0.0)


def test_mass_build_matrix_acceleration_imposed_shape(mass_right, cable_params):
    A = mass_right.build_matrix_acceleration_imposed(
        np.zeros(cable_params.nb_space_points), dt=1e-3
    )
    n = cable_params.nb_space_points
    assert A.shape == (6 + 2 * n, 6 + 2 * n)


# --- Stockbridge ----------------------------------------------------------


def test_stockbridge_default_K_C_none(sb):
    assert sb.K is None
    assert sb.C is None


def test_stockbridge_global_mass_matrix_is_block_diagonal(sb):
    """The 4x4 inverse should split into two 2x2 inverses on the diagonal."""
    inv = sb.mass_matrix_inv
    assert inv.shape == (4, 4)
    # Off-diagonal 2x2 blocks must be zero.
    assert np.allclose(inv[0:2, 2:4], 0)
    assert np.allclose(inv[2:4, 0:2], 0)
    # Diagonal blocks invert each Mass.mass_matrix.
    assert np.allclose(inv[0:2, 0:2], sb.mass_right.mass_matrix_inv)
    assert np.allclose(inv[2:4, 2:4], sb.mass_left.mass_matrix_inv)


def test_stockbridge_ab_signs(sb):
    """Vector b carries the epsilon sign, so right and left blocks have opposite signs."""
    a, b = sb.ab
    assert a.shape == (4,)
    assert b.shape == (4,)
    # b = [m*eps*bc, -m*eg*eps*bc, ...] -> right (eps=+1) and left (eps=-1) flip sign.
    assert np.sign(b[0]) == +1  # right
    assert np.sign(b[2]) == -1  # left
