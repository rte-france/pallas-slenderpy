import numpy as np
import pytest

from slenderpy.future.beam import bending
from slenderpy.future.beam.static import shape
from slenderpy.future.beam.static.shape import BendingModel, solve
from slenderpy.future.boundary_condition import BoundaryCondition
from slenderpy.future.components import Conductor, Span


def test_approx_curvature_constant_uses_ei_override():
    """Approximate curvature, constant model: y"" - y" = 0 on [0, 1]
    with y(0)=0, y"(0)=1, y(1)=0, y"(1)=0. Exercises the explicit ei override.

    The node count is bounded by the solver, not by accuracy: the residual is
    two composed second differences, so its rounding grows as 1/ds**4 while the
    truncation error only falls as ds**2. Past n ~ 300 the noise floor rises
    above the default tol and the solve reports a failure. n = 200 keeps a 7x
    margin there, at the price of a truncation error of 1.9e-04.
    """
    n = 200
    x = np.linspace(0.0, 1.0, n)

    left = [[1, 0, 0, 0], [0, 0, 1, 1]]
    right = [[1, 0, 0, 0], [0, 0, 1, 0]]
    bc = BoundaryCondition(4, left, right)

    conductor = Conductor(mass=1.0)
    span = Span(length=1.0, tension=1.0, boundary_conditions=bc)
    sol = solve(
        conductor,
        span,
        rhs=np.zeros(n),
        n=n,
        model=BendingModel.CONSTANT,
        ei=1.0,
        approx_curvature=True,
    )

    def exact(x):
        A = -1 / (np.exp(1) ** 2 - 1)
        B = np.exp(1) ** 2 / (np.exp(1) ** 2 - 1)
        D = -B - A
        C = -D - A * np.exp(1) - B * np.exp(-1)
        return A * np.exp(x) + B * np.exp(-x) + C * x + D

    assert np.allclose(exact(x), sol, atol=5.0e-04, rtol=1.0e-03)


def test_exact_curvature_constant_uses_ei_max_default():
    """Exact curvature, constant model: manufactured solution y = x**2 on
    [-1, 1]. The original case used tension=-5; rebuilt with tension=+5 (Span
    requires tension > 0) with the -tension*y'' rhs term flipped accordingly.
    Uses conductor.ei_max via the default (no ei override).
    """
    n = 1000
    lmin, lmax = -1.0, 1.0
    x = np.linspace(lmin, lmax, n)
    ei = 8.3
    tension = 5.0

    def rhs(x):
        return (
            8.3
            * (
                -24.0 * (1 + 4 * x**2) ** (-5.0 / 2)
                + 480 * x**2 * (1 + 4 * x**2) ** (-7.0 / 2)
            )
            - 2 * tension
        )

    left = [[1, 0, 0, lmin**2], [0, 1, 0, 2 * lmin]]
    right = [[1, 0, 0, lmax**2], [0, 1, 0, 2 * lmax]]
    bc = BoundaryCondition(4, left, right)

    conductor = Conductor(mass=1.0, ei_max=ei)
    span = Span(length=lmax - lmin, tension=tension, boundary_conditions=bc)
    sol = solve(
        conductor,
        span,
        rhs=rhs(x),
        n=n,
        model=BendingModel.CONSTANT,
        approx_curvature=False,
    )

    def exact(x):
        return x**2

    assert np.allclose(exact(x), sol, atol=1.0e-03, rtol=1.0e-09)


def test_approx_curvature_varying():
    """Approximate curvature, varying model: manufactured solution y = sin(x).

    n is capped for the same reason as in the constant case above; n = 500
    leaves a 16x margin on the residual noise floor.
    """
    n = 500
    lmin, lmax = -1.0, 3.0
    x = np.linspace(lmin, lmax, n)

    ei_min = 18.23
    ei_max = 589.64
    chi0 = 25.8
    chi_bar = (1 - ei_min / ei_max) * chi0
    H = 1485.24

    def curvature(x):
        return -np.sin(x)

    def curvature_first_derivative(x):
        return -np.cos(x)

    def curvature_second_derivative(x):
        return np.sin(x)

    def rhs(x):
        C = curvature(x)
        C1 = curvature_first_derivative(x)
        C2 = curvature_second_derivative(x)
        s = np.sign(C)
        E = np.exp(-np.abs(C) / chi_bar)
        return (
            s * ei_min * C2 * (1 - E)
            + 2 * ei_min * C1**2 * E / chi_bar
            + (ei_max * chi_bar + ei_min * C)
            * (C2 * E / chi_bar - s * C1**2 * E / chi_bar**2)
            - H * curvature(x)
        )

    def exact(x):
        return np.sin(x)

    left = [[1, 0, 0, exact(lmin)], [0, 1, 0, np.cos(lmin)]]
    right = [[1, 0, 0, exact(lmax)], [0, 1, 0, np.cos(lmax)]]
    bc = BoundaryCondition(4, left, right)

    # chi0 is derived from the flexion compliance and the span tension, so pass
    # beta_flexion = chi0 / H to keep the manufactured chi0 for this case.
    conductor = Conductor(mass=1.0, ei_min=ei_min, ei_max=ei_max, beta_flexion=chi0 / H)
    span = Span(length=lmax - lmin, tension=H, boundary_conditions=bc)
    sol = solve(
        conductor,
        span,
        rhs=rhs(x),
        n=n,
        model=BendingModel.VARYING,
        approx_curvature=True,
    )

    assert np.allclose(exact(x), sol, atol=3.0e-03, rtol=1.0e-09)


def test_exact_curvature_varying():
    """Exact curvature, varying model: manufactured solution y = cosh(x)."""
    n = 1000
    lmin, lmax = -1.0, 3.0
    x = np.linspace(lmin, lmax, n)

    ei_min = 253.2
    ei_max = 1234.9
    chi0 = 12.4
    chi_bar = (1 - ei_min / ei_max) * chi0
    H = 1587.2

    def curvature(x):
        return 1.0 / np.cosh(x) ** 2

    def curvature_first_derivative(x):
        return -2 * np.sinh(x) / np.cosh(x) ** 3

    def curvature_second_derivative(x):
        return -2 / np.cosh(x) ** 2 + 6.0 * np.sinh(x) ** 2 / np.cosh(x) ** 4

    def rhs(x):
        C = curvature(x)
        C1 = curvature_first_derivative(x)
        C2 = curvature_second_derivative(x)
        E = np.exp(-C / chi_bar)
        return (
            ei_min * C2 * (1 - E)
            + 2 * ei_min * C1**2 * E / chi_bar
            + (ei_max * chi_bar + ei_min * C)
            * (C2 * E / chi_bar - C1**2 * E / chi_bar**2)
            - H * np.cosh(x)
        )

    def exact(x):
        return np.cosh(x)

    left = [[1, 0, 0, np.cosh(lmin)], [0, 1, 0, np.sinh(lmin)]]
    right = [[1, 0, 0, np.cosh(lmax)], [0, 1, 0, np.sinh(lmax)]]
    bc = BoundaryCondition(4, left, right)

    # chi0 is derived from the flexion compliance and the span tension, so pass
    # beta_flexion = chi0 / H to keep the manufactured chi0 for this case.
    conductor = Conductor(mass=1.0, ei_min=ei_min, ei_max=ei_max, beta_flexion=chi0 / H)
    span = Span(length=lmax - lmin, tension=H, boundary_conditions=bc)
    sol = solve(
        conductor,
        span,
        rhs=rhs(x),
        n=n,
        model=BendingModel.VARYING,
        approx_curvature=False,
    )

    assert np.allclose(exact(x), sol, atol=1.0e-03, rtol=1.0e-03)


def test_string_model_matches_enum():
    """The model argument accepts the enum member or its string value."""
    n = 200
    x = np.linspace(0.0, 1.0, n)
    bc = BoundaryCondition(4)
    conductor = Conductor(mass=1.0, ei_max=10.0)
    span = Span(length=1.0, tension=100.0, boundary_conditions=bc)
    rhs = np.ones(n)

    via_str = solve(conductor, span, rhs=rhs, n=n, model="constant")
    via_enum = solve(conductor, span, rhs=rhs, n=n, model=BendingModel.CONSTANT)
    assert np.allclose(via_str, via_enum)
    # x is only used to size rhs; keep the linter happy about the shared setup.
    assert x.shape == (n,)


def test_missing_boundary_conditions_raises():
    conductor = Conductor(mass=1.0, ei_max=10.0)
    span = Span(length=1.0, tension=100.0)  # boundary_conditions defaults to None
    with pytest.raises(ValueError):
        solve(conductor, span, rhs=np.zeros(10), n=10, model=BendingModel.CONSTANT)


def test_varying_without_beta_flexion_raises():
    bc = BoundaryCondition(4)
    conductor = Conductor(mass=1.0, ei_min=1.0, ei_max=10.0)  # beta_flexion is None
    span = Span(length=1.0, tension=100.0, boundary_conditions=bc)
    with pytest.raises(ValueError):
        solve(conductor, span, rhs=np.zeros(10), n=10, model=BendingModel.VARYING)


def test_constant_without_any_ei_raises():
    bc = BoundaryCondition(4)
    conductor = Conductor(mass=1.0)  # ei_max is None and no ei override
    span = Span(length=1.0, tension=100.0, boundary_conditions=bc)
    with pytest.raises(ValueError):
        solve(conductor, span, rhs=np.zeros(10), n=10, model=BendingModel.CONSTANT)


def test_rhs_length_mismatch_raises():
    bc = BoundaryCondition(4)
    conductor = Conductor(mass=1.0, ei_max=10.0)
    span = Span(length=1.0, tension=100.0, boundary_conditions=bc)
    with pytest.raises(ValueError):
        solve(conductor, span, rhs=np.zeros(5), n=10, model=BendingModel.CONSTANT)


def test_bending_model_is_reexported():
    """The model selector lives in `bending`; `shape` re-exports it unchanged.

    The constitutive laws themselves are covered by test_future-beam-bending.py
    and test_future-beam-curvature.py; here only the wiring is checked.
    """
    assert shape.BendingModel is bending.BendingModel
