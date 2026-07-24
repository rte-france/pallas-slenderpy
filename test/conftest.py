import numpy as np
import pytest

from slenderpy.future.stockbridge import (
    Clamp,
    ClampParameters,
    Mass,
    MassParameters,
    MessengerCableParameters,
    Side,
    Stockbridge,
)


@pytest.fixture(scope="function")
def ast570():
    """Get conductor properties that matches ASTER570."""
    linm = 1.571
    axs = 3.653e07
    rts = 1.853e05
    ei = 2155.0
    return linm, axs, rts, ei


@pytest.fixture(scope="function")
def random_spans():
    """Generate 999 random spans with fixed seed."""
    lsmin = 50
    lsmax = 1000
    hrmin = 0.05
    hrmax = 0.5
    slmin = 0.0
    slmax = 0.5

    np.random.seed(1234)
    n = 999

    lspan = lsmin + (lsmax - lsmin) * np.random.rand(n)
    tratio = hrmin + (hrmax - hrmin) * np.random.rand(n)
    sld = lspan * (slmin + (slmax - slmin) * np.random.rand(n))

    return lspan, tratio, sld


@pytest.fixture
def cable_params() -> MessengerCableParameters:
    """Small messenger cable so per-step solves stay fast in tests."""
    return MessengerCableParameters(
        nb_space_points=5,
        ratio_boundary1=0.2,
        ratio_boundary2=0.2,
        ei_max_boundary=40.0,
        ei_max_cable=25.0,
        ei_min_boundary=5.0,
        ei_min_cable=2.5,
        chi0_boundary=15e-2,
        chi0_cable=3e-2,
    )


@pytest.fixture
def mass_params() -> MassParameters:
    return MassParameters(
        length_to_clamp=0.1875,
        length_to_centroid=0.0325,
        mass=0.856,
        moment_of_inertia=0.001814,
    )


@pytest.fixture
def mass_params_no_offset() -> MassParameters:
    """Mass with zero centroid offset (used by analytic-mode tests)."""
    return MassParameters(
        length_to_clamp=0.1875,
        length_to_centroid=0.0,
        mass=0.856,
        moment_of_inertia=0.001814,
    )


@pytest.fixture
def clamp_params() -> ClampParameters:
    return ClampParameters(mass=0.5, moment_of_inertia=0.0025, half_length=0.03)


@pytest.fixture
def mass_right(mass_params, cable_params) -> Mass:
    return Mass(mass_params, cable_params, Side.RIGHT)


@pytest.fixture
def mass_left(mass_params, cable_params) -> Mass:
    return Mass(mass_params, cable_params, Side.LEFT)


@pytest.fixture
def clamp(clamp_params) -> Clamp:
    return Clamp(clamp_params)


@pytest.fixture
def sb(clamp_params, mass_params, cable_params) -> Stockbridge:
    """Default stockbridge model, no linearised K/C."""
    return Stockbridge(
        clamp_params, mass_params, cable_params, mass_params, cable_params
    )


@pytest.fixture
def sb_linear(clamp_params, mass_params_no_offset, cable_params):
    """Stockbridge with diagonal K, C suitable for the analytic free-vibration test."""
    mass_right = Mass(mass_params_no_offset, cable_params, Side.RIGHT)
    mass_left = Mass(mass_params_no_offset, cable_params, Side.LEFT)
    clamp = Clamp(clamp_params)
    K = np.array([[500.0, 0.0], [0.0, 20.0]])
    C = np.zeros_like(K)
    return Stockbridge(clamp, mass_right, mass_left, K=K, C=C)
