import pytest
import numpy as np


@pytest.fixture(scope='function')
def ast570():
    """Get conductor properties that matches ASTER570."""
    linm = 1.571
    axs = 3.653E+07
    rts = 1.853E+05
    return linm, axs, rts


@pytest.fixture(scope='function')
def random_spans():
    """Generate 999 random spans with fixed seed."""
    lsmin = 50
    lsmax = 1000
    hrmin = 0.05
    hrmax = 0.5
    slmin = 0.
    slmax = 0.5

    np.random.seed(1234)
    n = 999

    lspan = lsmin + (lsmax - lsmin) * np.random.rand(n)
    tratio = hrmin + (hrmax - hrmin) * np.random.rand(n)
    sld = lspan * (slmin + (slmax - slmin) * np.random.rand(n))

    return lspan, tratio, sld
