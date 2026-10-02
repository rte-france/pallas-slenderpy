"""Simulation of slender structures: cables and beams.

The type aliases come before any submodule import: modules that do
``from slenderpy import floatArrayLike`` may be loaded while this file runs.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

floatLike = Union[float, np.floating]
floatArray = NDArray[floatLike]
floatArrayLike = Union[floatLike, floatArray]

from slenderpy.components import Conductor as Conductor  # noqa: E402
from slenderpy.components import Span as Span  # noqa: E402
