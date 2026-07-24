from typing import Union

import numpy as np
from numpy.typing import NDArray

from slenderpy.future.components import Conductor as Conductor
from slenderpy.future.components import Span as Span

floatLike = Union[float, np.floating]
floatArray = NDArray[floatLike]
floatArrayLike = Union[floatLike, floatArray]
