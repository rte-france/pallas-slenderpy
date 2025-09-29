from typing import Union

import numpy as np
from numpy.typing import NDArray

floatLike = Union[float, np.floating]
floatArray = NDArray[floatLike]
floatArrayLike = Union[floatLike, floatArray]
