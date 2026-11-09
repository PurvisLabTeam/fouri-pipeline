"""The part of phate that fouri uses.

Without this file, Pylance reads phate's own code to work out what its calls return,
which made editing fouri/embedding.py slow. Add a name here when fouri starts using another one.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

class PHATE:
    def __init__(self, **kwargs: Any) -> None: ...
    def fit_transform(self, X: npt.ArrayLike, **kwargs: Any) -> npt.NDArray[np.float64]: ...  # noqa: N803
