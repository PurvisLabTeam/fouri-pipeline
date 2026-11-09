"""The part of umap-learn that fouri uses.

Without this file, Pylance reads umap's own code to work out what its calls return,
which made editing fouri/embedding.py slow. Add a name here when fouri starts using another one.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

class UMAP:
    def __init__(self, **kwargs: Any) -> None: ...
    def fit_transform(
        self,
        X: npt.ArrayLike,  # noqa: N803
        y: npt.ArrayLike | None = None,
        ensure_all_finite: bool = True,
        **kwargs: Any,
    ) -> npt.NDArray[np.float32]: ...
