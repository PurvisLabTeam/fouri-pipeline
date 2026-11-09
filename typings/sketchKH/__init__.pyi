"""The part of sketchKH that fouri uses.

Without this file, Pylance reads sketchKH's own code to work out what its calls return,
which made editing fouri/embedding.py slow. Add a name here when fouri starts using another one.
"""

from collections.abc import Sequence

import anndata as ad
import numpy as np
import numpy.typing as npt

def sketch(
    adata: ad.AnnData,
    sample_set_key: str | None = None,
    sample_set_inds: Sequence[npt.ArrayLike] | None = None,
    gamma: float = 1,
    frequency_seed: int | None = None,
    num_subsamples: int = 500,
    backend: str = "threading",
    n_jobs: int = 1,
) -> tuple[list[npt.NDArray[np.int64]], ad.AnnData]: ...
