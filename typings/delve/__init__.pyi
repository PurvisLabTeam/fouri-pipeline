"""The part of delve-fs that fouri uses.

Without this file, Pylance reads delve's own code to work out what its calls return,
which made editing fouri/embedding.py slow. Add a name here when fouri starts using another one.
"""

from collections.abc import Sequence
from typing import Any

import numpy.typing as npt
import pandas as pd

def feature_select(
    X: Any = None,  # noqa: N803
    feature_names: npt.ArrayLike | None = None,
    dyn_feats: Sequence[str] | None = None,
    k: int = 10,
    n_pcs: int | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame: ...
