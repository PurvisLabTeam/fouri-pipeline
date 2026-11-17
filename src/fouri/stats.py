"""@TODO."""

from typing import Any, Final

import numpy as np
import numpy.typing as npt

MAD_TO_SD: Final[float] = 1.4826


def robust_location_scale(values: npt.NDArray[Any]) -> tuple[float, float]:
    """@TODO."""
    median = np.median(values)
    absolute_deviations = np.abs(values - median)
    mad = np.median(absolute_deviations)
    robust_sd = MAD_TO_SD * mad

    return float(median), float(robust_sd)
