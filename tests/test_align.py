"""Align rounds saves the rounds that finished when another round fails."""

import numpy as np
import pytest
from builders import converted, segmented

from fouri import align, io
from fouri.config import Experiment
from fouri.report import StepFailed


def test_the_rounds_that_finished_are_saved(experiment: Experiment) -> None:
    for well in ("B02", "B03"):
        converted(experiment, well)
        segmented(experiment, well)
    io.label_path(experiment, "B03", 2).write_bytes(b"these bytes are not a TIFF")

    with pytest.raises(StepFailed) as raised:
        align.find_transforms(experiment, wells=["B02", "B03"])

    assert np.isfinite(io.load_transforms(experiment, "B02")).all()
    b03 = io.load_transforms(experiment, "B03")
    assert np.isfinite(b03[1]).all()
    assert np.isnan(b03[2]).all()
    assert raised.value.failed == ["B03"]
