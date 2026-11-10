"""Measure cells keeps going past a well that fails."""

import pytest
from builders import aligned, converted, segmented

from fouri import io, measure
from fouri.config import Experiment
from fouri.report import StepFailed


def test_the_other_wells_are_measured_when_one_fails(experiment: Experiment) -> None:
    for well in ("B02", "B03"):
        converted(experiment, well)
        segmented(experiment, well)
    aligned(experiment, "B02")
    aligned(experiment, "B03", rounds=[1])  # round 2 left unaligned

    with pytest.raises(StepFailed) as raised:
        measure.measure_wells(experiment, wells=["B02", "B03"])

    assert io.cell_data_path(experiment, "B02").exists()
    assert not io.cell_data_path(experiment, "B03").exists()
    assert raised.value.failed == ["B03"]
