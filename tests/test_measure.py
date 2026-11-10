"""Measure cells keeps going past a well that fails, and leaves no half-written file behind."""

from pathlib import Path

import pandas as pd
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


def test_measuring_leaves_no_partial_file(experiment: Experiment) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    aligned(experiment, "B02")

    path, _ = measure.measure_well(experiment, "B02")

    assert path == io.cell_data_path(experiment, "B02")
    assert len(pd.read_csv(path)) == 4  # the made-up well's four cells
    assert list(Path(experiment.base_dir).rglob("*.partial")) == []
