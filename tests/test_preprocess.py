"""Convert images refuses only the well whose converted rounds no longer match info_exp.csv."""

from pathlib import Path

import pytest
from builders import converted

from fouri import io, preprocess
from fouri.config import Experiment
from fouri.report import StepFailed


def _files(exp: Experiment) -> dict[Path, bytes]:
    """Return every file Convert images writes, round tables and images, with its contents."""
    folders = (Path(exp.output_df), Path(exp.output_images))
    return {path: path.read_bytes() for folder in folders for path in folder.rglob("*") if path.is_file()}


def test_a_well_with_renamed_rounds_is_refused_alone(experiment: Experiment) -> None:
    for well in ("B02", "B03"):
        converted(experiment, well)
    table = io.read_round_table(experiment, "B03")
    table["round_name"] = table["round_name"].replace({"R1": "R2", "R2": "R1"})  # R0, R2, R1 by round_index
    io.save_round_table(experiment, "B03", table)
    before = _files(experiment)

    with pytest.raises(StepFailed) as raised:
        preprocess.preprocess(experiment, wells=["B02", "B03"])

    assert raised.value.failed == ["B03"]
    assert _files(experiment) == before
