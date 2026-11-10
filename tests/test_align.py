"""Align rounds keeps going past a well that fails or is set up wrongly.

The rounds that finished are saved when another round fails, and a well set up wrongly is refused
alone while the other wells are aligned.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from builders import converted, segmented

from fouri import align, io
from fouri.config import Experiment, load_config
from fouri.report import StepFailed


def _with_anchor(exp: Experiment, anchor_round: int) -> Experiment:
    """Set ``anchor_round`` in the experiment's experiment.json, then load it again."""
    config_path = Path(exp.config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["anchor_round"] = anchor_round
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return load_config(config_path, announce=False)


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


def test_a_well_whose_labels_dont_match_its_rounds_is_refused_and_the_others_align(experiment: Experiment) -> None:
    for well in ("B02", "B03"):
        converted(experiment, well)
        segmented(experiment, well)
    # one label image too many: 4 in segmented for the 3 rounds of B03's round table
    shutil.copyfile(io.label_path(experiment, "B03", 0), io.label_path(experiment, "B03", 3))

    with pytest.raises(StepFailed) as raised:
        align.find_transforms(experiment, wells=["B02", "B03"])

    assert np.isfinite(io.load_transforms(experiment, "B02")).all()
    assert not io.tmat_path(experiment, "B03", ".npy").exists()
    assert raised.value.failed == ["B03"]


def test_a_well_with_transforms_for_another_anchor_is_refused_and_the_others_align(experiment: Experiment) -> None:
    for well in ("B02", "B03"):
        converted(experiment, well)
        segmented(experiment, well)
    shifted = np.eye(3)
    shifted[0, 2] = 2.0
    # aligned onto round 1, while the experiment's anchor is round 0
    saved = np.stack([shifted, np.eye(3), shifted])
    io.save_transforms(experiment, "B03", saved)

    with pytest.raises(StepFailed) as raised:
        align.find_transforms(experiment, wells=["B02", "B03"])

    assert np.isfinite(io.load_transforms(experiment, "B02")).all()
    assert np.array_equal(io.load_transforms(experiment, "B03"), saved)
    assert raised.value.failed == ["B03"]


def test_a_well_without_a_label_for_the_anchor_round_is_refused(experiment: Experiment) -> None:
    exp = _with_anchor(experiment, 1)
    converted(exp, "B02")
    segmented(exp, "B02")
    converted(exp, "B03", rounds=1)
    segmented(exp, "B03", rounds=[0])

    with pytest.raises(StepFailed) as raised:
        align.find_transforms(exp, wells=["B02", "B03"])

    assert np.isfinite(io.load_transforms(exp, "B02")).all()
    assert not io.tmat_path(exp, "B03", ".npy").exists()
    assert raised.value.failed == ["B03"]
