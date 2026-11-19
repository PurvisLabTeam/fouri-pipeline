import json
from pathlib import Path

import pytest
from builders import converted, masked

from fouri import artifacts, io
from fouri.config import Experiment
from fouri.report import StepFailed

PAINTED = {"DNA": (slice(20, 30), slice(20, 30))}

BROKEN_RECORDS = [
    pytest.param("{not json", id="not-json"),
    pytest.param(json.dumps({"masks": {}}), id="no-edited"),
    pytest.param(json.dumps({"edited": "yes", "masks": {}}), id="edited-not-true-or-false"),
]


def _files(exp: Experiment, well: str) -> dict[str, bytes]:
    folder = Path(exp.masks) / well
    return {path.name: path.read_bytes() for path in folder.iterdir()} if folder.exists() else {}


def _round_files(exp: Experiment, well: str, round_index: int) -> dict[Path, bytes]:
    paths = (io.mask_path(exp, well, round_index), io.mask_record_path(exp, well, round_index))
    return {path: path.read_bytes() for path in paths if path.exists()}


def _edited(exp: Experiment, well: str, round_index: int) -> object:
    return json.loads(io.mask_record_path(exp, well, round_index).read_text(encoding="utf-8"))["edited"]


def test_a_round_painted_during_the_run_is_kept(experiment: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    converted(experiment, "B02")
    real_round_masks = artifacts.round_masks
    painted_during_run = {}

    def paint_then_find(images):
        masks = real_round_masks(images)
        if any(path.name.startswith("Round_R1_") for path in images.values()):
            masked(experiment, "B02", 1, PAINTED, edited=True)
            painted_during_run.update(_round_files(experiment, "B02", 1))
        return masks

    monkeypatch.setattr(artifacts, "round_masks", paint_then_find)

    artifacts.find_artifacts(experiment, wells=["B02"])

    assert painted_during_run
    assert _round_files(experiment, "B02", 1) == painted_during_run
    assert _edited(experiment, "B02", 0) is False
    assert _edited(experiment, "B02", 2) is False


def test_overwrite_never_replaces_a_painted_round(experiment: Experiment) -> None:
    converted(experiment, "B02")
    masked(experiment, "B02", 1, PAINTED, edited=True)
    painted = _round_files(experiment, "B02", 1)

    artifacts.find_artifacts(experiment, overwrite=True, wells=["B02"])

    assert _round_files(experiment, "B02", 1) == painted
    assert io.mask_path(experiment, "B02", 0).exists()
    assert io.mask_path(experiment, "B02", 2).exists()


@pytest.mark.parametrize("record", BROKEN_RECORDS)
def test_a_record_that_cannot_be_read_refuses_the_well(experiment: Experiment, record: str) -> None:
    for well in ("B02", "B03"):
        converted(experiment, well)
    masked(experiment, "B03", 1, PAINTED)
    io.mask_record_path(experiment, "B03", 1).write_text(record, encoding="utf-8")
    before = _files(experiment, "B03")

    with pytest.raises(StepFailed) as raised:
        artifacts.find_artifacts(experiment, wells=["B02", "B03"])

    assert raised.value.failed == ["B03"]
    assert _files(experiment, "B03") == before
    assert all(io.mask_path(experiment, "B02", round_index).exists() for round_index in range(3))


def test_a_painted_record_without_masks_is_kept(experiment: Experiment) -> None:
    converted(experiment, "B02")
    masked(experiment, "B02", 1, PAINTED, edited=True)
    io.mask_path(experiment, "B02", 1).unlink()
    record = io.mask_record_path(experiment, "B02", 1).read_bytes()

    artifacts.find_artifacts(experiment, wells=["B02"])

    assert not io.mask_path(experiment, "B02", 1).exists()
    assert io.mask_record_path(experiment, "B02", 1).read_bytes() == record


def test_masks_without_a_record_are_skipped_without_overwrite(experiment: Experiment) -> None:
    converted(experiment, "B02")
    masked(experiment, "B02", 1, PAINTED)
    io.mask_record_path(experiment, "B02", 1).unlink()
    masks = io.mask_path(experiment, "B02", 1).read_bytes()

    artifacts.find_artifacts(experiment, wells=["B02"])

    assert io.mask_path(experiment, "B02", 1).read_bytes() == masks
    assert not io.mask_record_path(experiment, "B02", 1).exists()
