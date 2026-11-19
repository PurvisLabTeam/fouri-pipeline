"""Segment nuclei keeps going past a well that fails."""

from pathlib import Path

import numpy as np
import pytest
from builders import SIZE, converted, hand_edited, segmented
from tifffile import imread

from fouri import io, segment
from fouri.config import Experiment
from fouri.report import StepFailed

FAILING_VALUE = 7  # every pixel of B03's DNA images, so the fake Cellpose knows them


class _FakeModel:
    """Stands in for a Cellpose model, so the test loads no weights and needs no GPU."""

    def __init__(self, **kwargs: object) -> None:
        import torch

        self.device = torch.device("cpu")


def _fake_segment_image(model, image, diameter, *, flow_threshold, cellprob_threshold, min_size, desc=None):
    if (image == FAILING_VALUE).all():
        raise RuntimeError("made-up Cellpose failure")
    return np.zeros(image.shape, dtype=np.uint16), None


@pytest.mark.slow
def test_segmentation_carries_on_past_a_failed_well(experiment: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    for well in ("B02", "B03", "C02"):
        converted(experiment, well)
    for path in io.alignment_paths(experiment, "B03"):
        io.write_tiff(path, np.full((SIZE, SIZE), FAILING_VALUE, dtype=np.uint16))
    monkeypatch.setattr("cellpose.models.CellposeModel", _FakeModel)
    monkeypatch.setattr(segment, "segment_image", _fake_segment_image)

    with pytest.raises(StepFailed) as raised:
        segment.segment_wells(experiment, allow_cpu=True)

    for well in ("B02", "C02"):
        assert all(io.label_path(experiment, well, round_index).exists() for round_index in range(3))
    assert not any(io.label_path(experiment, "B03", round_index).exists() for round_index in range(3))
    assert raised.value.failed == ["B03"]


def _round_files(exp: Experiment, well: str, round_index: int) -> dict[Path, bytes]:
    paths = (io.label_path(exp, well, round_index), io.original_label_path(exp, well, round_index))
    return {path: path.read_bytes() for path in paths if path.exists()}


def _label_files(exp: Experiment, well: str) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in (Path(exp.segmented) / well).rglob("*.tif")}


def test_overwrite_passes_over_hand_edited_rounds(experiment: Experiment, stand_in_cellpose: object) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    hand_edited(experiment, "B02", [0, 1, 2])
    before = _label_files(experiment, "B02")

    segment.segment_wells(experiment, overwrite=True, wells=["B02"])

    assert _label_files(experiment, "B02") == before


def test_overwrite_segments_only_the_rounds_not_hand_edited(experiment: Experiment, stand_in_cellpose: object) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    hand_edited(experiment, "B02", [1])
    for round_index in (0, 2):
        io.write_tiff(io.label_path(experiment, "B02", round_index), np.zeros((SIZE, SIZE), dtype=np.uint16))
    hand = _round_files(experiment, "B02", 1)

    segment.segment_wells(experiment, overwrite=True, wells=["B02"])

    assert _round_files(experiment, "B02", 1) == hand
    for round_index in (0, 2):
        assert np.unique(imread(io.label_path(experiment, "B02", round_index))).tolist() == [0, 1, 2, 3, 4]


def test_a_round_hand_edited_during_an_overwrite_run_is_kept(
    experiment: Experiment, stand_in_cellpose: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    real_segment_image = segment.segment_image
    edited_during_run = {}

    def segment_then_edit(*args, **kwargs):
        result = real_segment_image(*args, **kwargs)
        if kwargs.get("desc") == "Segmenting round 1":
            hand_edited(experiment, "B02", [1])
            edited_during_run.update(_round_files(experiment, "B02", 1))
        return result

    monkeypatch.setattr(segment, "segment_image", segment_then_edit)

    segment.segment_wells(experiment, overwrite=True, wells=["B02"])

    assert edited_during_run
    assert _round_files(experiment, "B02", 1) == edited_during_run
