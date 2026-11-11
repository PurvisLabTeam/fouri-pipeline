"""Segment nuclei keeps going past a well that fails."""

import numpy as np
import pytest
from builders import SIZE, converted

from fouri import io, segment
from fouri.config import Experiment
from fouri.report import StepFailed

pytestmark = pytest.mark.slow  # imports torch

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
