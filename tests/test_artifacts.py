import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import scipy.ndimage as ndi
from builders import converted, masked

from fouri import artifacts, io
from fouri.config import Experiment
from fouri.report import StepFailed

PAINTED = {"DNA": (slice(20, 30), slice(20, 30))}

BLOB_SIZE = 128
BLOB_RADIUS = 8
BIG_SEED_CENTER = (40, 40)
SMALL_SEED_CENTER = (90, 90)
SPREAD = 4
PAD = 3

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


def _disk(center: tuple[int, int]) -> npt.NDArray[np.bool_]:
    rows, cols = np.ogrid[:BLOB_SIZE, :BLOB_SIZE]
    return np.asarray((rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= BLOB_RADIUS**2)


def _excess() -> tuple[npt.NDArray[np.float32], float]:
    image = np.random.default_rng(0).normal(1000, 20, (BLOB_SIZE, BLOB_SIZE))
    for center in (BIG_SEED_CENTER, SMALL_SEED_CENTER):
        image[_disk(center)] += 400
    return artifacts.excess_over_background(image.astype(np.uint16))


def _seeds(*, small: bool = False) -> npt.NDArray[np.uint16]:
    outlines = np.zeros((BLOB_SIZE, BLOB_SIZE), dtype=np.uint16)
    row, col = BIG_SEED_CENTER
    outlines[row - 3 : row + 3, col - 3 : col + 3] = 1
    if small:
        row, col = SMALL_SEED_CENTER
        outlines[row - 1 : row + 2, col - 1 : col + 2] = 2
    return outlines


def _grown(outlines: npt.NDArray[np.uint16], spread: int, pad: int, **options: Any) -> npt.NDArray[np.bool_]:
    excess, noise = _excess()
    values = dict.fromkeys(range(1, int(outlines.max()) + 1), (spread, pad))
    return artifacts.grow_artifacts(outlines, excess, noise, values, **options)


def _near(center: tuple[int, int]) -> npt.NDArray[np.bool_]:
    return np.asarray(ndi.binary_dilation(_disk(center), iterations=PAD))


def test_spread_grows_an_artifact_over_its_excess() -> None:
    final = _grown(_seeds(), SPREAD, 0)

    assert final[_disk(BIG_SEED_CENTER)].all()
    assert not final[~_near(BIG_SEED_CENTER)].any()


def test_spread_0_keeps_the_painted_seed() -> None:
    outlines = _seeds()

    final = _grown(outlines, 0, 0)

    np.testing.assert_array_equal(final, outlines == 1)


def test_pad_widens_the_grown_mask() -> None:
    grown = _grown(_seeds(), SPREAD, 0)

    padded = _grown(_seeds(), SPREAD, PAD)

    np.testing.assert_array_equal(padded, ndi.binary_dilation(grown, iterations=PAD))


def test_an_artifact_under_the_minimum_size_is_left_out() -> None:
    outlines = _seeds(small=True)

    final = _grown(outlines, SPREAD, 0, min_size=25)
    kept = _grown(outlines, SPREAD, 0, min_size=9)

    assert final[_disk(BIG_SEED_CENTER)].all()
    assert not final[_near(SMALL_SEED_CENTER)].any()
    assert kept[_disk(SMALL_SEED_CENTER)].all()


def test_growth_leaves_the_outlines_unchanged() -> None:
    outlines = _seeds(small=True)
    painted = outlines.copy()

    _grown(outlines, SPREAD, PAD, min_size=9)

    np.testing.assert_array_equal(outlines, painted)
