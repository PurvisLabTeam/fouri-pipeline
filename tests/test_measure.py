"""Measure cells keeps going past a well that fails, and leaves no half-written file behind."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from builders import NUCLEUS_SIDE, NUCLEUS_STARTS, aligned, converted, masked, measured, segmented
from tifffile import imread

from fouri import io, measure
from fouri.config import Experiment
from fouri.report import StepFailed

COVER_COLUMNS = [
    "R0_DNA_covered",
    "R0_pRB_covered",
    "R0_RB_covered",
    "R1_DNA_covered",
    "R1_cMyc_covered",
    "R2_DNA_covered",
    "R2_CyclinA2_covered",
    "R2_EdU_covered",
]
SHIFT = 16

Arrays = dict[str, npt.NDArray[np.generic]]


def _nucleus(label: int, columns_moved: int = 0) -> tuple[slice, slice]:
    row = NUCLEUS_STARTS[(label - 1) // 2]
    col = NUCLEUS_STARTS[(label - 1) % 2] + columns_moved
    return slice(row, row + NUCLEUS_SIDE), slice(col, col + NUCLEUS_SIDE)


def _ready(exp: Experiment) -> None:
    for well in exp.well_list:
        converted(exp, well)
        segmented(exp, well)
        aligned(exp, well)


def _measure_every_well(exp: Experiment) -> None:
    for well in exp.well_list:
        measured(exp, well)


def _cell_data(exp: Experiment, well: str) -> pd.DataFrame:
    return pd.read_csv(io.cell_data_path(exp, well)).set_index("label")


def _shift_round(exp: Experiment, well: str, round_index: int, columns: int) -> None:
    table = io.read_round_table(exp, well)
    for name in table.loc[table["round_index"] == round_index, "raw_file_name"]:
        path = Path(exp.output_images) / well / str(name)
        image = imread(path)
        shifted = np.zeros_like(image)
        shifted[:, columns:] = image[:, :-columns]
        io.write_tiff(path, shifted)
    transforms = io.load_transforms(exp, well)
    transforms[round_index, 0, 2] = columns
    io.save_transforms(exp, well, transforms)


def _without_cyclin_a2(arrays: Arrays) -> Arrays:
    return {name: array for name, array in arrays.items() if name != "final_CyclinA2"}


def _larger(arrays: Arrays) -> Arrays:
    return {name: np.pad(array, ((0, 6), (0, 6))) for name, array in arrays.items()}


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


def test_a_mask_on_a_nucleus_covers_that_cell_in_that_channel_only(experiment: Experiment) -> None:
    _ready(experiment)
    masked(experiment, "B02", 1, {"cMyc": _nucleus(1)})

    _measure_every_well(experiment)

    cells = _cell_data(experiment, "B02")
    assert sorted(column for column in cells.columns if column.endswith("_covered")) == sorted(COVER_COLUMNS)
    assert cells["R1_cMyc_covered"].to_dict() == {1: True, 2: False, 3: False, 4: False}
    assert not cells[COVER_COLUMNS].drop(columns="R1_cMyc_covered").to_numpy().any()


def test_a_mask_on_a_ring_alone_covers_the_cell(experiment: Experiment) -> None:
    _ready(experiment)
    rows, columns = _nucleus(2)
    beside_the_nucleus = (rows, slice(columns.stop + 2, columns.stop + 5))
    masked(experiment, "B02", 2, {"EdU": beside_the_nucleus})

    _measure_every_well(experiment)

    assert _cell_data(experiment, "B02")["R2_EdU_covered"].to_dict() == {1: False, 2: True, 3: False, 4: False}


def test_a_mask_is_judged_in_its_own_round_frame(experiment: Experiment) -> None:
    _ready(experiment)
    _shift_round(experiment, "B02", 1, SHIFT)
    masked(experiment, "B02", 1, {"DNA": _nucleus(3, columns_moved=SHIFT), "cMyc": _nucleus(3)})

    _measure_every_well(experiment)

    cells = _cell_data(experiment, "B02")
    assert cells.loc[[1, 3], "R1_DNA_covered"].tolist() == [False, True]
    assert not cells["R1_cMyc_covered"].any()


def test_a_round_without_masks_is_uncovered(experiment: Experiment) -> None:
    _ready(experiment)
    masked(experiment, "B02", 0, {"DNA": _nucleus(1)})
    masked(experiment, "B02", 1, {"DNA": _nucleus(1)})

    _measure_every_well(experiment)

    cells = _cell_data(experiment, "B02")
    assert len(cells) == 4
    assert not cells[["R2_DNA_covered", "R2_CyclinA2_covered", "R2_EdU_covered"]].to_numpy().any()


@pytest.mark.parametrize("break_masks", [_without_cyclin_a2, _larger], ids=["missing-channel", "other-size"])
def test_masks_missing_a_channel_or_of_another_size_refuse_the_well_alone(
    experiment: Experiment, break_masks: Callable[[Arrays], Arrays]
) -> None:
    _ready(experiment)
    masked(experiment, "B02", 2, {})
    path = io.mask_path(experiment, "B02", 2)
    with np.load(path) as saved:
        arrays = dict(saved)
    np.savez_compressed(path, **break_masks(arrays))

    with pytest.raises(StepFailed) as raised:
        measure.measure_wells(experiment)

    assert raised.value.failed == ["B02"]
    assert not io.cell_data_path(experiment, "B02").exists()
    assert io.cell_data_path(experiment, "B03").exists()
    assert io.cell_data_path(experiment, "C02").exists()
    assert list(Path(experiment.base_dir).rglob("*.partial")) == []


def test_a_stray_label_copy_is_not_read_as_a_round(experiment: Experiment) -> None:
    _ready(experiment)
    labels = imread(io.label_path(experiment, "B02", 0))
    io.write_tiff(
        Path(experiment.segmented) / "B02" / "labels_B02_round_000 - Copy.tif",
        np.where(labels == 1, labels, 0).astype(labels.dtype),
    )

    measured(experiment, "B02")

    assert sorted(_cell_data(experiment, "B02").index) == [1, 2, 3, 4]
