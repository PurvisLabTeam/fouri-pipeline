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
from fouri.metadata import feature_prefix
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


RING_SETTINGS = [
    pytest.param(0, 0, id="width-0"),
    pytest.param(6, -1, id="gap-below-0"),
    pytest.param(6, 6, id="gap-equal-to-width"),
]


def _ready_well(exp: Experiment, well: str, **converted_options: bool) -> None:
    converted(exp, well, **converted_options)
    segmented(exp, well)
    aligned(exp, well)


def _well_images(exp: Experiment, well: str, round_index: int | None = None) -> list[Path]:
    table = io.read_round_table(exp, well)
    if round_index is not None:
        table = table[table["round_index"] == round_index]
    return [Path(exp.output_images) / well / str(name) for name in table["raw_file_name"]]


def _fill_background(exp: Experiment, well: str) -> None:
    for path in _well_images(exp, well):
        io.write_tiff(path, np.maximum(imread(path), 1))


def _blank(exp: Experiment, well: str, round_index: int, area: tuple[slice, slice]) -> None:
    for path in _well_images(exp, well, round_index):
        image = imread(path)
        image[area] = 0
        io.write_tiff(path, image)


def _anchor_labels(exp: Experiment, well: str) -> npt.NDArray[np.uint16]:
    return imread(io.label_path(exp, well, exp.anchor_round))


@pytest.mark.parametrize("edge_margin", [1, 0])
def test_a_ring_cut_by_the_image_edge_drops_the_cell(experiment: Experiment, edge_margin: int) -> None:
    _ready_well(experiment, "B02")
    _fill_background(experiment, "B02")
    labels = _anchor_labels(experiment, "B02")
    rows, columns = _nucleus(4)
    labels[rows, columns] = 0
    labels[rows, 56:62] = 4
    io.write_tiff(io.label_path(experiment, "B02", 0), labels)

    measure.measure_well(experiment, "B02", edge_margin=edge_margin)

    assert sorted(_cell_data(experiment, "B02").index) == [1, 2, 3]


def test_cells_without_a_ring_are_counted_as_dropped(experiment: Experiment) -> None:
    _ready_well(experiment, "B02")
    _fill_background(experiment, "B02")
    labels = _anchor_labels(experiment, "B02")
    labels[9:25, 9:25] = 1
    labels[14:18, 14:18] = 0
    labels[15:17, 15:17] = 5
    io.write_tiff(io.label_path(experiment, "B02", 0), labels)

    _, dropped = measure.measure_well(experiment, "B02")

    assert sorted(_cell_data(experiment, "B02").index) == [1, 2, 3, 4]
    assert dropped == 1


def test_a_nucleus_touching_the_border_is_counted_as_dropped(experiment: Experiment) -> None:
    _ready_well(experiment, "B02")
    labels = _anchor_labels(experiment, "B02")
    labels[0:4, 30:34] = 5
    io.write_tiff(io.label_path(experiment, "B02", 0), labels)

    _, dropped = measure.measure_well(experiment, "B02")

    assert sorted(_cell_data(experiment, "B02").index) == [1, 2, 3, 4]
    assert dropped == 1


@pytest.mark.parametrize(("ring_width", "ring_gap"), RING_SETTINGS)
def test_ring_settings_that_make_no_ring_are_refused(experiment: Experiment, ring_width: int, ring_gap: int) -> None:
    _ready(experiment)

    with pytest.raises(ValueError):
        measure.measure_wells(experiment, ring_width=ring_width, ring_gap=ring_gap)

    assert not any(io.cell_data_path(experiment, well).exists() for well in experiment.well_list)


@pytest.mark.parametrize("edge_margin", [0, 1])
def test_a_well_left_with_no_cell_is_refused_and_not_written(experiment: Experiment, edge_margin: int) -> None:
    for well in ("B02", "B03"):
        _ready_well(experiment, well)
    io.write_tiff(io.label_path(experiment, "B03", 0), np.zeros_like(_anchor_labels(experiment, "B03")))

    with pytest.raises(StepFailed) as raised:
        measure.measure_wells(experiment, edge_margin=edge_margin, wells=["B02", "B03"])

    assert raised.value.failed == ["B03"]
    assert io.cell_data_path(experiment, "B02").exists()
    assert not io.cell_data_path(experiment, "B03").exists()


@pytest.mark.parametrize("saved_rounds", ["shifted-anchor", "fewer-rounds"])
def test_transforms_for_another_anchor_refuse_the_well_alone(experiment: Experiment, saved_rounds: str) -> None:
    for well in ("B02", "B03"):
        _ready_well(experiment, well)
    shifted = np.eye(3)
    shifted[:2, 2] = (1.0, 0.0)
    saved = (
        np.stack([shifted, np.eye(3), np.eye(3)]) if saved_rounds == "shifted-anchor" else np.stack([np.eye(3)] * 2)
    )
    io.save_transforms(experiment, "B03", saved)

    with pytest.raises(StepFailed) as raised:
        measure.measure_wells(experiment, wells=["B02", "B03"])

    assert raised.value.failed == ["B03"]
    assert io.cell_data_path(experiment, "B02").exists()
    assert not io.cell_data_path(experiment, "B03").exists()


def test_each_value_is_measured_on_its_own_cell_channel_and_compartment(experiment: Experiment) -> None:
    _ready_well(experiment, "B02", per_cell=True)

    measured(experiment, "B02")

    cells = _cell_data(experiment, "B02")
    assert sorted(cells.index) == [1, 2, 3, 4]
    table = io.read_round_table(experiment, "B02")
    for row in table.itertuples():
        image = imread(Path(experiment.output_images) / "B02" / str(row.raw_file_name))
        prefix = feature_prefix(str(row.round_name), str(row.signal))
        for label in range(1, 5):
            rows, columns = _nucleus(label)
            assert cells.loc[label, f"{prefix}_nuc_mean"] == image[rows, columns].mean()
            assert cells.loc[label, f"{prefix}_ring_mean"] == image[rows.start - 3, columns.start]


def test_a_blank_band_through_a_ring_drops_the_cell(experiment: Experiment) -> None:
    _ready_well(experiment, "B02")
    _blank(experiment, "B02", 1, (slice(8, 10), slice(5, 27)))

    measured(experiment, "B02")

    assert sorted(_cell_data(experiment, "B02").index) == [2, 3, 4]


@pytest.mark.parametrize(("edge_margin", "kept"), [(2, [1, 2, 3, 4]), (3, [2, 3, 4])])
def test_edge_margin_counts_pixels_from_a_blank_area(
    experiment: Experiment, edge_margin: int, kept: list[int]
) -> None:
    _ready_well(experiment, "B02")
    _fill_background(experiment, "B02")
    _blank(experiment, "B02", 0, (slice(0, 5), slice(0, 27)))

    measure.measure_well(experiment, "B02", edge_margin=edge_margin)

    assert sorted(_cell_data(experiment, "B02").index) == kept


def test_overwrite_measures_a_measured_well_again(experiment: Experiment) -> None:
    _ready_well(experiment, "B02")
    measure.measure_wells(experiment, wells=["B02"])
    before = _cell_data(experiment, "B02")["R0_DNA_nuc_mean"]
    dna = Path(experiment.output_images) / "B02" / "Round_R0_wellB02_DNA.tif"
    io.write_tiff(dna, imread(dna) * 2)

    measure.measure_wells(experiment, wells=["B02"])
    kept = _cell_data(experiment, "B02")["R0_DNA_nuc_mean"]
    measure.measure_wells(experiment, overwrite=True, wells=["B02"])
    measured_again = _cell_data(experiment, "B02")["R0_DNA_nuc_mean"]

    pd.testing.assert_series_equal(kept, before)
    pd.testing.assert_series_equal(measured_again, before * 2)


def test_a_well_lacking_a_later_rounds_columns_is_measured_again(experiment: Experiment) -> None:
    _ready_well(experiment, "B02")
    measure.measure_wells(experiment, wells=["B02"])
    path = io.cell_data_path(experiment, "B02")
    table = pd.read_csv(path)
    table.drop(columns=[column for column in table.columns if column.startswith("R2_")]).to_csv(path, index=False)

    measure.measure_wells(experiment, wells=["B02"])

    assert "R2_EdU_nuc_mean" in pd.read_csv(path).columns
