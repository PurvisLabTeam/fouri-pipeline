"""Convert images refuses only the well whose converted rounds no longer match info_exp.csv."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from builders import INFO_EXP, ND2_FILE, ND2_FOLDER, converted, hand_edited, masked, raw_files, segmented
from tifffile import imread

from fouri import io, preprocess
from fouri.config import Experiment
from fouri.report import StepFailed

SMALLER = (56, 60)
MASKED = {"pRB": (slice(50, 62), slice(54, 62))}


def _convert_segment_and_edit(exp: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_files(exp, monkeypatch)
    preprocess.preprocess(exp, wells=["B02"])
    segmented(exp, "B02")
    hand_edited(exp, "B02", [0])


def _add_smaller_round(exp: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    with open(exp.info_file, "a", encoding="utf-8") as fh:
        fh.write("R3,DNA,,\n")
    raw_files(exp, monkeypatch, sizes={"R3": SMALLER})


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


def test_a_smaller_crop_trims_the_cellpose_originals(experiment: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    _convert_segment_and_edit(experiment, monkeypatch)
    original = imread(io.original_label_path(experiment, "B02", 0))
    _add_smaller_round(experiment, monkeypatch)

    preprocess.preprocess(experiment, wells=["B02"])

    height, width = SMALLER
    np.testing.assert_array_equal(imread(io.original_label_path(experiment, "B02", 0)), original[:height, :width])


def test_a_smaller_crop_with_overwrite_trims_labels_and_masks(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch
) -> None:
    _convert_segment_and_edit(experiment, monkeypatch)
    masked(experiment, "B02", 0, MASKED)
    _add_smaller_round(experiment, monkeypatch)

    preprocess.preprocess(experiment, overwrite=True, wells=["B02"])

    for round_index in range(3):
        assert imread(io.label_path(experiment, "B02", round_index)).shape == SMALLER
    assert imread(io.original_label_path(experiment, "B02", 0)).shape == SMALLER
    with np.load(io.mask_path(experiment, "B02", 0)) as arrays:
        assert {name: arrays[name].shape for name in arrays.files} == dict.fromkeys(arrays.files, SMALLER)


def test_the_crop_trims_only_the_bottom_and_right(experiment: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_files(experiment, monkeypatch)
    preprocess.preprocess(experiment, wells=["B02"])
    segmented(experiment, "B02")
    masked(experiment, "B02", 0, MASKED)
    image_path = Path(experiment.output_images) / "B02" / "Round_R0_wellB02_DNA.tif"
    image = imread(image_path)
    labels = imread(io.label_path(experiment, "B02", 0))
    with np.load(io.mask_path(experiment, "B02", 0)) as arrays:
        masks = {name: arrays[name] for name in arrays.files}
    _add_smaller_round(experiment, monkeypatch)

    preprocess.preprocess(experiment, wells=["B02"])

    height, width = SMALLER
    np.testing.assert_array_equal(imread(image_path), image[:height, :width])
    np.testing.assert_array_equal(imread(io.label_path(experiment, "B02", 0)), labels[:height, :width])
    with np.load(io.mask_path(experiment, "B02", 0)) as arrays:
        for name, array in masks.items():
            np.testing.assert_array_equal(arrays[name], array[:height, :width])


ALIGNMENT_CHANNELS = [
    pytest.param("R1,Hoechst,cMyc,", id="none"),
    pytest.param("R1,DNA,cMyc,DNA", id="two"),
]


def _with_info(exp: Experiment, old: str, new: str) -> None:
    Path(exp.info_file).write_text(INFO_EXP.replace(old, new), encoding="utf-8")


def _nothing_converted(exp: Experiment, well: str) -> bool:
    images = Path(exp.output_images) / well
    return not io.round_table_path(exp, well).exists() and not (images.exists() and any(images.iterdir()))


def _second_file(exp: Experiment) -> list[str]:
    folder = Path(exp.data_dir) / ND2_FOLDER.format(round_name="R2")
    (folder / ND2_FILE.format(well="B02").replace("Seq0000", "Seq0001")).touch()
    return ["B02"]


def _second_folder(exp: Experiment) -> list[str]:
    folder = Path(exp.data_dir) / "Round_R2_again_made_up"
    folder.mkdir()
    for well in exp.well_list:
        (folder / ND2_FILE.format(well=well)).touch()
    return ["B02", "B03"]


@pytest.mark.parametrize("add_match", [_second_file, _second_folder], ids=["two-files", "two-folders"])
def test_two_files_matching_one_round_are_refused(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch, add_match: Callable[[Experiment], list[str]]
) -> None:
    raw_files(experiment, monkeypatch)
    refused = add_match(experiment)

    with pytest.raises(StepFailed) as raised:
        preprocess.preprocess(experiment, wells=["B02", "B03"])

    assert raised.value.failed == refused
    for well in ("B02", "B03"):
        assert _nothing_converted(experiment, well) == (well in refused)


def test_a_signal_named_twice_in_a_round_is_refused(experiment: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    _with_info(experiment, "R0,DNA,pRB,RB", "R0,DNA,pRB,pRB")
    raw_files(experiment, monkeypatch)

    with pytest.raises(StepFailed) as raised:
        preprocess.preprocess(experiment, wells=["B02"])

    assert raised.value.failed == ["B02"]
    assert _nothing_converted(experiment, "B02")


@pytest.mark.parametrize("round_1", ALIGNMENT_CHANNELS)
def test_a_round_without_one_alignment_channel_is_refused(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch, round_1: str
) -> None:
    _with_info(experiment, "R1,DNA,cMyc,", round_1)
    raw_files(experiment, monkeypatch)

    with pytest.raises(StepFailed) as raised:
        preprocess.preprocess(experiment, wells=["B02"])

    assert raised.value.failed == ["B02"]
    assert _nothing_converted(experiment, "B02")


def test_an_empty_cell_converts_no_channel(experiment: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_files(experiment, monkeypatch)

    preprocess.preprocess(experiment, wells=["B02"])

    table = io.read_round_table(experiment, "B02")
    assert table.loc[table["round_name"] == "R1", "signal"].tolist() == ["DNA", "cMyc"]
    written = sorted(path.name for path in (Path(experiment.output_images) / "B02").glob("Round_R1_*"))
    assert written == ["Round_R1_wellB02_DNA.tif", "Round_R1_wellB02_cMyc.tif"]


def test_each_channel_is_written_to_its_own_file(experiment: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_files(experiment, monkeypatch)

    preprocess.preprocess(experiment, wells=["B02"])

    for row in io.read_round_table(experiment, "B02").itertuples():
        nd2 = preprocess.read_nd2(Path(str(row.dir)) / str(row.sub_dir) / str(row.file))
        written = imread(Path(experiment.output_images) / "B02" / str(row.raw_file_name))
        np.testing.assert_array_equal(written, nd2[int(row.channel_in_file)])
