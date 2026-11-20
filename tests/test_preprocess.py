"""Convert images refuses only the well whose converted rounds no longer match info_exp.csv."""

import dataclasses
import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from builders import (
    INFO_EXP,
    ND2_FILE,
    ND2_FOLDER,
    SWAPPED_INFO,
    aligned,
    hand_edited,
    masked,
    measured,
    raw_files,
    segmented,
    well_outputs,
    with_info,
)
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


CHANGES = [
    pytest.param(INFO_EXP.replace("R1,DNA,cMyc,", "R7,DNA,cMyc,"), id="round-renamed"),
    pytest.param(SWAPPED_INFO, id="rounds-swapped"),
    pytest.param(INFO_EXP.replace("R2,DNA,CyclinA2,EdU\n", ""), id="round-removed"),
    pytest.param(INFO_EXP.replace("R1,DNA,cMyc,", "R1,DNA,p21,"), id="signal-changed"),
]


def _fully_processed(exp: Experiment, monkeypatch: pytest.MonkeyPatch, well: str) -> None:
    raw_files(exp, monkeypatch)
    preprocess.preprocess(exp, wells=[well])
    segmented(exp, well)
    hand_edited(exp, well, [0])
    aligned(exp, well)
    masked(exp, well, 1, {"cMyc": (slice(20, 30), slice(20, 30))}, edited=True)
    measured(exp, well)


def _round_order(exp: Experiment, well: str) -> list[str]:
    table = io.read_round_table(exp, well)
    return table.sort_values("round_index")["round_name"].unique().tolist()


@pytest.mark.parametrize("overwrite", [False, True], ids=["overwrite-off", "overwrite-on"])
@pytest.mark.parametrize("info", CHANGES)
def test_a_changed_well_is_refused_alone_without_reconvert_changed(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch, info: str, overwrite: bool
) -> None:
    raw_files(experiment, monkeypatch)
    preprocess.preprocess(experiment, wells=["B02"])
    segmented(experiment, "B02")
    aligned(experiment, "B02")
    before = well_outputs(experiment, "B02")
    with_info(experiment, info)
    raw_files(experiment, monkeypatch)

    with pytest.raises(StepFailed) as raised:
        preprocess.preprocess(experiment, overwrite=overwrite, wells=["B02", "B03"])

    assert raised.value.failed == ["B02"]
    assert well_outputs(experiment, "B02") == before
    assert io.round_table_path(experiment, "B03").exists()


def test_a_round_added_at_the_end_is_not_a_change(experiment: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_files(experiment, monkeypatch)
    preprocess.preprocess(experiment, wells=["B02"])
    with_info(experiment, INFO_EXP + "R3,DNA,,\n")
    raw_files(experiment, monkeypatch)

    changed = preprocess.changed_wells(experiment)
    preprocess.preprocess(experiment, wells=["B02"])

    assert changed == []
    assert _round_order(experiment, "B02") == ["R0", "R1", "R2", "R3"]


@pytest.mark.parametrize("fully", [True, False], ids=["every-output", "converted-and-segmented"])
def test_changed_wells_names_rounds_hand_edits_and_outputs(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch, fully: bool
) -> None:
    if fully:
        _fully_processed(experiment, monkeypatch, "B02")
    else:
        raw_files(experiment, monkeypatch)
        preprocess.preprocess(experiment, wells=["B02"])
        segmented(experiment, "B02")
        hand_edited(experiment, "B02", [0])
    with_info(experiment, SWAPPED_INFO)

    changed = preprocess.changed_wells(experiment)

    outputs = [
        io.round_table_path(experiment, "B02"),
        Path(experiment.output_images) / "B02",
        Path(experiment.segmented) / "B02",
        io.tmat_path(experiment, "B02", ".npy"),
        Path(experiment.masks) / "B02",
        io.cell_data_path(experiment, "B02"),
    ]
    assert changed == [
        preprocess.ChangedWell(
            well="B02",
            rounds=["R1", "R2"],
            hand_edited=["R0", "R1"] if fully else ["R0"],
            outputs=outputs if fully else outputs[:3],
        )
    ]


def test_reconvert_changed_deletes_the_outputs_and_converts_again(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fully_processed(experiment, monkeypatch, "B02")
    with_info(experiment, SWAPPED_INFO)

    preprocess.preprocess(experiment, wells=["B02"], reconvert_changed=True)

    assert _round_order(experiment, "B02") == ["R0", "R2", "R1"]
    assert not (Path(experiment.segmented) / "B02").exists()
    assert not io.tmat_path(experiment, "B02", ".npy").exists()
    assert not (Path(experiment.masks) / "B02").exists()
    assert not io.cell_data_path(experiment, "B02").exists()
    table = io.read_round_table(experiment, "B02")
    assert all((Path(experiment.output_images) / "B02" / str(name)).exists() for name in table["raw_file_name"])


def _old_files(exp: Experiment, well: str, *, keep_new: bool) -> list[Path]:
    table = io.read_round_table(exp, well).rename(
        columns={"round_name": "nameRound", "round_index": "alignRound", "is_align_channel": "alignIm"}
    )
    old_table = Path(exp.output_df) / f"df_{well}.pkl"
    table.drop(columns="raw_file_name").to_pickle(old_table)
    old_transforms = io.tmat_path(exp, well, ".pkl")
    with open(old_transforms, "wb") as fh:
        pickle.dump(list(io.load_transforms(exp, well)), fh)
    if not keep_new:
        io.round_table_path(exp, well).unlink()
        io.tmat_path(exp, well, ".npy").unlink()
    return [old_table, old_transforms]


def _with_bytes(exp: Experiment, well: str, paths: list[Path]) -> dict[Path, bytes]:
    return {**well_outputs(exp, well), **{path: path.read_bytes() for path in paths if path.exists()}}


def test_a_well_with_only_an_old_round_table_is_refused_as_changed(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fully_processed(experiment, monkeypatch, "B02")
    old = _old_files(experiment, "B02", keep_new=False)
    before = _with_bytes(experiment, "B02", old)

    changed = preprocess.changed_wells(experiment)
    with pytest.raises(StepFailed) as raised:
        preprocess.preprocess(experiment, wells=["B02", "B03"])

    assert [well.well for well in changed] == ["B02"]
    assert changed[0].rounds == ["R0", "R1", "R2"]
    assert changed[0].hand_edited == ["round 0", "round 1"]
    assert raised.value.failed == ["B02"]
    assert _with_bytes(experiment, "B02", old) == before
    assert not io.round_table_path(experiment, "B02").exists()
    assert io.round_table_path(experiment, "B03").exists()


@pytest.mark.parametrize("keep_new", [False, True], ids=["only-the-old-files", "old-files-beside-a-changed-well"])
def test_reconverting_deletes_the_old_pkl_files(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch, keep_new: bool
) -> None:
    _fully_processed(experiment, monkeypatch, "B02")
    old = _old_files(experiment, "B02", keep_new=keep_new)
    if keep_new:
        with_info(experiment, SWAPPED_INFO)

    preprocess.preprocess(experiment, wells=["B02"], reconvert_changed=True)

    assert [path for path in old if path.exists()] == []
    assert not (Path(experiment.segmented) / "B02").exists()
    assert not io.tmat_path(experiment, "B02", ".npy").exists()
    assert not (Path(experiment.masks) / "B02").exists()
    assert not io.cell_data_path(experiment, "B02").exists()
    assert _round_order(experiment, "B02") == (["R0", "R2", "R1"] if keep_new else ["R0", "R1", "R2"])
    table = io.read_round_table(experiment, "B02")
    assert all((Path(experiment.output_images) / "B02" / str(name)).exists() for name in table["raw_file_name"])


WITH_FITC = """\
myRound,DAPI,Cy3,Cy5,FITC
R0,DNA,pRB,RB,
R1,DNA,cMyc,,p21
R2,DNA,CyclinA2,EdU,
"""


def _fitc_in_r1_except(exp: Experiment, monkeypatch: pytest.MonkeyPatch, well: str) -> None:
    nd2_metadata = preprocess.nd2_metadata
    read_nd2 = preprocess.read_nd2

    def has_fitc(path: str | Path) -> bool:
        return Path(path).parent.name.startswith("Round_R1_") and not Path(path).name.startswith(f"Well{well}_")

    def metadata(path: str | Path) -> io.ND2Metadata:
        found = nd2_metadata(path)
        if not has_fitc(path):
            return found
        return dataclasses.replace(found, channels=[*found.channels, io.ChannelMetadata("FITC", 488.0, 520.0, 150.0)])

    def image(path: str | Path) -> np.ndarray:
        stack = read_nd2(path)
        return np.concatenate([stack, stack[:1]]) if has_fitc(path) else stack

    monkeypatch.setattr(preprocess, "nd2_metadata", metadata)
    monkeypatch.setattr(preprocess, "read_nd2", image)


def test_a_signal_under_a_configuration_the_nd2_lacks_refuses_the_well(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_info(experiment, WITH_FITC)
    raw_files(experiment, monkeypatch)
    _fitc_in_r1_except(experiment, monkeypatch, "B02")

    with pytest.raises(StepFailed) as raised:
        preprocess.preprocess(experiment, wells=["B02", "B03"])

    assert raised.value.failed == ["B02"]
    assert _nothing_converted(experiment, "B02")
    table = io.read_round_table(experiment, "B03")
    assert table.loc[table["round_name"] == "R1", "signal"].tolist() == ["DNA", "cMyc", "p21"]
    assert preprocess.changed_wells(experiment) == []


def _hand_work(exp: Experiment, well: str) -> dict[Path, bytes]:
    folders = [Path(exp.segmented) / well, Path(exp.masks) / well]
    return {path: path.read_bytes() for folder in folders for path in folder.rglob("*") if path.is_file()}


def test_an_unreadable_new_round_leaves_the_hand_work_uncut(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch
) -> None:
    _convert_segment_and_edit(experiment, monkeypatch)
    masked(experiment, "B02", 0, MASKED)
    _add_smaller_round(experiment, monkeypatch)
    read_nd2 = preprocess.read_nd2

    def unreadable(path: str | Path) -> np.ndarray:
        if Path(path).parent.name.startswith("Round_R3_"):
            raise OSError("the ND2 file is truncated")
        return read_nd2(path)

    monkeypatch.setattr(preprocess, "read_nd2", unreadable)
    hand_work = _hand_work(experiment, "B02")
    table = io.round_table_path(experiment, "B02").read_bytes()

    with pytest.raises(StepFailed) as raised:
        preprocess.preprocess(experiment, wells=["B02"])

    assert raised.value.failed == ["B02"]
    assert _hand_work(experiment, "B02") == hand_work
    assert io.round_table_path(experiment, "B02").read_bytes() == table


def test_a_changed_well_is_refused_when_no_other_well_has_work(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_files(experiment, monkeypatch)
    preprocess.preprocess(experiment, wells=["B02"])
    segmented(experiment, "B02")
    aligned(experiment, "B02")
    before = well_outputs(experiment, "B02")
    with_info(experiment, SWAPPED_INFO)
    raw_files(experiment, monkeypatch)

    with pytest.raises(StepFailed) as raised:
        preprocess.preprocess(experiment, wells=["B02"])

    assert raised.value.failed == ["B02"]
    assert well_outputs(experiment, "B02") == before
