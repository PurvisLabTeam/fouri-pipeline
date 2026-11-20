"""Align rounds keeps going past a well that fails or is set up wrongly.

The rounds that finished are saved when another round fails, and a well set up wrongly is refused
alone while the other wells are aligned.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from builders import aligned, converted, in_process_map, measured, segmented
from tifffile import imread

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


HAND_AFFINE = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])


def _shift(rows: float, columns: float) -> npt.NDArray[np.float64]:
    matrix = np.eye(3)
    matrix[:2, 2] = (columns, rows)
    return matrix


def _centers(labels: npt.NDArray[np.generic]) -> npt.NDArray[np.float64]:
    return np.array([np.argwhere(labels == label).mean(axis=0) for label in range(1, int(labels.max()) + 1)])


def test_a_hand_alignment_saved_during_the_run_is_kept(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch
) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    aligned(experiment, "B02", rounds=[0, 2])
    saved_during_run = {}

    def save_then_map(function, items, **kwargs):
        align.save_manual_transform(experiment, "B02", 2, HAND_AFFINE)
        saved_during_run["round 2"] = io.load_transforms(experiment, "B02")[2]
        return in_process_map(function, items, **kwargs)

    monkeypatch.setattr(align, "parallel_map", save_then_map)

    align.find_transforms(experiment, wells=["B02"])

    transforms = io.load_transforms(experiment, "B02")
    np.testing.assert_array_equal(transforms[2], saved_during_run["round 2"])
    assert np.isfinite(transforms[1]).all()


@pytest.mark.parametrize(("overwrite", "aligned_rounds"), [(False, [0]), (True, None)], ids=["new-round", "overwrite"])
def test_a_hand_alignment_saved_during_the_run_for_a_round_it_computes_is_kept(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch, overwrite: bool, aligned_rounds: list[int] | None
) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    aligned(experiment, "B02", rounds=aligned_rounds)
    saved_during_run = {}

    def save_then_map(function, items, **kwargs):
        align.save_manual_transform(experiment, "B02", 1, HAND_AFFINE)
        saved_during_run["round 1"] = io.load_transforms(experiment, "B02")[1]
        return in_process_map(function, items, **kwargs)

    monkeypatch.setattr(align, "parallel_map", save_then_map)

    align.find_transforms(experiment, overwrite=overwrite, wells=["B02"])

    transforms = io.load_transforms(experiment, "B02")
    np.testing.assert_array_equal(transforms[1], saved_during_run["round 1"])
    assert np.isfinite(transforms[2]).all()


def test_a_rerun_keeps_hand_alignments(experiment: Experiment) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    align.save_manual_transform(experiment, "B02", 1, HAND_AFFINE)
    hand = io.load_transforms(experiment, "B02")[1]

    align.find_transforms(experiment, wells=["B02"])

    transforms = io.load_transforms(experiment, "B02")
    np.testing.assert_array_equal(transforms[1], hand)
    assert np.isfinite(transforms[2]).all()


def test_a_shorter_saved_file_keeps_its_rounds(experiment: Experiment) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    io.save_transforms(experiment, "B02", np.stack([np.eye(3), _shift(1.5, -0.5)]))

    align.find_transforms(experiment, wells=["B02"])

    transforms = io.load_transforms(experiment, "B02")
    assert transforms.shape == (3, 3, 3)
    np.testing.assert_array_equal(transforms[1], _shift(1.5, -0.5))
    assert np.isfinite(transforms[2]).all()


def test_a_hand_alignment_on_a_new_file_leaves_other_rounds_unaligned(experiment: Experiment) -> None:
    converted(experiment, "B02")

    align.save_manual_transform(experiment, "B02", 1, HAND_AFFINE)

    transforms = io.load_transforms(experiment, "B02")
    np.testing.assert_array_equal(transforms[0], np.eye(3))
    assert np.isfinite(transforms[1]).all()
    assert np.isnan(transforms[2]).all()


def test_a_hand_alignment_keeps_the_other_rounds(experiment: Experiment) -> None:
    converted(experiment, "B02")
    saved = np.stack([np.eye(3), _shift(1.5, -0.5), _shift(-2.0, 0.75)])
    io.save_transforms(experiment, "B02", saved)

    align.save_manual_transform(experiment, "B02", 2, HAND_AFFINE)

    transforms = io.load_transforms(experiment, "B02")
    np.testing.assert_array_equal(transforms[:2], saved[:2])
    assert not np.array_equal(transforms[2], saved[2])


def test_a_hand_alignment_moves_outlines_onto_the_round(experiment: Experiment) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02", rounds=[0])
    anchor = imread(io.label_path(experiment, "B02", 0))
    round_labels = np.roll(anchor, (6, 4), axis=(0, 1))
    fit = align.fit_hand_alignment(_centers(anchor), _centers(round_labels))
    assert fit is not None

    align.save_manual_transform(experiment, "B02", 1, fit[0])

    moved = align.move_to_round(anchor, io.load_transforms(experiment, "B02")[1])
    np.testing.assert_array_equal(moved, round_labels)


def test_a_stray_label_copy_is_not_read_as_a_round(experiment: Experiment) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    labels = imread(io.label_path(experiment, "B02", 0))
    io.write_tiff(
        Path(experiment.segmented) / "B02" / "labels_B02_round_000 - Copy.tif", np.roll(labels, (6, 4), (0, 1))
    )

    align.find_transforms(experiment, wells=["B02"])

    transforms = io.load_transforms(experiment, "B02")
    assert np.allclose(transforms, np.eye(3), atol=0.1)


def test_a_stray_copy_with_a_round_missing_refuses_the_well_naming_the_round(experiment: Experiment) -> None:
    for well in ("B02", "B03"):
        converted(experiment, well)
        segmented(experiment, well)
    labels = imread(io.label_path(experiment, "B03", 0))
    io.write_tiff(Path(experiment.segmented) / "B03" / "labels_B03_round_000 - Copy.tif", labels)
    io.label_path(experiment, "B03", 2).unlink()

    with pytest.raises(StepFailed) as raised:
        align.find_transforms(experiment, wells=["B02", "B03"])

    assert np.isfinite(io.load_transforms(experiment, "B02")).all()
    assert not io.tmat_path(experiment, "B03", ".npy").exists()
    assert raised.value.failed == ["B03"]


def test_a_label_file_for_a_round_the_table_lacks_refuses_the_well_with_a_round_missing(
    experiment: Experiment,
) -> None:
    for well in ("B02", "B03"):
        converted(experiment, well)
        segmented(experiment, well)
    shutil.copyfile(io.label_path(experiment, "B03", 0), io.label_path(experiment, "B03", 3))
    io.label_path(experiment, "B03", 2).unlink()

    with pytest.raises(StepFailed) as raised:
        align.find_transforms(experiment, wells=["B02", "B03"])

    assert np.isfinite(io.load_transforms(experiment, "B02")).all()
    assert not io.tmat_path(experiment, "B03", ".npy").exists()
    assert raised.value.failed == ["B03"]


def test_a_well_not_yet_segmented_is_refused_alone(experiment: Experiment) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    converted(experiment, "B03")

    with pytest.raises(StepFailed) as raised:
        align.find_transforms(experiment, wells=["B02", "B03"])

    assert np.isfinite(io.load_transforms(experiment, "B02")).all()
    assert not io.tmat_path(experiment, "B03", ".npy").exists()
    assert raised.value.failed == ["B03"]


def test_a_shifted_round_is_registered_and_moved_back(experiment: Experiment) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    anchor = imread(io.label_path(experiment, "B02", 0))
    shifted = np.roll(anchor, (6, 4), axis=(0, 1))
    io.write_tiff(io.label_path(experiment, "B02", 1), shifted)

    align.find_transforms(experiment, wells=["B02"])

    transform = io.load_transforms(experiment, "B02")[1]
    np.testing.assert_allclose(transform[:2, 2], [4.0, 6.0], atol=0.1)
    np.testing.assert_allclose(transform[:2, :2], np.eye(2), atol=0.01)
    np.testing.assert_array_equal(align.move_to_anchor(shifted, transform), anchor)


def test_a_refused_well_is_not_reported_done(experiment: Experiment) -> None:
    for well in ("B02", "B03"):
        converted(experiment, well)
        segmented(experiment, well)
    aligned(experiment, "B02")
    shutil.copyfile(io.label_path(experiment, "B03", 0), io.label_path(experiment, "B03", 3))

    with pytest.raises(StepFailed) as raised:
        align.find_transforms(experiment, wells=["B02", "B03"])

    assert raised.value.failed == ["B03"]


def test_overwrite_with_every_round_failing_keeps_hand_alignments(experiment: Experiment) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    align.save_manual_transform(experiment, "B02", 1, HAND_AFFINE)
    align.save_manual_transform(experiment, "B02", 2, HAND_AFFINE)
    hand = io.load_transforms(experiment, "B02")
    for round_index in (1, 2):
        io.label_path(experiment, "B02", round_index).write_bytes(b"these bytes are not a TIFF")

    with pytest.raises(StepFailed):
        align.find_transforms(experiment, overwrite=True, wells=["B02"])

    np.testing.assert_array_equal(io.load_transforms(experiment, "B02"), hand)


def test_a_well_without_a_round_table_is_refused_alone(experiment: Experiment) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    segmented(experiment, "B03")

    with pytest.raises(StepFailed) as raised:
        align.find_transforms(experiment, wells=["B02", "B03"])

    assert np.isfinite(io.load_transforms(experiment, "B02")).all()
    assert not io.tmat_path(experiment, "B03", ".npy").exists()
    assert raised.value.failed == ["B03"]


def test_overwrite_after_an_anchor_change_writes_the_identity_at_the_anchor(experiment: Experiment) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    io.save_transforms(experiment, "B02", np.stack([np.eye(3), _shift(1.5, -0.5), np.eye(3)]))
    exp = _with_anchor(experiment, 1)

    align.find_transforms(exp, overwrite=True, wells=["B02"])
    measured(exp, "B02")

    transforms = io.load_transforms(exp, "B02")
    np.testing.assert_array_equal(transforms[1], np.eye(3))
    assert np.isfinite(transforms).all()
    assert io.cell_data_path(exp, "B02").exists()
