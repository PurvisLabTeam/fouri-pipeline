"""Outputs are written under a temporary name first, so a write cut short never counts as done."""

from pathlib import Path

import numpy as np
import pytest
from builders import REMOVED_NUCLEUS, aligned, converted, measured, segmented
from tifffile import imread

from fouri import io, status
from fouri.config import Experiment
from fouri.status import State, Step


class WriteError(Exception):
    """Stands in for any error raised while an output is being written."""


# How a write can end badly: an error or an interrupt once something is written, or an error first.
FAILURES = [
    pytest.param(WriteError, True, id="error-after-writing"),
    pytest.param(KeyboardInterrupt, True, id="interrupt-after-writing"),
    pytest.param(WriteError, False, id="error-before-writing"),
]


def test_a_finished_write_lands_at_the_path_and_leaves_no_partial(tmp_path: Path) -> None:
    path = tmp_path / "cell_data_B02_df.csv"

    with io.written_whole(path) as partial:
        assert partial == tmp_path / "cell_data_B02_df.csv.partial"
        partial.write_text("label,well\n1,B02\n", encoding="utf-8")
        assert not path.exists()

    assert path.read_text(encoding="utf-8") == "label,well\n1,B02\n"
    assert sorted(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(("error", "wrote_something"), FAILURES)
def test_a_failed_write_leaves_no_file_and_no_partial(
    tmp_path: Path, error: type[BaseException], wrote_something: bool
) -> None:
    path = tmp_path / "cell_data_B02_df.csv"

    with pytest.raises(error), io.written_whole(path) as partial:
        if wrote_something:
            partial.write_text("label,well\n", encoding="utf-8")
        raise error

    assert sorted(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("error", "wrote_something"), FAILURES)
def test_a_failed_write_keeps_the_file_already_there(
    tmp_path: Path, error: type[BaseException], wrote_something: bool
) -> None:
    path = tmp_path / "cell_data_B02_df.csv"
    path.write_text("label,well\n1,B02\n2,B02\n", encoding="utf-8")

    with pytest.raises(error), io.written_whole(path) as partial:
        if wrote_something:
            partial.write_text("label,well\n", encoding="utf-8")
        raise error

    assert path.read_text(encoding="utf-8") == "label,well\n1,B02\n2,B02\n"
    assert sorted(tmp_path.iterdir()) == [path]


def test_saved_transforms_load_back_the_same(experiment: Experiment) -> None:
    transforms = np.full((3, 3, 3), np.nan)  # round 2 not aligned yet
    transforms[0] = np.eye(3)
    transforms[1] = [[1.0, 0.0, 2.5], [0.0, 1.0, -1.25], [0.0, 0.0, 1.0]]
    Path(experiment.output_df).mkdir(parents=True, exist_ok=True)

    saved = io.save_transforms(experiment, "B02", transforms)

    assert saved == io.tmat_path(experiment, "B02", ".npy")
    np.testing.assert_array_equal(io.load_transforms(experiment, "B02"), transforms)
    assert sorted(Path(experiment.output_df).iterdir()) == [saved]


def test_label_paths_pass_over_a_partial_tiff(experiment: Experiment) -> None:
    segmented(experiment, "B02", rounds=[0, 1])

    with io.written_whole(io.label_path(experiment, "B02", 2)) as partial:
        partial.write_bytes(b"the first bytes of a TIFF")
        found = io.label_paths(experiment.segmented, "B02")

    assert found == [io.label_path(experiment, "B02", 0), io.label_path(experiment, "B02", 1)]


def test_a_half_written_cell_data_file_does_not_count_as_measured(experiment: Experiment) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    aligned(experiment, "B02")
    measured(experiment, "B02")
    # the full header names every column the status looks for
    assert status.step_status(experiment, Step.MEASURE, ["B02"])["B02"].state is State.COMPLETE
    path = io.cell_data_path(experiment, "B02")
    header = path.read_text(encoding="utf-8").splitlines()[0]
    path.unlink()

    with io.written_whole(path) as partial:
        partial.write_text(header + "\n", encoding="utf-8")
        state = status.step_status(experiment, Step.MEASURE, ["B02"])["B02"].state

    assert state is State.NOT_STARTED


def test_a_stray_label_copy_is_not_read_as_a_round(experiment: Experiment) -> None:
    segmented(experiment, "B02")
    labels = imread(io.label_path(experiment, "B02", 0))
    io.write_tiff(Path(experiment.segmented) / "B02" / "labels_B02_round_000 - Copy.tif", np.zeros_like(labels))

    found = io.label_paths(experiment.segmented, "B02")
    anchor = io.read_labels(experiment.segmented, "B02", [0])[0]

    assert found == [io.label_path(experiment, "B02", round_index) for round_index in range(3)]
    np.testing.assert_array_equal(anchor, labels)


def test_read_labels_reads_rounds_by_name(experiment: Experiment) -> None:
    segmented(experiment, "B02", rounds=[1, 2])
    round_1 = imread(io.label_path(experiment, "B02", 1))
    round_2 = np.where(round_1 == REMOVED_NUCLEUS, 0, round_1).astype(round_1.dtype)
    io.write_tiff(io.label_path(experiment, "B02", 2), round_2)

    read = io.read_labels(experiment.segmented, "B02", [1, 2])

    np.testing.assert_array_equal(read, np.stack([round_1, round_2]))
    with pytest.raises(FileNotFoundError):
        io.read_labels(experiment.segmented, "B02", [0])
