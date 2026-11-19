import json
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from builders import SIZE, converted, hand_edited, segmented
from tifffile import imread

from fouri import finetune, io, models
from fouri.config import Experiment

KEPT = "20261117T204206"
LEFTOVER = "20261101T090000.partial"
MODEL_JSON = {
    "created": "2026-11-17T20:42:06",
    "experiment_key": "test",
    "config": "/data/4i/experiment.json",
    "wells": {"B02": [0]},
    "squares": 1,
    "n_epochs": 100,
}
WINDOWS = [
    (0, (0, 20), (44, 64)),
    (0, (20, 40), (20, 40)),
    (0, (21, 41), (21, 41)),
    (0, (38, 58), (38, 58)),
    (1, (38, 58), (38, 58)),
]


class _TrainingError(Exception):
    pass


@pytest.fixture
def models_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(models, "app_data_dir", lambda: tmp_path / "app-data")
    return models.models_dir()


def _hand_edited_well(exp: Experiment) -> None:
    converted(exp, "B02")
    segmented(exp, "B02")
    hand_edited(exp, "B02", [0])


def _kept_model(folder: Path, name: str) -> None:
    directory = folder / name
    (directory / "models").mkdir(parents=True)
    (directory / "models" / "model").write_bytes(b"stand-in model")
    (directory / "model.json").write_text(json.dumps(MODEL_JSON), encoding="utf-8")


def _files(folder: Path) -> dict[Path, bytes]:
    return {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


def _placed(square: finetune.Square) -> tuple[str, int, tuple[int, int], tuple[int, int]]:
    return (
        square.well,
        square.round_index,
        (square.rows.start, square.rows.stop),
        (square.columns.start, square.columns.stop),
    )


def _window(window: tuple[int, tuple[int, int], tuple[int, int]]) -> tuple[slice, slice]:
    return slice(*window[1]), slice(*window[2])


def test_leftover_partial_directories_are_removed_before_training(
    experiment: Experiment,
    stand_in_cellpose: dict[str, ModuleType],
    models_folder: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hand_edited_well(experiment)
    leftover = models_folder / LEFTOVER
    (leftover / "models").mkdir(parents=True)
    (leftover / "models" / "model").write_bytes(b"unfinished model")
    train = stand_in_cellpose["cellpose.train"]
    stand_in_train_seg = train.train_seg
    present: list[bool] = []

    def train_seg(*args: Any, **kwargs: Any) -> Any:
        present.append(leftover.exists())
        return stand_in_train_seg(*args, **kwargs)

    monkeypatch.setattr(train, "train_seg", train_seg)

    kept = finetune.finetune_cellpose(experiment, n_epochs=1, wells=["B02"])

    assert present == [False]
    assert not leftover.exists()
    assert [model.path for model in models.kept_models()[0]] == [kept / "models" / "model"]


def test_the_partial_directory_is_renamed_after_model_json(
    experiment: Experiment,
    stand_in_cellpose: dict[str, ModuleType],
    models_folder: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hand_edited_well(experiment)
    renamed: list[tuple[str, bool]] = []

    def recording(move: Callable[..., Any]) -> Callable[..., Any]:
        def moved(source: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
            if Path(source).name.endswith(".partial") and Path(source).is_dir():
                renamed.append((Path(source).name, (Path(source) / "model.json").is_file()))
            return move(source, target, *args, **kwargs)

        return moved

    monkeypatch.setattr(os, "rename", recording(os.rename))
    monkeypatch.setattr(os, "replace", recording(os.replace))

    kept = finetune.finetune_cellpose(experiment, n_epochs=1, wells=["B02"])

    assert renamed == [(f"{kept.name}.partial", True)]
    assert [model.name for model in models.kept_models()[0]] == [kept.name]


def test_a_failed_training_removes_only_its_own_directory(
    experiment: Experiment,
    stand_in_cellpose: dict[str, ModuleType],
    models_folder: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hand_edited_well(experiment)
    _kept_model(models_folder, KEPT)
    before = _files(models_folder)
    save_paths: list[Path] = []

    def train_seg(
        net: Any, train_data: Any, train_labels: Any, *, save_path: str | Path, model_name: str, **settings: Any
    ) -> Any:
        save_paths.append(Path(save_path))
        model_file = Path(save_path) / "models" / model_name
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_bytes(b"half-trained model")
        raise _TrainingError

    monkeypatch.setattr(stand_in_cellpose["cellpose.train"], "train_seg", train_seg)

    with pytest.raises(_TrainingError):
        finetune.finetune_cellpose(experiment, n_epochs=1, wells=["B02"])

    assert [path.name.endswith(".partial") for path in save_paths] == [True]
    assert not save_paths[0].exists()
    assert sorted(path.name for path in models_folder.iterdir()) == [KEPT]
    assert _files(models_folder) == before


def test_training_squares_cut_one_matching_window_per_corrected_region(
    experiment: Experiment,
    stand_in_cellpose: dict[str, ModuleType],
    models_folder: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    hand_edited(experiment, "B02", [0, 1])
    labels_path = io.label_path(experiment, "B02", 0)
    labels = imread(labels_path)
    labels[0:3, 60:64] = 5
    labels[30, 30] = 6
    labels[31, 31] = 6
    io.write_tiff(labels_path, labels)
    images = []
    for round_index, path in enumerate(io.alignment_paths(experiment, "B02")):
        image = (np.arange(SIZE * SIZE).reshape(SIZE, SIZE) + SIZE * SIZE * round_index).astype(np.uint16)
        io.write_tiff(path, image)
        images.append(image)
    saved = [imread(io.label_path(experiment, "B02", round_index)) for round_index in range(len(images))]
    monkeypatch.setattr(finetune, "SQUARE_SIZE", 20)
    train = stand_in_cellpose["cellpose.train"]
    stand_in_train_seg = train.train_seg
    received: list[tuple[npt.NDArray[Any], npt.NDArray[Any]]] = []

    def train_seg(net: Any, train_data: Any, train_labels: Any, **settings: Any) -> Any:
        received.extend(zip(train_data, train_labels, strict=True))
        return stand_in_train_seg(net, train_data, train_labels, **settings)

    monkeypatch.setattr(train, "train_seg", train_seg)

    squares = finetune.training_squares(experiment, ["B02"])
    finetune.finetune_cellpose(experiment, n_epochs=1, wells=["B02"])

    assert sorted(_placed(square) for square in squares) == [("B02", *window) for window in WINDOWS]
    cut = []
    for data, cut_labels in received:
        matches = [window for window in WINDOWS if np.array_equal(images[window[0]][_window(window)], data)]
        assert len(matches) == 1
        assert np.array_equal(saved[matches[0][0]][_window(matches[0])], cut_labels)
        cut.append(matches[0])
    assert sorted(cut) == WINDOWS
