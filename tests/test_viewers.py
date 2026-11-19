import functools
from collections.abc import Callable, Iterator
from typing import Any

import napari
import numpy as np
import pytest
from builders import aligned, converted, masked, segmented
from napari.layers import Labels
from qtpy.QtWidgets import QApplication, QMessageBox
from tifffile import imread

from fouri import io
from fouri.config import Experiment
from fouri.viewers.alignment import AlignmentViewer
from fouri.viewers.artifacts import ArtifactViewer
from fouri.viewers.base import BaseViewer
from fouri.viewers.segmentation import SegmentationViewer

CELLS = [(15, 15), (15, 47), (47, 15)]
NEW_CELL = (slice(28, 36), slice(28, 36))
OTHER_CELL = (slice(2, 8), slice(56, 62))

Opener = Callable[..., Any]


@pytest.fixture
def open_editor(monkeypatch: pytest.MonkeyPatch) -> Iterator[Opener]:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(napari, "Viewer", functools.partial(napari.Viewer, show=False))
    opened: list[BaseViewer] = []

    def opener(kind: type[BaseViewer], exp: Experiment, well: str = "B02") -> Any:
        editor = kind(exp, well)
        opened.append(editor)
        return editor

    yield opener
    for editor in opened:
        editor._self_close = True
        editor.viewer.close()


def _answer(monkeypatch: pytest.MonkeyPatch, button: QMessageBox.StandardButton) -> list[str]:
    asked: list[str] = []

    def answer(box: QMessageBox) -> QMessageBox.StandardButton:
        asked.append(box.text())
        return button

    monkeypatch.setattr(QMessageBox, "exec", answer)
    return asked


def _paint(layer: Labels, region: tuple[slice, slice], number: int) -> None:
    painted = np.zeros(layer.data.shape, dtype=bool)
    painted[region] = True
    layer.data_setitem(np.nonzero(painted), number)
    QApplication.processEvents()


def _fitted(editor: AlignmentViewer) -> None:
    for cell in CELLS:
        editor.anchor_clicks.add(list(cell))
        editor.round_clicks.add(list(cell))
    editor.fit()
    assert editor.fitted_pairs == len(CELLS)


def _failing(*args: Any, **kwargs: Any) -> None:
    raise OSError("disk full")


def test_a_failed_well_switch_leaves_no_fit_to_save(
    experiment: Experiment, open_editor: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    for well in ("B02", "B03"):
        converted(experiment, well)
    segmented(experiment, "B02")
    segmented(experiment, "B03", [0])
    aligned(experiment, "B02")
    _answer(monkeypatch, QMessageBox.StandardButton.Discard)
    editor = open_editor(AlignmentViewer, experiment)
    _fitted(editor)

    editor.well_picker.value = "B03"
    editor.save()

    assert not io.tmat_path(experiment, "B03", ".npy").exists()


def test_an_undo_after_save_marks_the_round_unsaved(experiment: Experiment, open_editor: Opener) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    path = io.label_path(experiment, "B02", 0)
    cellpose = imread(path)
    editor = open_editor(SegmentationViewer, experiment)
    _paint(editor.masks["R0"], NEW_CELL, 9)
    editor.save()

    editor.undo()

    assert editor._unsaved_rounds() == ["R0"]
    editor.save()
    np.testing.assert_array_equal(imread(path), cellpose)


@pytest.mark.parametrize("earlier_record", [False, True], ids=["no-record", "a-record"])
def test_a_failed_mask_save_restores_the_record(
    experiment: Experiment, open_editor: Opener, monkeypatch: pytest.MonkeyPatch, earlier_record: bool
) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    if earlier_record:
        masked(experiment, "B02", 0, {"DNA": OTHER_CELL})
    record = io.mask_record_path(experiment, "B02", 0)
    before = record.read_bytes() if earlier_record else None
    editor = open_editor(ArtifactViewer, experiment)
    _paint(editor.rounds["R0"].channels["DNA"].artifacts, NEW_CELL, 2)
    monkeypatch.setattr(io, "write_npz", _failing)

    with pytest.raises(OSError):
        editor.save()

    assert (record.read_bytes() if record.exists() else None) == before


@pytest.mark.parametrize("earlier_save", [False, True], ids=["first-save", "after-an-earlier-save"])
def test_a_failed_label_save_removes_the_new_original(
    experiment: Experiment, open_editor: Opener, monkeypatch: pytest.MonkeyPatch, earlier_save: bool
) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    path = io.label_path(experiment, "B02", 0)
    original = io.original_label_path(experiment, "B02", 0)
    cellpose = imread(path)
    editor = open_editor(SegmentationViewer, experiment)
    if earlier_save:
        _paint(editor.masks["R0"], OTHER_CELL, 10)
        editor.save()
    saved = path.read_bytes()
    _paint(editor.masks["R0"], NEW_CELL, 9)
    monkeypatch.setattr(io, "write_tiff", _failing)

    with pytest.raises(OSError):
        editor.save()

    assert path.read_bytes() == saved
    if earlier_save:
        np.testing.assert_array_equal(imread(original), cellpose)
    else:
        assert not original.exists()


def test_a_second_save_keeps_the_cellpose_original(experiment: Experiment, open_editor: Opener) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    path = io.label_path(experiment, "B02", 0)
    cellpose = imread(path)
    editor = open_editor(SegmentationViewer, experiment)
    _paint(editor.masks["R0"], NEW_CELL, 9)
    editor.save()

    _paint(editor.masks["R0"], OTHER_CELL, 10)
    editor.save()

    np.testing.assert_array_equal(imread(io.original_label_path(experiment, "B02", 0)), cellpose)
    saved = imread(path)
    assert (saved[NEW_CELL] == 9).all()
    assert (saved[OTHER_CELL] == 10).all()


def test_leaving_an_unsaved_fit_asks_first(
    experiment: Experiment, open_editor: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    for well in ("B02", "B03"):
        converted(experiment, well)
        segmented(experiment, well)
        aligned(experiment, well)
    asked = _answer(monkeypatch, QMessageBox.StandardButton.Cancel)
    editor = open_editor(AlignmentViewer, experiment)
    _fitted(editor)

    editor.well_picker.value = "B03"
    editor.round_picker.value = "R2"
    editor.viewer.window._qt_window.close()

    assert len(asked) == 3
    assert (editor.well, editor.round_name, editor.fitted_pairs) == ("B02", "R1", len(CELLS))
    editor.save()
    editor.round_picker.value = "R2"
    editor.well_picker.value = "B03"
    assert len(asked) == 3
    assert (editor.well, editor.round_name) == ("B03", "R2")


@pytest.mark.parametrize(("min_size", "small_kept"), [(10, True), (30, False)])
def test_minimum_mark_size_is_saved(
    experiment: Experiment, open_editor: Opener, min_size: int, small_kept: bool
) -> None:
    converted(experiment, "B02")
    segmented(experiment, "B02")
    editor = open_editor(ArtifactViewer, experiment)
    layer = editor.rounds["R0"].channels["DNA"].artifacts
    _paint(layer, NEW_CELL, 1)
    _paint(layer, (slice(4, 8), slice(4, 8)), 2)

    editor.min_size_slider.value = min_size
    editor.save()

    with np.load(io.mask_path(experiment, "B02", 0)) as masks:
        final = masks["final_DNA"]
    assert final[NEW_CELL].all()
    if small_kept:
        assert final[4:8, 4:8].all()
    else:
        assert not final[4:8, 4:8].any()
