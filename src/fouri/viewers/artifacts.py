import inspect
import json
import os
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.ndimage as ndi
from magicgui.widgets import Label, PushButton, Slider, Widget
from napari.layers import Image, Labels
from napari.utils.colormaps import CyclicLabelColormap
from qtpy.QtCore import QTimer
from skimage.segmentation import clear_border
from tifffile import TiffFile, imread

from fouri import io
from fouri.align import move_to_round
from fouri.artifacts import excess_over_background, grow_artifacts
from fouri.measure import measure_wells
from fouri.rings import make_rings
from fouri.viewers.base import BaseViewer, KeyboardShortcuts
from fouri.viewers.segmentation import (
    CLICK_MODES,
    DRAG_PIXELS,
    KEY_CLEAR,
    KEY_DELETE,
    KEY_NEW,
    SELECTED_COLORS,
    _contrast_limits,
)

ARTIFACT_COLORS: CyclicLabelColormap = CyclicLabelColormap(
    colors=["transparent", "cyan", "deepskyblue", "turquoise", "aquamarine", "lightskyblue"]
)
GROWN_COLORS: CyclicLabelColormap = CyclicLabelColormap(colors=["transparent", "lime"])

SPREAD_RANGE = (0, 6)
PAD_RANGE = (0, 20)
MIN_SIZE_RANGE = (0, 200)
MIN_SIZE_START = 25
ARTIFACT_OPACITY = 0.6
KEY_REDO = "Ctrl+Shift+Z"
OUTLINE_WIDTH = 2
COUNT_NOTICE = "Includes cells at image edges and blank areas, already left out by Measure cells."

_MEASURE_DEFAULTS = inspect.signature(measure_wells).parameters
RING_WIDTH: int = _MEASURE_DEFAULTS["ring_width"].default
RING_GAP: int = _MEASURE_DEFAULTS["ring_gap"].default

Values = dict[int, tuple[int, int]]


@dataclass
class _Channel:
    artifacts: Labels
    spread: int
    pad: int
    values: Values


@dataclass
class _Round:
    index: int
    align_signal: str
    paths: dict[str, Path]
    channels: dict[str, _Channel]


@dataclass
class _Masks:
    outlines: dict[str, npt.NDArray[np.uint16]]
    finals: dict[str, npt.NDArray[np.bool_]]
    channel_values: dict[str, tuple[int, int]]
    artifact_values: dict[str, Values]


class ArtifactViewer(BaseViewer):
    title = "Artifact masks"

    rounds: dict[str, _Round]
    images: dict[str, Image]
    grown: dict[str, Labels]
    excess: dict[str, Future[tuple[npt.NDArray[np.float32], float]]]
    finals: dict[str, npt.NDArray[np.bool_]]
    shown: str | None
    shown_signals: set[str]
    edited: str
    selection: list[int]
    changed: set[str]
    min_size: int
    moved: dict[str, npt.NDArray[Any]]
    saved_covered: dict[str, set[int]]
    no_count: str | None
    highlight: Labels
    spread_slider: Slider
    pad_slider: Slider
    min_size_slider: Slider
    count_line: Label
    count_notice: Label

    _pending: dict[str, _Masks]
    _edits_due: set[tuple[str, str]]
    _quiet: bool
    _pool: ThreadPoolExecutor

    def _build_layers(self) -> None:
        self.rounds = {}
        self.images = {}
        self.grown = {}
        self.excess = {}
        self.finals = {}
        self.shown = None
        self.shown_signals = set()
        self.edited = ""
        self.selection = []
        self.changed = set()
        self.min_size = MIN_SIZE_START
        self.moved = {}
        self.saved_covered = {}
        self.no_count = None
        self._pending = {}
        self._edits_due = set()
        self._quiet = False
        self._pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))
        self.highlight = self.viewer.add_labels(
            np.zeros((1, 1), dtype=np.uint8), name="selected artifacts", colormap=SELECTED_COLORS, opacity=0.5
        )
        self.highlight.editable = False
        self.viewer.layers.selection.events.active.connect(self._on_active)

    def _tool_widgets(self) -> list[Widget]:
        delete_button = PushButton(text=f"Delete artifact ({KEY_DELETE})")
        delete_button.changed.connect(self.delete_artifacts)
        new_button = PushButton(text=f"New artifact ({KEY_NEW})")
        new_button.changed.connect(self.new_artifact)

        self.spread_slider = Slider(
            value=SPREAD_RANGE[0], min=SPREAD_RANGE[0], max=SPREAD_RANGE[1], step=1, tracking=False, label="Spread"
        )
        self.pad_slider = Slider(
            value=PAD_RANGE[0], min=PAD_RANGE[0], max=PAD_RANGE[1], step=1, tracking=False, label="Pad (px)"
        )
        self.min_size_slider = Slider(
            value=MIN_SIZE_START,
            min=MIN_SIZE_RANGE[0],
            max=MIN_SIZE_RANGE[1],
            step=1,
            tracking=False,
            label="Minimum mark size (px)",
        )
        self.spread_slider.changed.connect(self.set_spread)
        self.pad_slider.changed.connect(self.set_pad)
        self.min_size_slider.changed.connect(self.set_min_size)

        self.count_line = Label(value="", label="")
        self.count_notice = Label(value=COUNT_NOTICE, label="")
        for line in (self.count_line, self.count_notice):
            line.native.setWordWrap(True)

        return [
            delete_button,
            new_button,
            self.spread_slider,
            self.pad_slider,
            self.min_size_slider,
            self.count_line,
            self.count_notice,
        ]

    def _extra_keys(self) -> dict[str, Callable[[], None]]:
        return {KEY_DELETE: self.delete_artifacts, KEY_NEW: self.new_artifact}

    def _extra_shortcuts(self) -> list[str]:
        return [
            "Click: select an artifact",
            "Shift+click: add or remove an artifact",
            f"{KEY_CLEAR} or click the background: clear the selection",
            "Drag: pan",
            f"{KEY_DELETE}: delete the selected artifacts",
            f"{KEY_NEW}: paint a new artifact",
        ]

    def _round_choices(self, round_table: pd.DataFrame) -> list[str]:
        choices = [name for name, _, _ in _round_rows(round_table)]
        if not choices:
            raise ValueError("There are no rounds to edit.")
        return choices

    def _start_round(self) -> str | None:
        rounds = io.read_info_file(self.exp)
        return str(rounds.iloc[self.exp.anchor_round]["round_name"])

    def _load_well(self, well: str, round_table: pd.DataFrame) -> None:
        moved, saved_covered, no_count = self._count_inputs(well, round_table)
        pending: dict[str, _Masks] = {}
        if self.shown is not None:
            choices = self._round_choices(round_table)
            target = self.shown if self.shown in choices else choices[0]
            pending[target] = self._read_round_masks(well, round_table, target)

        if self.images:
            self.shown_signals = {signal for signal, layer in self.images.items() if layer.visible}
        self._quiet = True
        try:
            for round_ in self.rounds.values():
                for channel in round_.channels.values():
                    self.viewer.layers.remove(channel.artifacts)
            for layer in [*self.images.values(), *self.grown.values()]:
                self.viewer.layers.remove(layer)
        finally:
            self._quiet = False
        for future in self.excess.values():
            future.cancel()

        self.rounds = {}
        self.images = {}
        self.grown = {}
        self.excess = {}
        self.finals = {}
        self.shown = None
        self.selection = []
        self.changed = set()
        self.highlight.data = np.zeros((1, 1), dtype=np.uint8)
        self.moved = moved
        self.saved_covered = saved_covered
        self.no_count = no_count
        self._pending = pending
        self._edits_due = set()

    def _load_round(self, round_name: str) -> None:
        index, align_signal = self._round_info(round_name)
        paths = self._round_paths(round_name)
        masks = None
        if round_name not in self.rounds:
            masks = self._pending.pop(round_name, None) or self._read_round_masks(
                self.well, self.round_table, round_name
            )
        images = {signal: imread(path) for signal, path in paths.items()}

        first_round = not self.images
        if self.images:
            self.shown_signals = {signal for signal, layer in self.images.items() if layer.visible}
        self.shown_signals = self.shown_signals.intersection(images) | {align_signal}

        self._quiet = True
        try:
            if masks is not None:
                self.rounds[round_name] = self._make_round(round_name, index, align_signal, paths, masks)
            round_ = self.rounds[round_name]

            for layer in [*self.images.values(), *self.grown.values()]:
                self.viewer.layers.remove(layer)
            for future in self.excess.values():
                future.cancel()

            self.images = {}
            self.grown = {}
            self.excess = {}
            self.finals = {}
            for signal, image in images.items():
                visible = signal in self.shown_signals
                layer = Image(
                    image,
                    name=f"{round_name} {signal}",
                    colormap="magma",
                    contrast_limits=_contrast_limits(image),
                    visible=visible,
                )
                self.viewer.layers.insert(0, layer)
                layer.events.visible.connect(
                    lambda _event, round_name=round_name, signal=signal, layer=layer: self._on_visible(
                        round_name, signal, layer
                    )
                )
                self.images[signal] = layer
                self.excess[signal] = self._pool.submit(excess_over_background, image)

                grown = self.viewer.add_labels(
                    np.zeros(image.shape, dtype=np.uint8),
                    name=f"{round_name} {signal} grown",
                    colormap=GROWN_COLORS,
                    visible=visible,
                )
                grown.editable = False
                grown.events.visible.connect(
                    lambda _event, round_name=round_name, signal=signal, layer=grown: self._on_visible(
                        round_name, signal, layer
                    )
                )
                self.grown[signal] = grown

            for name, other in self.rounds.items():
                for signal, channel in other.channels.items():
                    channel.artifacts.visible = name == round_name and signal in self.shown_signals

            self.viewer.layers.move(self.viewer.layers.index(self.highlight), len(self.viewer.layers))
            self.shown = round_name
            self.edited = align_signal
            self.selection = []
            self.viewer.layers.selection.active = round_.channels[align_signal].artifacts
        finally:
            self._quiet = False

        self._show_selection()
        self._show_values()
        self._regrow(list(round_.channels))
        self._update_count()
        if first_round:
            self.viewer.reset_view()
        self.prompt("Choose a tool, or paint.")

    def _make_round(
        self, round_name: str, index: int, align_signal: str, paths: dict[str, Path], masks: _Masks
    ) -> _Round:
        channels: dict[str, _Channel] = {}
        for signal in paths:
            layer = self.viewer.add_labels(
                masks.outlines[signal],
                name=f"{round_name} {signal} artifacts",
                colormap=ARTIFACT_COLORS,
                opacity=ARTIFACT_OPACITY,
                visible=False,
            )
            layer.events.paint.connect(
                lambda _event, round_name=round_name, signal=signal: self._on_paint(round_name, signal)
            )
            layer.events.visible.connect(
                lambda _event, round_name=round_name, signal=signal, layer=layer: self._on_visible(
                    round_name, signal, layer
                )
            )
            layer.mouse_drag_callbacks.append(self._on_click)
            layer.bind_key(KeyboardShortcuts.UNDO, lambda _layer: self.undo(), overwrite=True)
            layer.bind_key(KEY_REDO, lambda _layer: self._redo(), overwrite=True)
            layer.events.mode.connect(lambda _event, layer=layer: self._bind_clear(layer))
            self._bind_clear(layer)

            spread, pad = masks.channel_values[signal]
            saved = masks.artifact_values[signal]
            values = {number: saved.get(number, (spread, pad)) for number in _present(masks.outlines[signal])}
            channels[signal] = _Channel(artifacts=layer, spread=spread, pad=pad, values=values)
        return _Round(index=index, align_signal=align_signal, paths=paths, channels=channels)

    def save(self) -> None:
        unsaved = self._unsaved_rounds()
        if not unsaved:
            self.prompt(f"{self.well} has no unsaved edits.")
            return
        for round_name in unsaved:
            round_ = self.rounds[round_name]
            finals = self._finals_for(round_name)
            record = {
                "edited": True,
                "masks": {
                    signal: {
                        "spread": channel.spread,
                        "pad": channel.pad,
                        "artifacts": {
                            str(number): {"spread": channel.values[number][0], "pad": channel.values[number][1]}
                            for number in _present(channel.artifacts.data)
                        },
                    }
                    for signal, channel in round_.channels.items()
                },
            }
            mask_path = io.mask_path(self.exp, self.well, round_.index)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                io.written_whole(io.mask_record_path(self.exp, self.well, round_.index)) as partial,
                open(partial, "w", encoding="utf-8") as fh,
            ):
                json.dump(record, fh, indent=2)
            arrays = {
                f"{prefix}{signal}": array
                for signal, channel in round_.channels.items()
                for prefix, array in (
                    ("final_", finals[signal]),
                    ("outlines_", np.asarray(channel.artifacts.data, dtype=np.uint16)),
                )
            }
            io.write_npz(mask_path, arrays)
            self.changed.discard(round_name)
            if round_name in self.moved:
                self.saved_covered[round_name] = _covered(self.moved[round_name], finals.values())

        self._update_count()
        self.prompt(
            f"Saved {', '.join(unsaved)} for {self.well}. "
            f"Run Measure cells on {self.well} with Overwrite to drop the cells covered by these masks."
        )

    def undo(self) -> None:
        channel = self._edited_channel()
        if channel is None or self.shown is None:
            return
        channel.artifacts.undo()
        self._after_edit(self.shown, self.edited)

    def _redo(self) -> None:
        channel = self._edited_channel()
        if channel is None or self.shown is None:
            return
        channel.artifacts.redo()
        self._after_edit(self.shown, self.edited)

    def _unsaved_rounds(self) -> list[str]:
        unsaved = []
        for round_name in self.round_names:
            if round_name not in self.changed or round_name not in self.rounds:
                continue
            round_ = self.rounds[round_name]
            saved = self._read_round_masks(self.well, self.round_table, round_name)
            for signal, channel in round_.channels.items():
                data = np.asarray(channel.artifacts.data)
                default = saved.channel_values[signal]
                if not np.array_equal(data, saved.outlines[signal]) or any(
                    channel.values[number] != saved.artifact_values[signal].get(number, default)
                    for number in _present(data)
                ):
                    unsaved.append(round_name)
                    break
        return unsaved

    def delete_artifacts(self) -> None:
        channel = self._edited_channel()
        if channel is None:
            return
        if not self.selection:
            self.prompt("Select an artifact first.")
            return
        deleted = self.selection
        self.selection = []
        self._show_channel(self.edited)
        layer = channel.artifacts
        layer.data_setitem(np.nonzero(np.isin(layer.data, deleted)), 0)
        if self.shown is not None:
            self._after_edit(self.shown, self.edited)
        self._show_values()
        self.prompt(f"Deleted {_artifact_list(deleted)} in {self.edited}.")

    def new_artifact(self) -> None:
        channel = self._edited_channel()
        if channel is None:
            return
        self._show_channel(self.edited)
        layer = channel.artifacts
        number = int(np.asarray(layer.data).max()) + 1
        self.selection = []
        self._show_selection()
        self._show_values()
        layer.selected_label = number
        layer.mode = "paint"
        self.prompt(f"Painting new artifact {number:,} in {self.edited}. Paint it, then choose another tool.")

    def select_artifact(self, number: int, *, add: bool) -> None:
        channel = self._edited_channel()
        if channel is None:
            return
        if number == 0:
            self._clear_selection()
            return
        if not add:
            self.selection = [number]
        elif number in self.selection:
            self.selection.remove(number)
        else:
            self.selection.append(number)
        self._show_selection()
        self._show_values()
        if self.selection:
            self.prompt(
                f"Selected {_artifact_list(self.selection)} in {self.edited}. "
                "Shift+click more, Delete to delete them, or move Spread and Pad."
            )
        else:
            self.prompt("Choose a tool, or paint.")

    def set_spread(self, spread: int) -> None:
        self._set_values(spread=int(spread))

    def set_pad(self, pad: int) -> None:
        self._set_values(pad=int(pad))

    def set_min_size(self, min_size: int) -> None:
        self.min_size = int(min_size)
        if self.shown is None:
            return
        self._regrow(list(self.rounds[self.shown].channels))
        self._update_count()

    def _set_values(self, *, spread: int | None = None, pad: int | None = None) -> None:
        if self.shown is None:
            return
        round_ = self.rounds[self.shown]
        if self.selection:
            targets = {self.edited: self.selection}
        else:
            targets = {signal: list(channel.values) for signal, channel in round_.channels.items()}
        for signal, numbers in targets.items():
            values = round_.channels[signal].values
            for number in numbers:
                old_spread, old_pad = values[number]
                values[number] = (old_spread if spread is None else spread, old_pad if pad is None else pad)
        self.changed.add(self.shown)
        self._regrow([signal for signal, numbers in targets.items() if numbers])
        self._update_count()

    def _on_paint(self, round_name: str, signal: str) -> None:
        key = (round_name, signal)
        if key in self._edits_due:
            return
        self._edits_due.add(key)
        QTimer.singleShot(0, lambda: self._run_due_edit(key))

    def _run_due_edit(self, key: tuple[str, str]) -> None:
        if key in self._edits_due:
            self._after_edit(*key)

    def _after_edit(self, round_name: str, signal: str) -> None:
        self._edits_due.discard((round_name, signal))
        if round_name not in self.rounds or signal not in self.rounds[round_name].channels:
            return
        self.changed.add(round_name)
        channel = self.rounds[round_name].channels[signal]
        _sync_values(channel)
        if round_name != self.shown:
            return
        if signal == self.edited:
            self._show_selection()
        self._regrow([signal])
        self._update_count()

    def _regrow(self, signals: list[str]) -> None:
        if self.shown is None or not signals:
            return
        round_ = self.rounds[self.shown]
        grown: dict[str, Future[npt.NDArray[np.bool_]]] = {}
        for signal in signals:
            channel = round_.channels[signal]
            data = np.asarray(channel.artifacts.data)
            if not data.any():
                self.finals[signal] = np.zeros(data.shape, dtype=np.bool_)
                continue
            excess, noise = self.excess[signal].result()
            grown[signal] = self._pool.submit(
                grow_artifacts, data, excess, noise, _with_defaults(channel), min_size=self.min_size
            )
        for signal, future in grown.items():
            self.finals[signal] = future.result()
        for signal in signals:
            self.grown[signal].data = _outline(self.finals[signal])

    def _finals_for(self, round_name: str) -> dict[str, npt.NDArray[np.bool_]]:
        if round_name == self.shown:
            return dict(self.finals)
        round_ = self.rounds[round_name]
        finals: dict[str, npt.NDArray[np.bool_]] = {}
        for signal, channel in round_.channels.items():
            _sync_values(channel)
            data = np.asarray(channel.artifacts.data)
            if not data.any():
                finals[signal] = np.zeros(data.shape, dtype=np.bool_)
                continue
            excess, noise = excess_over_background(imread(round_.paths[signal]))
            finals[signal] = grow_artifacts(data, excess, noise, _with_defaults(channel), min_size=self.min_size)
        return finals

    def _update_count(self) -> None:
        if self.no_count is not None or self.shown is None:
            self.count_line.value = self.no_count or ""
            self.count_notice.visible = False
            return
        live = _covered(self.moved[self.shown], self.finals.values())
        others = set().union(*(cells for name, cells in self.saved_covered.items() if name != self.shown))
        self.count_line.value = (
            f"These masks drop {len(live | others):,} cells in {self.well} ({len(live):,} in {self.shown})."
        )
        self.count_notice.visible = True

    def _count_inputs(
        self, well: str, round_table: pd.DataFrame
    ) -> tuple[dict[str, npt.NDArray[Any]], dict[str, set[int]], str | None]:
        rows = _round_rows(round_table)
        try:
            transforms = io.load_transforms(self.exp, well)
        except FileNotFoundError:
            return {}, {}, f"No count: {well} has no alignment. Run Align rounds on {well}."
        unaligned = [
            name for name, index, _ in rows if index >= len(transforms) or bool(np.isnan(transforms[index]).any())
        ]
        if unaligned:
            verb = "is" if len(unaligned) == 1 else "are"
            return (
                {},
                {},
                f"No count: {', '.join(unaligned)} of {well} {verb} not aligned. Run Align rounds on {well}.",
            )
        anchor_path = io.label_path(self.exp, well, self.exp.anchor_round)
        if not anchor_path.exists():
            anchor = next((name for name, index, _ in rows if index == self.exp.anchor_round), "")
            return (
                {},
                {},
                f"No count: {well} has no segmentation for the anchor round {anchor}. Run Segment nuclei on {well}.",
            )

        cleaned = clear_border(imread(anchor_path))
        rings = make_rings(cleaned, RING_WIDTH, RING_GAP)
        cells = np.where(cleaned > 0, cleaned, rings)
        moved: dict[str, npt.NDArray[Any]] = {}
        saved_covered: dict[str, set[int]] = {}
        for name, index, _ in rows:
            try:
                masks = self._read_round_masks(well, round_table, name, finals=True)
            except ValueError as error:
                return {}, {}, f"No count: {error}"
            moved[name] = cells if index == self.exp.anchor_round else move_to_round(cells, transforms[index])
            saved_covered[name] = _covered(moved[name], masks.finals.values())
        return moved, saved_covered, None

    def _read_round_masks(
        self, well: str, round_table: pd.DataFrame, round_name: str, *, outlines: bool = True, finals: bool = False
    ) -> _Masks:
        rows = round_table[round_table["round_name"] == round_name]
        index = int(rows["round_index"].iloc[0])
        io.require_raw_files(round_table, well)
        shapes = {
            str(signal): _image_shape(Path(self.exp.output_images) / well / str(file_name))
            for signal, file_name in zip(rows["signal"], rows["raw_file_name"], strict=True)
        }
        return _read_masks(
            io.mask_record_path(self.exp, well, index),
            io.mask_path(self.exp, well, index),
            round_name,
            shapes,
            outlines=outlines,
            finals=finals,
        )

    def _round_info(self, round_name: str) -> tuple[int, str]:
        rows = self.round_table[
            (self.round_table["round_name"] == round_name) & (self.round_table["is_align_channel"])
        ]
        if rows.empty:
            raise ValueError(f"{round_name} is not one of {self.well}'s rounds.")
        return int(rows["round_index"].iloc[0]), str(rows["signal"].iloc[0])

    def _round_paths(self, round_name: str) -> dict[str, Path]:
        io.require_raw_files(self.round_table, self.well)
        rows = self.round_table[self.round_table["round_name"] == round_name]
        base_path = Path(self.exp.output_images) / self.well
        return {
            str(signal): base_path / str(file_name)
            for signal, file_name in zip(rows["signal"], rows["raw_file_name"], strict=True)
        }

    def _edited_channel(self) -> _Channel | None:
        if self.shown is None:
            return None
        return self.rounds[self.shown].channels.get(self.edited)

    def _signal_of(self, layer: Any) -> str | None:
        if self.shown is None:
            return None
        channels = self.rounds[self.shown].channels
        for signal in channels:
            if (
                layer is channels[signal].artifacts
                or layer is self.images.get(signal)
                or layer is self.grown.get(signal)
            ):
                return signal
        return None

    def _on_active(self, _event: Any = None) -> None:
        if self._quiet or self.shown is None:
            return
        active = self.viewer.layers.selection.active
        signal = self._signal_of(active)
        if signal is None:
            return
        artifacts = self.rounds[self.shown].channels[signal].artifacts
        if active is not artifacts:
            self._quiet = True
            try:
                self.viewer.layers.selection.active = artifacts
            finally:
                self._quiet = False
        if signal != self.edited:
            self.edited = signal
            self.selection = []
            self._show_channel(signal)
            self._show_selection()
            self._show_values()
            self.prompt("Choose a tool, or paint.")

    def _show_channel(self, signal: str) -> None:
        if self.shown is None:
            return
        self._quiet = True
        try:
            for layer in (self.images[signal], self.rounds[self.shown].channels[signal].artifacts, self.grown[signal]):
                layer.visible = True
        finally:
            self._quiet = False

    def _on_visible(self, round_name: str, signal: str, layer: Any) -> None:
        if self._quiet or round_name != self.shown:
            return
        linked = [self.images[signal], self.rounds[round_name].channels[signal].artifacts, self.grown[signal]]
        self._quiet = True
        try:
            for other in linked:
                if other is not layer:
                    other.visible = layer.visible
        finally:
            self._quiet = False

    def _on_click(self, layer: Labels, event: Any) -> Generator[None]:
        if layer.mode not in CLICK_MODES:
            return
        start = np.asarray(event.pos, dtype=float)
        yield
        while event.type == "mouse_move":
            if np.abs(np.asarray(event.pos, dtype=float) - start).max() > DRAG_PIXELS:
                return
            yield
        number = layer.get_value(event.position, world=True)
        self.select_artifact(int(number or 0), add="Shift" in event.modifiers)

    def _clear_selection(self) -> None:
        self.selection = []
        self._show_selection()
        self._show_values()
        self.prompt("Choose a tool, or paint.")

    def _bind_clear(self, layer: Labels) -> None:
        if layer.mode in CLICK_MODES:
            layer.bind_key(KEY_CLEAR, lambda _layer: self._clear_selection(), overwrite=True)
        else:
            layer.bind_key(KEY_CLEAR, None, overwrite=True)

    def _show_selection(self) -> None:
        channel = self._edited_channel()
        if channel is None:
            return
        data = np.asarray(channel.artifacts.data)
        present = _present(data)
        self.selection = [number for number in self.selection if number in present]
        shown = np.isin(data, self.selection) if self.selection else np.zeros(data.shape, dtype=bool)
        self.highlight.data = shown.astype(np.uint8)

    def _show_values(self) -> None:
        channel = self._edited_channel()
        if channel is None:
            return
        spread, pad = channel.values[self.selection[0]] if self.selection else (channel.spread, channel.pad)
        for slider, value in ((self.spread_slider, spread), (self.pad_slider, pad)):
            with slider.changed.blocked():
                slider.value = value


def _round_rows(round_table: pd.DataFrame) -> list[tuple[str, int, str]]:
    rows = round_table[round_table["is_align_channel"]].sort_values(by="round_index")
    return [
        (str(name), int(index), str(signal))
        for name, index, signal in zip(
            rows["round_name"].tolist(), rows["round_index"].tolist(), rows["signal"].tolist(), strict=True
        )
    ]


def _image_shape(path: Path) -> tuple[int, ...]:
    with TiffFile(path) as tiff:
        return tuple(tiff.series[0].shape)


def _read_masks(
    record_path: Path,
    mask_path: Path,
    round_name: str,
    shapes: dict[str, tuple[int, ...]],
    *,
    outlines: bool,
    finals: bool,
) -> _Masks:
    refused = f"{round_name}'s masks can't be used:"
    channel_values = dict.fromkeys(shapes, (0, 0))
    artifact_values: dict[str, Values] = {signal: {} for signal in shapes}
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if not isinstance(record["edited"], bool):
                raise TypeError(record["edited"])
            masks = record.get("masks", {})
            for signal in shapes:
                entry = masks.get(signal, {})
                channel_values[signal] = (int(entry.get("spread", 0)), int(entry.get("pad", 0)))
                artifact_values[signal] = {
                    int(number): (int(saved["spread"]), int(saved["pad"]))
                    for number, saved in entry.get("artifacts", {}).items()
                }
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            raise ValueError(f"{refused} the mask record can't be read.") from error

    outline_arrays = {signal: np.zeros(shape, dtype=np.uint16) for signal, shape in shapes.items()}
    final_arrays = {signal: np.zeros(shape, dtype=np.bool_) for signal, shape in shapes.items()}
    if mask_path.exists() and (outlines or finals):
        with np.load(mask_path) as fh:
            for signal, shape in shapes.items():
                wanted = [
                    ("outlines_", outlines, "outlines", "outlines are"),
                    ("final_", finals, "final mask", "final mask is"),
                ]
                for prefix, read, what, subject in wanted:
                    if not read:
                        continue
                    key = f"{prefix}{signal}"
                    if key not in fh.files:
                        raise ValueError(f"{refused} the masks file has no {what} for {signal}.")
                    array = fh[key]
                    if array.shape != shape:
                        raise ValueError(
                            f"{refused} the {signal} {subject} {array.shape[0]} x {array.shape[1]} "
                            f"and the images {shape[0]} x {shape[1]}."
                        )
                    if prefix == "outlines_":
                        outline_arrays[signal] = array.astype(np.uint16, copy=False)
                    else:
                        final_arrays[signal] = array.astype(np.bool_, copy=False)
    return _Masks(
        outlines=outline_arrays, finals=final_arrays, channel_values=channel_values, artifact_values=artifact_values
    )


def _present(data: npt.NDArray[Any]) -> set[int]:
    return {number for number, box in enumerate(ndi.find_objects(np.asarray(data)), start=1) if box is not None}


def _sync_values(channel: _Channel) -> None:
    for number in _present(channel.artifacts.data):
        channel.values.setdefault(number, (channel.spread, channel.pad))


def _with_defaults(channel: _Channel) -> defaultdict[int, tuple[int, int]]:
    spread, pad = channel.spread, channel.pad
    return defaultdict(lambda: (spread, pad), channel.values)


def _covered(cells: npt.NDArray[Any], masks: Iterable[npt.NDArray[np.bool_]]) -> set[int]:
    union = None
    for mask in masks:
        if not mask.any():
            continue
        union = mask.copy() if union is None else union | mask
    if union is None:
        return set()
    return {int(number) for number in np.unique(cells[union]) if number != 0}


def _outline(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint8]:
    filled = mask.view(np.uint8)
    inner = cv2.erode(filled, np.ones((3, 3), dtype=np.uint8), iterations=OUTLINE_WIDTH)
    return filled & ~inner.astype(bool)


def _artifact_list(numbers: list[int]) -> str:
    listed = ", ".join(f"{number:,}" for number in numbers)
    return f"artifact {listed}" if len(numbers) == 1 else f"artifacts {listed}"
