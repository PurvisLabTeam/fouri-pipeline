"""@TODO."""

import shutil
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from magicgui.widgets import PushButton, Widget
from napari.layers import Image, Labels
from napari.utils.colormaps import CyclicLabelColormap
from tifffile import imread

from fouri import io
from fouri.viewers.base import BaseViewer, KeyboardShortcuts

LABEL_COLORS: CyclicLabelColormap = CyclicLabelColormap(
    colors=[
        "transparent",
        "cyan",
        "springgreen",
        "chartreuse",
        "deepskyblue",
        "turquoise",
        "aquamarine",
        "lime",
        "palegreen",
    ]
)
SELECTED_COLORS: CyclicLabelColormap = CyclicLabelColormap(colors=["transparent", "white"])

KEY_DELETE = "Delete"
KEY_NEW = "Shift+N"
KEY_MERGE = "Shift+M"
KEY_CLEAR = "Escape"
KEY_REDO = "Ctrl+Shift+Z"

CONTRAST_PERCENTILES = (1, 99.5)
CONTRAST_SAMPLE_STEP = 8
DRAG_PIXELS = 4
CLICK_MODES = ("pan_zoom", "pick")


class SegmentationViewer(BaseViewer):
    title = "Segmentation review"

    channels: dict[str, Image]
    shown_signals: set[str]
    masks: dict[str, Labels]
    changed: set[str]
    selection: list[int]
    highlight: Labels

    def _round_choices(self, round_table: pd.DataFrame) -> list[str]:
        choices = [
            str(row.round_name)
            for row in round_table[round_table["is_align_channel"]].sort_values(by="round_index").itertuples()
        ]
        if not choices:
            raise ValueError("There are no rounds to edit.")
        return choices

    def _start_round(self) -> str | None:
        rounds = io.read_info_file(self.exp)
        anchor = str(rounds.iloc[self.exp.anchor_round]["round_name"])
        return anchor

    def _build_layers(self) -> None:
        self.channels = {}
        self.masks = {}
        self.changed = set()
        self.shown_signals = set()
        self.selection = []
        self.highlight = self.viewer.add_labels(
            np.zeros((1, 1), dtype=np.uint8), name="selected cells", colormap=SELECTED_COLORS, opacity=0.5
        )

    def _load_well(self, well: str, round_table: pd.DataFrame) -> None:
        for layer in self.masks.values():
            self.viewer.layers.remove(layer)

        if self.channels:
            self.shown_signals = {signal for signal, layer in self.channels.items() if layer.visible}
        for layer in self.channels.values():
            self.viewer.layers.remove(layer)

        self.channels = {}
        self.masks = {}
        self.changed = set()
        self.selection = []
        self.highlight.data = np.zeros((1, 1), dtype=np.uint8)

    def _round_channels(self, round_name: str) -> list[tuple[str, Path]]:
        row = self.round_table[self.round_table["round_name"] == round_name]
        base_path = Path(self.exp.output_images) / self.well
        return [
            (channel, base_path / fname) for channel, fname in zip(row["signal"], row["raw_file_name"], strict=True)
        ]

    def _load_round(self, round_name: str) -> None:
        round_index, align_signal = self._round_info(round_name)
        channels = {channel: imread(path) for channel, path in self._round_channels(round_name)}
        if round_name not in self.masks:
            labels_path = io.label_path(self.exp, self.well, round_index)
            if not labels_path.exists():
                raise FileNotFoundError(
                    f"{self.well} has no segmentation masks for {round_name}. Run segment nuclei on it first."
                )
            labels = imread(labels_path)
            layer = self.viewer.add_labels(labels, name=f"{round_name} masks", colormap=LABEL_COLORS)
            layer.contour = 2
            layer.events.paint.connect(lambda _event: self.changed.add(round_name))
            layer.events.paint.connect(lambda _event, layer=layer: self._show_selection(layer))
            layer.mouse_drag_callbacks.append(self._on_click)
            layer.bind_key(KeyboardShortcuts.UNDO, lambda _layer: self.undo(), overwrite=True)
            layer.bind_key(KEY_REDO, lambda _layer: self.redo(), overwrite=True)
            layer.events.mode.connect(lambda _event, layer=layer: self._bind_clear(layer))
            self._bind_clear(layer)
            self.masks[round_name] = layer

        layer = self.masks[round_name]

        first_round = not self.channels
        if self.channels:
            self.shown_signals = {signal for signal, shown in self.channels.items() if shown.visible}
        self.shown_signals = self.shown_signals.intersection(channels) or {align_signal}

        for old in self.channels.values():
            self.viewer.layers.remove(old)
        self.channels = {}
        for signal, image in channels.items():
            channel = Image(
                image,
                name=f"{round_name} {signal}",
                colormap="magma",
                contrast_limits=_contrast_limits(image),
                visible=signal in self.shown_signals,
            )
            self.viewer.layers.insert(0, channel)
            self.channels[signal] = channel

        for name, mask in self.masks.items():
            mask.visible = name == round_name

        self.selection = []
        self._show_selection(layer)
        self.viewer.layers.selection.active = layer
        if first_round:
            self.viewer.reset_view()

        self.prompt(
            f"{self.well}, {round_name}{' (the anchor round)' if round_index == self.exp.anchor_round else ''}: "
            f"{len(np.unique(layer.data[layer.data != 0])):,} cells."
        )

    def _round_info(self, round_name: str) -> tuple[int, str]:
        row = self.round_table[(self.round_table["round_name"] == round_name) & (self.round_table["is_align_channel"])]
        if row.empty:
            raise ValueError(f"{round_name} is not one of {self.well}'s rounds.")
        index = int(row["round_index"].iloc[0])
        signal = str(row["signal"].iloc[0])
        return index, signal

    def save(self) -> None:
        unsaved = self._unsaved_rounds()
        if not unsaved:
            self.prompt(f"Nothing to save on {self.well}.")
            return
        kept: list[str] = []
        anchor: list[str] = []
        others: list[str] = []
        for round_name in unsaved:
            round_index, _ = self._round_info(round_name)
            path = io.label_path(self.exp, self.well, round_index)
            original = io.original_label_path(self.exp, self.well, round_index)
            made_original = path.exists() and not original.exists()
            if made_original:
                original.parent.mkdir(parents=True, exist_ok=True)
                with io.written_whole(original) as partial:
                    shutil.copyfile(path, partial)
            try:
                io.write_tiff(path, np.asarray(self.masks[round_name].data))
            except Exception:
                if made_original:
                    original.unlink(missing_ok=True)
                raise
            if made_original:
                kept.append(round_name)
            self.changed.discard(round_name)
            (anchor if round_index == self.exp.anchor_round else others).append(round_name)

        lines = []
        if kept:
            lines.append(f"Kept Cellpose's original segmentation masks for {', '.join(kept)}.")
        saved = ", ".join(anchor + others)
        if anchor and others:
            lines.append(
                f"Saved {self.well}'s segmentation masks for {saved}. Run Measure cells on {self.well} with Overwrite "
                f"to use {anchor[0]}'s; {', '.join(others)}'s only matter if you rerun Align rounds with Overwrite, "
                "which replaces hand alignments."
            )
        elif anchor:
            lines.append(
                f"Saved {self.well}'s segmentation masks for {saved}. "
                f"Run Measure cells on {self.well} with Overwrite to use them."
            )
        else:
            lines.append(
                f"Saved {self.well}'s segmentation masks for {saved}. They only matter if you rerun Align rounds "
                "with Overwrite, which replaces hand alignments."
            )
        self.prompt(" ".join(lines))

    def undo(self) -> None:
        layer = self.masks.get(self.round_name)
        if layer is not None:
            layer.undo()
            self.changed.add(self.round_name)
            self._show_selection(layer)

    def redo(self) -> None:
        layer = self.masks.get(self.round_name)
        if layer is not None:
            layer.redo()
            self.changed.add(self.round_name)
            self._show_selection(layer)

    def _unsaved_rounds(self) -> list[str]:
        unsaved = []
        for round_name in self.round_names:
            if round_name not in self.changed:
                continue
            round_index, _ = self._round_info(round_name)
            path = io.label_path(self.exp, self.well, round_index)
            if not path.exists() or not np.array_equal(imread(path), self.masks[round_name].data):
                unsaved.append(round_name)
        return unsaved

    def delete_cells(self) -> None:
        layer = self.masks.get(self.round_name)
        if layer is None:
            return
        if not self.selection:
            self.prompt("Select cells to delete: click one, Shift+click more.")
            return
        deleted = self.selection
        self.selection = []
        layer.data_setitem(np.nonzero(np.isin(layer.data, deleted)), 0)
        self._show_selection(layer)
        self.prompt(f"Deleted {_cell_list(deleted)}.")

    def new_cell(self) -> None:
        layer = self.masks.get(self.round_name)
        if layer is None:
            return
        number = int(layer.data.max()) + 1
        self.selection = []
        self._show_selection(layer)
        layer.selected_label = number
        layer.mode = "paint"
        self.prompt(f"Painting new cell {number:,}. Paint it, then choose another tool.")

    def select_cell(self, label: int, *, add: bool) -> None:
        layer = self.masks.get(self.round_name)
        if layer is None:
            return
        if label == 0:
            self._clear_selection()
            return
        if not add:
            self.selection = [label]
        elif label in self.selection:
            self.selection.remove(label)
        else:
            self.selection.append(label)
        self._show_selection(layer)
        if self.selection:
            self.prompt(
                f"Selected {_cell_list(self.selection)}. Shift+click more, Delete to delete them, "
                "or Shift+M to merge them."
            )
        else:
            self.prompt("Choose a tool, or paint.")

    def merge_cells(self) -> None:
        layer = self.masks.get(self.round_name)
        if layer is None:
            return
        if len(self.selection) < 2:
            self.prompt("Select at least two cells to merge.")
            return
        if not _all_touch(layer.data, self.selection):
            self.prompt("The selected cells don't all touch, so they can't merge into one cell.")
            return
        keep, *merged = self.selection
        self.selection = []
        layer.data_setitem(np.nonzero(np.isin(layer.data, merged)), keep)
        self._show_selection(layer)
        self.prompt(f"Merged {_cell_list(merged)} into cell {keep:,}.")

    def _on_click(self, layer: Labels, event: Any) -> Generator[None]:
        if layer.mode not in CLICK_MODES:
            return
        start = np.asarray(event.pos, dtype=float)
        yield
        while event.type == "mouse_move":
            if np.abs(np.asarray(event.pos, dtype=float) - start).max() > DRAG_PIXELS:
                return
            yield
        label = layer.get_value(event.position, world=True)
        self.select_cell(int(label or 0), add="Shift" in event.modifiers)

    def _clear_selection(self) -> None:
        layer = self.masks.get(self.round_name)
        if layer is None:
            return
        self.selection = []
        self._show_selection(layer)
        self.prompt("Choose a tool, or paint.")

    def _bind_clear(self, layer: Labels) -> None:
        if layer.mode in CLICK_MODES:
            layer.bind_key(KEY_CLEAR, lambda _layer: self._clear_selection(), overwrite=True)
        else:
            layer.bind_key(KEY_CLEAR, None, overwrite=True)

    def _show_selection(self, layer: Labels) -> None:
        data = layer.data
        self.selection = [cell for cell in self.selection if (data == cell).any()]
        shown = np.isin(data, self.selection) if self.selection else np.zeros(data.shape, dtype=bool)
        self.highlight.data = shown.astype(np.uint8)

    def _tool_widgets(self) -> list[Widget]:
        tools: list[tuple[str, str, Callable[[], None]]] = [
            ("Delete cell", KEY_DELETE, self.delete_cells),
            ("New cell", KEY_NEW, self.new_cell),
            ("Merge cells", KEY_MERGE, self.merge_cells),
        ]
        buttons: list[Widget] = []
        for text, key, action in tools:
            button = PushButton(text=f"{text} ({key})")
            button.changed.connect(action)
            buttons.append(button)
        return buttons

    def _extra_keys(self) -> dict[str, Callable[[], None]]:
        return {KEY_DELETE: self.delete_cells, KEY_NEW: self.new_cell, KEY_MERGE: self.merge_cells}

    def _extra_shortcuts(self) -> list[str]:
        return [
            "Click: select a cell; Shift+click: add or take out one",
            f"{KEY_CLEAR} or a click on the background: clear the selection",
            "Drag: pan",
            f"{KEY_DELETE}: delete the selected cells",
            f"{KEY_NEW}: new cell",
            f"{KEY_MERGE}: merge the selected cells",
        ]


def _contrast_limits(image: npt.NDArray[Any]) -> tuple[float, float]:
    sample = image[::CONTRAST_SAMPLE_STEP, ::CONTRAST_SAMPLE_STEP]
    low, high = (float(value) for value in np.percentile(sample, CONTRAST_PERCENTILES))
    return low, max(high, low + 1)


def _cell_list(cells: list[int]) -> str:
    numbers = ", ".join(f"{cell:,}" for cell in cells)
    return f"cell {numbers}" if len(cells) == 1 else f"cells {numbers}"


def _all_touch(data: npt.NDArray[Any], cells: list[int]) -> bool:
    rows, cols = np.nonzero(np.isin(data, cells))
    box = data[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    picked = np.where(np.isin(box, cells), box, 0)
    pairs: set[tuple[int, int]] = set()
    for first, second in (
        (picked[:, :-1], picked[:, 1:]),
        (picked[:-1, :], picked[1:, :]),
        (picked[:-1, :-1], picked[1:, 1:]),
        (picked[:-1, 1:], picked[1:, :-1]),
    ):
        touching = (first != 0) & (second != 0) & (first != second)
        pairs.update(zip(first[touching].tolist(), second[touching].tolist(), strict=True))
    joined = {cells[0]}
    grew = True
    while grew:
        grew = False
        for one, other in pairs:
            if (one in joined) != (other in joined):
                joined.update((one, other))
                grew = True
    return joined >= set(cells)
