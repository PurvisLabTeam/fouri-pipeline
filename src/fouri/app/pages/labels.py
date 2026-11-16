"""The Labels & gates page."""

from __future__ import annotations

import copy
import itertools
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from nicegui import run, ui
from nicegui.events import GenericEventArguments, ValueChangeEventArguments

from fouri.app.cells import Cells, load_cells
from fouri.app.embeddings import sample_names, sample_number
from fouri.app.figures import cells_line
from fouri.app.gates import (
    FREEHAND,
    NO_CLASS,
    RECTANGLE,
    SPLITS,
    area,
    class_color,
    drawn_points,
    gate_figure,
    plot_ranges,
    plotted,
    split_gates,
    to_drawn,
    to_saved,
)
from fouri.app.names import MEASURES, VALUE, marker_menu
from fouri.app.rules import kept_cells, save_labels
from fouri.app.shell import empty_state, page_frame
from fouri.app.state import STATE
from fouri.app.theme import restyle
from fouri.config import Experiment
from fouri.features import Threshold, middle_values, suggest_cutoffs
from fouri.io import read_info_file
from fouri.labels import LabelRule, label_cells, saved_labels
from fouri.plots import plot_threshold_histogram

LINE = (
    "Label columns made by a cut-off on one feature or by gates on two. "
    "Build cell table adds the labels switched on to the cell table."
)
ONE, TWO = "One feature", "Two features"
SAMPLES_LINE = "The cut-offs are set on these samples' cells and applied to every cell."
SDS_LABEL = "Standard deviations kept for the suggestions and the bars"
GATES_TITLE = "Gates (a cell in overlapping gates takes the first; gates with the same name form one class)"
GATES_HINT = (
    "Draw a gate freehand or with the rectangle tool in the toolbar. To change a gate, delete it and draw it again."
)
NEW_TAB = "new"
TABS_PROPS = "dense align=left no-caps outside-arrows mobile-arrows"
NEW_TAB_LABEL = "+ New label"
UNNAMED = "unnamed"
USE_LABEL = "Add to cell table"
SHOW_LINES = "Show suggestions"
OTSU_GAP = "Otsu and valley"
SUBTITLE_JOIN = " · "
RATIO = re.compile(r"\s*(\w+)\s*/\s*(\w+)\s*")
RATIO_MARKERS = ("prb", "rb")
NO_SAMPLE = "Pick at least one sample."
BAD_SDS = "Standard deviations: enter a number above 0."
NO_NAME = "Name the label column."
INCLUDED = "included"
INCLUDED_TAKEN = 'Build cell table adds a column "included". Choose another name.'
NO_CLASS_NAME = "Name every class."
NO_GATE = "Draw at least one gate."
NEW_FEATURE = "nuc_area"
NEW_CLASSES = ("low", "high")
THREE_CLASSES = ("low", "mid", "high")
MIDDLE_CLASS = "mid"
NEW_SDS = 3.0
NEW_GATE = "gate {number}"
OPENING_MEASURE = MEASURES["nuc_mean"]
SIGNIFICANT_DIGITS = 3
TINY_SHARE = 0.0005
EDGE_PAD = 0.05
LOG = "log10 "
SDS_STEP = 0.5
TYPING_MS = 400
MODE_SECONDS = 0.25
SECTION_TITLE = "text-sm font-medium"
NO_SELECTING = ["select2d", "lasso2d"]
HISTOGRAM_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": NO_SELECTING,
    "edits": {"shapePosition": True},
    "toImageButtonOptions": {"format": "png", "filename": "labels-histogram", "scale": 2},
}
GATES_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": NO_SELECTING,
    "toImageButtonOptions": {"format": "png", "filename": "labels-gates", "scale": 2},
}
SHAPE_KINDS = {"rect": RECTANGLE, "path": FREEHAND}
DRAG_MODES = {RECTANGLE: "drawrect", FREEHAND: "drawclosedpath"}

Menu = dict[str, dict[str, str]]
Suggestions = dict[Threshold, list[float]]


@dataclass
class _Threshold:
    feature: str | None
    samples: list[int]
    cutoffs: list[float]
    classes: list[str]
    sds: float | None


@dataclass
class _Gates:
    x: str
    y: str
    log_x: bool
    log_y: bool
    gates: list[dict[str, Any]]


@dataclass
class _Label:
    column: str
    use: bool
    mode: LabelRule
    threshold: _Threshold | None = None
    gates: _Gates | None = None
    saved_at: int | None = None


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _numbers(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [number for number in (_number(item) for item in value) if number is not None]


def _whole_numbers(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value if isinstance(item, int) and not isinstance(item, bool)]


def _points(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    points = []
    for point in value:
        pair = point if isinstance(point, list | tuple) and len(point) == 2 else (None, None)
        x, y = _number(pair[0]), _number(pair[1])
        points.append([x if x is not None else float("nan"), y if y is not None else float("nan")])
    return points


def _rounded(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def _as_saved(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _good_sds(sds: float | None) -> bool:
    return sds is not None and np.isfinite(sds) and sds > 0


def _reason(error: Exception) -> str:
    text = error.strerror if isinstance(error, OSError) and error.strerror else str(error)
    return text.rstrip(".")


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith(".") else text + "."


def _percent(count: int, total: int) -> str:
    return f"{count / total * 100:.1f} %" if total else "0.0 %"


def _counts(classes: pd.Categorical) -> tuple[dict[str, int], int]:
    counts = pd.Series(classes).value_counts(sort=False)
    found = {str(name): int(count) for name, count in counts.items()}
    return found, len(classes) - sum(found.values())


def _shares(classes: pd.Categorical) -> str:
    found, none = _counts(classes)
    total = len(classes)
    parts = [f"{name} {_percent(count, total)}" for name, count in found.items()]
    if none:
        parts.append(f"{NO_CLASS} {_percent(none, total)}")
    return " · ".join(parts)


def _two_classes(classes: list[str]) -> list[str]:
    if len(classes) == 3:
        return [classes[0], classes[2]]
    return list(classes) if len(classes) == 2 else list(NEW_CLASSES)


def _three_classes(classes: list[str]) -> list[str]:
    if len(classes) == 2:
        return [classes[0], MIDDLE_CLASS, classes[1]]
    return list(classes) if len(classes) == 3 else list(THREE_CLASSES)


def _saved_list(exp: Experiment) -> list[dict[str, Any]]:
    app = exp.config.get("app")
    labels = app.get("labels") if isinstance(app, dict) else None
    return [dict(label) for label in labels if isinstance(label, dict)] if isinstance(labels, list) else []


def _place(menu: Menu, column: str | None) -> tuple[str, str] | None:
    for marker, measures in menu.items():
        for measure, name in measures.items():
            if name == column:
                return marker, measure
    return None


def _opening_measure(options: list[str]) -> str:
    return OPENING_MEASURE if OPENING_MEASURE in options else options[0]


def _picker(menu: Menu, label: str, column: str | None, on_pick: Callable[[str], Awaitable[None]]) -> None:
    menu = dict(menu)
    place = _place(menu, column)
    if place is None and column:
        menu[column] = {VALUE: column}
        place = column, VALUE
    marker, measure = place if place is not None else (None, None)
    with ui.row().classes("w-full no-wrap gap-1"):
        marker_box = ui.select(list(menu), value=marker, label=label, with_input=True).classes("w-7/12")
        measure_box = ui.select(list(menu[marker]) if marker else [], value=measure, label="measure").classes("w-5/12")

    async def new_marker(e: ValueChangeEventArguments[Any]) -> None:
        if not e.value:
            return
        options = list(menu[e.value])
        kept = measure_box.value if measure_box.value in options else _opening_measure(options)
        measure_box.set_options(options, value=kept)
        await on_pick(menu[e.value][kept])

    async def new_measure(e: ValueChangeEventArguments[Any]) -> None:
        if e.value and marker_box.value and e.value in menu[marker_box.value]:
            await on_pick(menu[marker_box.value][e.value])

    marker_box.on_value_change(new_marker)
    measure_box.on_value_change(new_measure)


def _edges(values: npt.NDArray[np.float64]) -> tuple[float, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    low, high = float(finite.min()), float(finite.max())
    pad = (high - low) * EDGE_PAD if high > low else 1.0
    return low - pad, high + pad


def _suggest(values: npt.NDArray[np.float64], sds: float) -> Suggestions:
    if values.size == 0:
        return {}
    try:
        return suggest_cutoffs(values, sds)
    except (ValueError, np.linalg.LinAlgError):
        return {}


def _ratio_feature(exp: Experiment, cells: Cells) -> str | None:
    for derived in exp.derived_features:
        name, formula = (derived.get("name"), derived.get("formula")) if isinstance(derived, dict) else (None, None)
        if not isinstance(name, str) or not isinstance(formula, str) or name not in cells.table.columns:
            continue
        match = RATIO.fullmatch(formula)
        if match is None:
            continue
        top, bottom = cells.features.get(match[1]), cells.features.get(match[2])
        if top is None or bottom is None:
            continue
        if (top.marker.casefold(), bottom.marker.casefold()) == RATIO_MARKERS:
            return name
    return None


def _histogram(
    values: npt.NDArray[np.float64], cutoffs: list[float], sds: float, title: str, lines: bool
) -> tuple[go.Figure, int]:
    fig = plot_threshold_histogram(values, cutoffs, sds=sds, title=title, x_title=title)
    if not lines:
        fig.data = tuple(trace for trace in fig.data if trace.type == "histogram")
        subtitle = fig.layout.title.subtitle.text or ""
        kept = [part for part in subtitle.split(SUBTITLE_JOIN) if not part.startswith(OTSU_GAP)]
        fig.update_layout(title_subtitle_text=SUBTITLE_JOIN.join(kept))
    return fig, int(values.size - middle_values(values, sds).size)


def _otsu(values: npt.NDArray[np.float64]) -> float | None:
    suggestions = _suggest(values, NEW_SDS)
    return suggestions[Threshold.OTSU][0] if Threshold.OTSU in suggestions else None


async def _ask(question: str, yes: str) -> bool:
    with ui.dialog() as dialog, ui.card().classes("p-5 gap-3 max-w-lg"):
        ui.label(question).classes("text-sm")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
            ui.button(yes, on_click=lambda: dialog.submit(True))
    answer = await dialog
    dialog.delete()
    return answer is True


class _LabelsPage:
    def __init__(self, exp: Experiment, cells: Cells, menu: Menu, kept: pd.Series | None, problem: str | None) -> None:
        self.exp = exp
        self.cells = cells
        self.menu = menu
        self.kept = kept
        self.filter_problem = problem
        self.every_cell = False
        self.table = cells.table
        self.covered = 0
        self.filtered = 0
        self.samples = sample_names(cells.table, exp)
        self.sample_ids = (
            cells.table["sample_ID"].astype(str).map(sample_number) if "sample_ID" in cells.table else None
        )
        self.saved: list[dict[str, Any]] = []
        self.unreadable = ""
        self.label = _Label("", True, LabelRule.THRESHOLD)
        self.opened: dict[str, Any] = {}
        self.classes: pd.Categorical | None = None
        self.problem = ""
        self.suggestions: dict[tuple[Any, ...], Suggestions] = {}
        self.guides: dict[tuple[Any, ...], tuple[float, float] | None] = {}
        self.guide_lines: tuple[float, float] | None = None
        self.suggest = False
        self.ratio = _ratio_feature(exp, cells)
        self.lines = False
        self.split = next(iter(SPLITS))
        self.dragmode = DRAG_MODES[FREEHAND]
        self.asking = False
        self.ticket = 0
        self.figure: go.Figure | None = None
        self.config: dict[str, Any] = HISTOGRAM_CONFIG
        self.dark = STATE.dark
        self.swatches: list[tuple[dict[str, Any], ui.element]] = []
        self.set_shown()
        self.read_saved()

    def set_shown(self) -> None:
        table = self.cells.table
        if self.every_cell or self.kept is None:
            self.table, self.covered, self.filtered = table, 0, 0
            return
        inside = self.kept.to_numpy(dtype=bool)
        self.table = table[inside]
        self.covered = int((self.cells.covered.to_numpy(dtype=bool) & ~inside).sum())
        self.filtered = len(table) - len(self.table) - self.covered

    def read_saved(self) -> None:
        self.saved = _saved_list(self.exp)
        try:
            saved_labels(self.exp)
            self.unreadable = ""
        except ValueError as error:
            self.unreadable = _sentence(str(error))

    def build(self) -> None:
        with ui.column().classes("gap-1"):
            ui.label("Labels & gates").classes("page-title")
            ui.label(LINE).classes("text-sm muted")
        empty_state(self.cells)
        with ui.row().classes("w-full items-center no-wrap gap-4"):
            self.tabs = ui.tabs(on_change=self.tab_picked).props(TABS_PROPS).classes("grow min-w-0")
            ui.switch("Show all cells", value=self.every_cell, on_change=self.show_all).classes("shrink-0")
        self.unreadable_line = ui.label().classes("text-sm text-negative")
        with ui.row().classes("w-full no-wrap items-start gap-6"):
            self.editor = ui.column().classes("w-80 shrink-0 gap-2")
            with ui.column().classes("grow min-w-0 gap-1"):
                if self.filter_problem:
                    ui.label(self.filter_problem).classes("text-sm text-warning")
                self.lines_switch = ui.switch(SHOW_LINES, value=self.lines, on_change=self.set_lines)
                self.plot = ui.plotly({"data": [], "layout": {}}).classes("w-full h-[78vh]")
                self.plot.on("plotly_relayout", self.relayout)
                self.message = ui.label().classes("text-sm muted")
                with ui.row().classes("items-center gap-2"):
                    self.spinner = ui.spinner(size="sm")
                    self.line = ui.label().classes("text-sm muted")
        self.refresh_tabs()
        ui.timer(MODE_SECONDS, self.follow_mode)
        ui.timer(0, self.start, once=True)

    async def start(self) -> None:
        await self.open(0 if self.saved else None)

    def tab_key(self) -> str:
        index = self.label.saved_at
        return NEW_TAB if index is None or index >= len(self.saved) else str(index)

    def refresh_tabs(self) -> None:
        self.unreadable_line.text = f"The saved labels can't be read: {self.unreadable} Open one and rename it."
        self.unreadable_line.set_visibility(bool(self.unreadable))
        self.tabs.clear()
        with self.tabs:
            for index, saved in enumerate(self.saved):
                ui.tab(str(index), label=_text(saved.get("column")).strip() or UNNAMED)
            ui.tab(NEW_TAB, label=NEW_TAB_LABEL)
        self.tabs.set_value(self.tab_key())

    async def tab_picked(self, e: ValueChangeEventArguments[Any]) -> None:
        key = _text(e.value)
        if key == self.tab_key():
            return
        if self.asking or (self.edited() and not await self.ask_discard()):
            self.tabs.set_value(self.tab_key())
            return
        await self.open(int(key) if key.isdigit() and int(key) < len(self.saved) else None)

    async def switch(self, value: bool) -> None:
        index = self.label.saved_at
        saved = copy.deepcopy(self.saved)
        if index is None or index >= len(saved) or (saved[index].get("use") is True) == value:
            return
        saved[index]["use"] = value
        try:
            self.exp = save_labels(self.exp, saved)
        except (OSError, ValueError) as error:
            ui.notify(f"Not saved: {_reason(error)}", type="negative")
            self.build_editor()
            await self.update()
            return
        self.read_saved()
        self.label.use = value
        self.opened["use"] = value

    async def show_all(self, e: ValueChangeEventArguments[Any]) -> None:
        self.every_cell = bool(e.value)
        self.set_shown()
        await self.update()

    def state(self) -> dict[str, Any]:
        label = self.label
        settings = label.threshold if label.mode is LabelRule.THRESHOLD else label.gates
        return copy.deepcopy(
            {
                "column": label.column,
                "use": label.use,
                "mode": label.mode,
                "settings": asdict(settings) if settings is not None else None,
            }
        )

    def edited(self) -> bool:
        return self.state() != self.opened

    def name_for_question(self) -> str:
        column = self.label.column.strip()
        return f'"{column}"' if column else "the new label"

    async def ask_discard(self) -> bool:
        self.asking = True
        try:
            return await _ask(f"Discard the changes to {self.name_for_question()}?", "Discard")
        finally:
            self.asking = False

    async def open(self, index: int | None) -> None:
        if index is None or index >= len(self.saved):
            self.label = _Label("", True, LabelRule.THRESHOLD, threshold=await self.new_threshold())
        else:
            self.label = self.loaded(index, self.saved[index])
        self.opened = self.state()
        self.follow_threshold()
        self.refresh_tabs()
        self.build_editor()
        await self.update()

    def follow_threshold(self) -> None:
        threshold = self.label.threshold if self.label.mode is LabelRule.THRESHOLD else None
        self.lines = threshold is not None and self.ratio is not None and threshold.feature == self.ratio
        self.lines_switch.set_value(self.lines)
        self.lines_switch.set_visibility(self.label.mode is LabelRule.THRESHOLD)

    async def set_lines(self, e: ValueChangeEventArguments[Any]) -> None:
        if bool(e.value) == self.lines:
            return
        self.lines = bool(e.value)
        await self.draw()

    def loaded(self, index: int, saved: Mapping[str, Any]) -> _Label:
        column, use = _text(saved.get("column")), saved.get("use") is True
        if saved.get("rule") == LabelRule.GATES:
            log_x, log_y = saved.get("x_log10") is True, saved.get("y_log10") is True
            listed = saved.get("gates")
            gates = [
                {
                    "class": _text(gate.get("class")),
                    "shape": _text(gate.get("shape")),
                    "points": to_drawn(_points(gate.get("points")), log_x, log_y),
                }
                for gate in (listed if isinstance(listed, list) else [])
                if isinstance(gate, Mapping)
            ]
            return _Label(
                column,
                use,
                LabelRule.GATES,
                gates=_Gates(_text(saved.get("x")), _text(saved.get("y")), log_x, log_y, gates),
                saved_at=index,
            )
        samples = (
            [sample for sample in _whole_numbers(saved.get("set_on_samples")) if sample in self.samples]
            if "set_on_samples" in saved
            else list(self.samples)
        )
        sds = _number(saved.get("sds")) if "sds" in saved else NEW_SDS
        classes = saved.get("classes")
        threshold = _Threshold(
            feature=_text(saved.get("feature")) or None,
            samples=samples,
            cutoffs=_numbers(saved.get("cutoffs")),
            classes=[_text(name) for name in classes] if isinstance(classes, list) else [],
            sds=sds,
        )
        return _Label(column, use, LabelRule.THRESHOLD, threshold=threshold, saved_at=index)

    async def new_threshold(self) -> _Threshold:
        feature = self.ratio or (
            NEW_FEATURE if NEW_FEATURE in self.cells.features else next(iter(self.cells.features), None)
        )
        threshold = _Threshold(feature, list(self.samples), [], list(NEW_CLASSES), NEW_SDS)
        suggestions = await self.suggestions_for(threshold)
        if Threshold.OTSU in suggestions:
            threshold.cutoffs = [_rounded(suggestions[Threshold.OTSU][0])]
        return threshold

    def new_gates(self) -> _Gates:
        markers = list(self.menu)
        first = markers[0]
        second = markers[1] if len(markers) > 1 else markers[0]
        x = self.menu[first][_opening_measure(list(self.menu[first]))]
        y = self.menu[second][_opening_measure(list(self.menu[second]))]
        return _Gates(x, y, False, False, [])

    def picked(self, threshold: _Threshold) -> npt.NDArray[np.bool_]:
        if not self.samples or self.sample_ids is None:
            return np.ones(len(self.table), dtype=bool)
        return self.sample_ids.loc[self.table.index].isin(threshold.samples).to_numpy(dtype=bool)

    def threshold_values(self, threshold: _Threshold) -> tuple[npt.NDArray[np.float64], int, int]:
        feature = threshold.feature
        if feature is None or feature not in self.table.columns:
            return np.array([], dtype=np.float64), 0, 0
        picked = self.picked(threshold)
        chosen = self.table[feature].to_numpy(dtype=np.float64, na_value=np.nan)[picked]
        finite = chosen[np.isfinite(chosen)]
        return finite, len(self.table) - int(picked.sum()), int(chosen.size - finite.size)

    async def suggestions_for(self, threshold: _Threshold) -> Suggestions:
        if not _good_sds(threshold.sds) or (self.samples and not threshold.samples):
            return {}
        assert threshold.sds is not None
        key = (threshold.feature, tuple(threshold.samples), threshold.sds, self.every_cell)
        if key not in self.suggestions:
            values, _, _ = self.threshold_values(threshold)
            self.suggestions[key] = await run.io_bound(_suggest, values, threshold.sds)
        return self.suggestions[key]

    async def guides_for(self, gates: _Gates) -> tuple[float, float] | None:
        key = (gates.x, gates.y, gates.log_x, gates.log_y, self.every_cell)
        if key not in self.guides:
            x = plotted(self.table, gates.x, gates.log_x)
            y = plotted(self.table, gates.y, gates.log_y)
            drawable = np.isfinite(x) & np.isfinite(y)
            gx = await run.io_bound(_otsu, x[drawable])
            gy = await run.io_bound(_otsu, y[drawable])
            self.guides[key] = (gx, gy) if gx is not None and gy is not None else None
        return self.guides[key]

    def spans(self, gates: _Gates) -> tuple[float, float]:
        x_range, y_range = plot_ranges(self.table, self.plot_rule(gates))
        return (
            x_range[1] - x_range[0] if x_range else 0.0,
            y_range[1] - y_range[0] if y_range else 0.0,
        )

    def plot_rule(self, gates: _Gates) -> dict[str, Any]:
        return {"x": gates.x, "y": gates.y, "x_log10": gates.log_x, "y_log10": gates.log_y, "gates": gates.gates}

    def saved_rule(self) -> dict[str, Any]:
        label = self.label
        rule: dict[str, Any] = {"column": label.column.strip(), "use": label.use, "rule": str(label.mode)}
        if label.mode is LabelRule.THRESHOLD:
            threshold = label.threshold
            assert threshold is not None
            if threshold.feature is not None:
                rule["feature"] = threshold.feature
            rule["cutoffs"] = [_as_saved(cut) for cut in threshold.cutoffs]
            rule["classes"] = [name.strip() for name in threshold.classes]
            rule["set_on_samples"] = list(threshold.samples) if self.samples else []
            if threshold.sds is not None:
                rule["sds"] = _as_saved(threshold.sds)
            return rule
        gates = label.gates
        assert gates is not None
        spans = self.spans(gates)
        rule.update({"x": gates.x, "y": gates.y, "x_log10": gates.log_x, "y_log10": gates.log_y})
        rule["gates"] = [
            {
                "class": _text(gate["class"]).strip(),
                "shape": gate["shape"],
                "points": to_saved(gate["points"], gate["shape"], gates.log_x, gates.log_y, spans),
            }
            for gate in gates.gates
        ]
        return rule

    def work_out_classes(self) -> None:
        label = self.label
        self.classes, self.problem = None, ""
        if label.mode is LabelRule.GATES and label.gates is not None and not label.gates.gates:
            self.classes = pd.Categorical([None] * len(self.table), categories=[])
            return
        if label.mode is LabelRule.THRESHOLD and label.threshold is not None:
            names = Counter(name.strip() for name in label.threshold.classes)
            twice = next((name for name, count in names.items() if count > 1), None)
            if twice is not None:
                self.problem = f'Two classes are named "{twice}".'
                return
        try:
            self.classes = label_cells(self.table, self.saved_rule())
        except (ValueError, TypeError, KeyError) as error:
            self.problem = str(error)

    def refusal(self) -> str:
        label = self.label
        column = label.column.strip()
        if not column:
            return NO_NAME
        if column == INCLUDED:
            return INCLUDED_TAKEN
        others = [_text(saved.get("column")) for index, saved in enumerate(self.saved) if index != label.saved_at]
        if column in others:
            return f'A saved label already makes the column "{column}".'
        if column in self.cells.table.columns:
            return f'The cells already have a column "{column}". Choose another name.'
        if label.mode is LabelRule.THRESHOLD:
            threshold = label.threshold
            assert threshold is not None
            names = [name.strip() for name in threshold.classes]
            if not all(names):
                return NO_CLASS_NAME
            twice = next((name for name, count in Counter(names).items() if count > 1), None)
            if twice is not None:
                return f'Two classes are named "{twice}".'
            if self.samples and not threshold.samples:
                return NO_SAMPLE
            if not _good_sds(threshold.sds):
                return BAD_SDS
        else:
            gates = label.gates
            assert gates is not None
            if not gates.gates:
                return NO_GATE
            if not all(_text(gate["class"]).strip() for gate in gates.gates):
                return NO_CLASS_NAME
        return self.problem

    def build_editor(self) -> None:
        label = self.label
        self.swatches = []
        self.editor.clear()
        with self.editor:
            if label.saved_at is not None:
                ui.switch(USE_LABEL, value=label.use, on_change=lambda e: self.switch(bool(e.value)))
            self.shares_line = ui.label().classes("text-sm muted break-words")
            self.problem_line = ui.label().classes("text-sm text-negative break-words")
            ui.input(
                "Name of the label column", value=label.column, on_change=lambda e: self.set_column(e.value)
            ).classes("w-full")
            ui.toggle(
                [ONE, TWO],
                value=ONE if label.mode is LabelRule.THRESHOLD else TWO,
                on_change=lambda e: self.set_mode(e.value),
            )
            if label.mode is LabelRule.THRESHOLD:
                self.threshold_controls()
            else:
                self.gates_controls()
            with ui.row().classes("gap-2 mt-2"):
                ui.button("Save label", icon="save", on_click=self.save)
                ui.button("Delete label", icon="delete", on_click=self.delete).props("outline color=negative")

    def threshold_controls(self) -> None:
        threshold = self.label.threshold
        assert threshold is not None
        _picker(self.menu, "Feature", threshold.feature, self.set_feature)
        if self.samples:
            ui.select(
                self.samples,
                value=list(threshold.samples),
                label="Samples that set the cut-offs",
                multiple=True,
                on_change=lambda e: self.set_samples(e.value),
            ).props("use-chips").classes("w-full")
            ui.label(SAMPLES_LINE).classes("text-xs muted -mt-1")
        ui.switch(
            "Two cut-offs (3 classes)", value=len(threshold.cutoffs) == 2, on_change=lambda e: self.set_two(e.value)
        )
        ui.number(SDS_LABEL, value=threshold.sds, step=SDS_STEP, on_change=lambda e: self.set_sds(e.value)).props(
            f"debounce={TYPING_MS}"
        ).classes("w-full")
        self.suggestion_row = ui.row().classes("gap-1")
        ui.label("Class names").classes(SECTION_TITLE)
        with ui.row().classes("w-full no-wrap gap-1"):
            for index, name in enumerate(threshold.classes):
                ui.input(value=name, on_change=lambda e, index=index: self.set_class(index, e.value)).props(
                    f"debounce={TYPING_MS}"
                ).classes("grow")

    def gates_controls(self) -> None:
        gates = self.label.gates
        assert gates is not None
        _picker(self.menu, "x", gates.x, lambda column: self.set_axis("x", column))
        ui.checkbox("log10", value=gates.log_x, on_change=lambda e: self.set_axis("log_x", bool(e.value)))
        _picker(self.menu, "y", gates.y, lambda column: self.set_axis("y", column))
        ui.checkbox("log10", value=gates.log_y, on_change=lambda e: self.set_axis("log_y", bool(e.value)))
        ui.checkbox("Suggest a threshold on each axis (Otsu)", value=self.suggest, on_change=self.set_suggest)
        if self.suggest:
            with ui.row().classes("items-end no-wrap gap-2 w-full"):
                ui.select(
                    list(SPLITS),
                    value=self.split,
                    label="Split at the thresholds into",
                    on_change=lambda e: setattr(self, "split", e.value),
                ).classes("grow")
                ui.button("Make gates", icon="grid_view", on_click=self.make_gates).props("outline")
        ui.label(GATES_TITLE).classes(f"{SECTION_TITLE} mt-2")
        with ui.column().classes("w-full gap-1"):
            for gate in gates.gates:
                with ui.row().classes("items-center no-wrap gap-2 w-full"):
                    swatch = ui.element("div").classes("w-3 h-3 rounded-sm shrink-0")
                    ui.input(
                        value=gate["class"], on_change=lambda e, gate=gate: self.rename_gate(gate, e.value)
                    ).props(f"debounce={TYPING_MS}").classes("grow")
                    ui.button(icon="delete", on_click=lambda gate=gate: self.delete_gate(gate)).props(
                        "flat round dense size=sm"
                    )
                self.swatches.append((gate, swatch))
        ui.label(GATES_HINT).classes("text-xs muted")
        self.class_list = ui.column().classes("w-full gap-0")

    def color_swatches(self) -> None:
        names = list(dict.fromkeys(_text(gate["class"]).strip() for gate, _ in self.swatches))
        for gate, swatch in self.swatches:
            color = class_color(names.index(_text(gate["class"]).strip()))
            swatch.style(replace=f"background: {color}")

    def show_classes(self) -> None:
        self.problem_line.text = self.problem
        self.problem_line.set_visibility(bool(self.problem))
        classes = self.classes
        self.shares_line.text = f"Shown cells: {_shares(classes)}" if classes is not None else ""
        self.shares_line.set_visibility(classes is not None)
        if self.label.mode is LabelRule.THRESHOLD:
            return
        self.color_swatches()
        self.class_list.clear()
        if classes is None:
            return
        gates = self.label.gates
        assert gates is not None
        found, none = _counts(classes)
        total = len(classes)
        shared = Counter(_text(gate["class"]).strip() for gate in gates.gates)
        with self.class_list:
            for name, count in found.items():
                note = f" ({shared[name]} gates)" if shared[name] > 1 else ""
                ui.label(f"{name}{note}: {count:,} cells ({_percent(count, total)})").classes("text-sm")
            if none:
                ui.label(f"{NO_CLASS}: {none:,} cells ({_percent(none, total)})").classes("text-sm muted")

    async def update(self) -> None:
        self.work_out_classes()
        self.show_classes()
        await self.draw()

    async def draw(self) -> None:
        self.ticket += 1
        ticket = self.ticket
        self.spinner.set_visibility(True)
        try:
            if self.label.mode is LabelRule.THRESHOLD:
                fig, line, message, config = await self.threshold_figure(ticket)
            else:
                fig, line, message, config = await self.gates_figure()
        finally:
            if ticket == self.ticket:
                self.spinner.set_visibility(False)
        if ticket != self.ticket:
            return
        self.figure, self.config = fig, config
        self.message.text = message
        self.message.set_visibility(bool(message))
        self.line.text = line
        self.plot.set_visibility(fig is not None)
        self.show()

    async def threshold_figure(self, ticket: int) -> tuple[go.Figure | None, str, str, dict[str, Any]]:
        threshold = self.label.threshold
        assert threshold is not None
        if self.samples and not threshold.samples:
            self.show_suggestions({})
            return None, "", NO_SAMPLE, HISTOGRAM_CONFIG
        if not _good_sds(threshold.sds):
            self.show_suggestions({})
            return None, "", BAD_SDS, HISTOGRAM_CONFIG
        assert threshold.sds is not None
        suggestions = await self.suggestions_for(threshold)
        if ticket != self.ticket:
            return None, "", "", HISTOGRAM_CONFIG
        self.show_suggestions(suggestions)
        values, not_picked, no_value = self.threshold_values(threshold)
        feature = self.cells.features.get(threshold.feature or "")
        title = feature.title if feature is not None else _text(threshold.feature)
        try:
            fig, beyond = await run.io_bound(
                _histogram, values, list(threshold.cutoffs), threshold.sds, title, self.lines
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            return None, "", str(error), HISTOGRAM_CONFIG
        line = cells_line(
            int(values.size) - beyond,
            len(self.cells.table),
            covered=self.covered,
            filtered=self.filtered,
            not_picked=not_picked,
            no_value=no_value,
            beyond_sds=beyond,
            sds=threshold.sds,
        )
        return fig, line, "", HISTOGRAM_CONFIG

    async def gates_figure(self) -> tuple[go.Figure | None, str, str, dict[str, Any]]:
        gates = self.label.gates
        assert gates is not None
        missing = [column for column in (gates.x, gates.y) if column not in self.table.columns]
        if missing:
            return None, "", f'The cells have no column "{missing[0]}". Pick another feature.', GATES_CONFIG
        self.guide_lines = await self.guides_for(gates) if self.suggest else None
        classes = self.classes
        if classes is None:
            classes = pd.Categorical([None] * len(self.table), categories=[])
        titles = (self.axis_title(gates.x, gates.log_x), self.axis_title(gates.y, gates.log_y))
        fig, line = await run.io_bound(
            gate_figure,
            self.table,
            self.plot_rule(gates),
            classes,
            titles,
            self.guide_lines,
            self.covered,
            self.filtered,
        )
        fig.update_layout(dragmode=self.dragmode)
        return fig, line, "", GATES_CONFIG

    def axis_title(self, column: str, log: bool) -> str:
        feature = self.cells.features.get(column)
        title = feature.title if feature is not None else column
        return LOG + title if log else title

    def show(self) -> None:
        if self.figure is None:
            return
        fig = go.Figure(self.figure)
        restyle(fig, dark=STATE.dark)
        self.dark = STATE.dark
        self.plot.update_figure({**fig.to_plotly_json(), "config": self.config})

    def follow_mode(self) -> None:
        if self.figure is not None and self.dark != STATE.dark:
            self.show()

    def show_suggestions(self, suggestions: Suggestions) -> None:
        if self.label.mode is not LabelRule.THRESHOLD:
            return
        self.suggestion_row.clear()
        with self.suggestion_row:
            for method, cuts in suggestions.items():
                rounded = [_rounded(cut) for cut in cuts]
                text = f"{method}: {', '.join(f'{cut:g}' for cut in rounded)}"
                ui.button(text, on_click=lambda rounded=rounded: self.use_cutoffs(rounded)).props("outline dense")

    def set_column(self, value: Any) -> None:
        self.label.column = _text(value)

    async def set_mode(self, value: Any) -> None:
        mode = LabelRule.THRESHOLD if value == ONE else LabelRule.GATES
        if mode is self.label.mode:
            return
        self.label.mode = mode
        if mode is LabelRule.GATES and self.label.gates is None:
            self.label.gates = self.new_gates()
        if mode is LabelRule.THRESHOLD and self.label.threshold is None:
            self.label.threshold = await self.new_threshold()
            self.follow_threshold()
        self.lines_switch.set_visibility(mode is LabelRule.THRESHOLD)
        self.build_editor()
        await self.update()

    async def set_feature(self, column: str) -> None:
        threshold = self.label.threshold
        if threshold is None or column == threshold.feature or self.asking:
            return
        threshold.feature = column
        suggestions = await self.suggestions_for(threshold)
        if Threshold.OTSU in suggestions:
            threshold.cutoffs = [_rounded(suggestions[Threshold.OTSU][0])]
            threshold.classes = _two_classes(threshold.classes)
        self.build_editor()
        await self.update()

    async def set_samples(self, value: Any) -> None:
        threshold = self.label.threshold
        if threshold is None:
            return
        threshold.samples = sorted(int(sample) for sample in value or [])
        await self.draw()

    async def set_sds(self, value: Any) -> None:
        threshold = self.label.threshold
        if threshold is None:
            return
        sds = _number(value)
        if sds == threshold.sds:
            return
        threshold.sds = sds
        await self.draw()

    async def set_two(self, value: Any) -> None:
        threshold = self.label.threshold
        if threshold is None or bool(value) == (len(threshold.cutoffs) == 2):
            return
        suggestions = await self.suggestions_for(threshold)
        method = Threshold.OTSU_2CUT if value else Threshold.OTSU
        if method in suggestions:
            self.set_cutoffs([_rounded(cut) for cut in suggestions[method]])
        self.build_editor()
        await self.update()

    def set_cutoffs(self, cutoffs: list[float]) -> None:
        threshold = self.label.threshold
        assert threshold is not None
        threshold.cutoffs = sorted(cutoffs)
        threshold.classes = _three_classes(threshold.classes) if len(cutoffs) == 2 else _two_classes(threshold.classes)

    async def use_cutoffs(self, cutoffs: list[float]) -> None:
        threshold = self.label.threshold
        if threshold is None:
            return
        count = len(threshold.cutoffs)
        self.set_cutoffs(cutoffs)
        if len(cutoffs) != count:
            self.build_editor()
        await self.update()

    async def set_class(self, index: int, value: Any) -> None:
        threshold = self.label.threshold
        if threshold is None or index >= len(threshold.classes):
            return
        threshold.classes[index] = _text(value)
        self.work_out_classes()
        self.show_classes()

    async def set_axis(self, name: str, value: Any) -> None:
        gates = self.label.gates
        if gates is None or getattr(gates, name) == value or self.asking:
            return
        if gates.gates:
            count = len(gates.gates)
            question = (
                f"Changing the axes clears the {count} gates. Go on?"
                if count > 1
                else ("Changing the axes clears the gate. Go on?")
            )
            self.asking = True
            try:
                go_on = await _ask(question, "Clear the gates")
            finally:
                self.asking = False
            if not go_on:
                self.build_editor()
                await self.update()
                return
            gates.gates = []
        setattr(gates, name, value)
        self.build_editor()
        await self.update()

    async def set_suggest(self, e: ValueChangeEventArguments[Any]) -> None:
        self.suggest = bool(e.value)
        self.build_editor()
        await self.update()

    def taken_names(self) -> set[str]:
        gates = self.label.gates
        assert gates is not None
        return {_text(gate["class"]).strip() for gate in gates.gates}

    async def make_gates(self) -> None:
        gates = self.label.gates
        if gates is None or self.guide_lines is None:
            return
        x_edges = _edges(plotted(self.cells.table, gates.x, gates.log_x))
        y_edges = _edges(plotted(self.cells.table, gates.y, gates.log_y))
        if x_edges is None or y_edges is None:
            return
        taken = self.taken_names()
        for gate in split_gates(self.split, self.guide_lines, x_edges, y_edges):
            name = gate["class"]
            if name in taken:
                name = next(f"{name} {n}" for n in itertools.count(2) if f"{name} {n}" not in taken)
                gate["class"] = name
            taken.add(name)
            gates.gates.append(gate)
        self.build_editor()
        await self.update()

    async def rename_gate(self, gate: dict[str, Any], value: Any) -> None:
        gate["class"] = _text(value)
        await self.update()

    async def delete_gate(self, gate: dict[str, Any]) -> None:
        gates = self.label.gates
        if gates is None:
            return
        gates.gates = [kept for kept in gates.gates if kept is not gate]
        self.build_editor()
        await self.update()

    async def relayout(self, e: GenericEventArguments) -> None:
        args = e.args if isinstance(e.args, dict) else {}
        if self.label.mode is LabelRule.THRESHOLD:
            await self.dragged(args)
        else:
            await self.drawn(args)

    async def dragged(self, args: dict[str, Any]) -> None:
        threshold = self.label.threshold
        if threshold is None:
            return
        cutoffs = list(threshold.cutoffs)
        moved = False
        for index in range(len(cutoffs)):
            value = _number(args.get(f"shapes[{index}].x0"))
            if value is not None:
                cutoffs[index] = _rounded(value)
                moved = True
        if not moved:
            return
        threshold.cutoffs = sorted(cutoffs)
        await self.update()

    async def drawn(self, args: dict[str, Any]) -> None:
        gates = self.label.gates
        if gates is None:
            return
        mode = args.get("dragmode")
        if isinstance(mode, str) and mode in DRAG_MODES.values():
            self.dragmode = mode
        shapes = args.get("shapes")
        if not isinstance(shapes, list):
            return
        spans = self.spans(gates)
        smallest = TINY_SHARE * spans[0] * spans[1]
        taken = self.taken_names()
        for shape in shapes[len(gates.gates) :]:
            kind = SHAPE_KINDS.get(str(shape.get("type"))) if isinstance(shape, dict) else None
            if kind is None:
                continue
            points = drawn_points(shape)
            if (kind == FREEHAND and len(points) < 3) or area(points, kind) < smallest:
                continue
            name = next(name for n in itertools.count(1) if (name := NEW_GATE.format(number=n)) not in taken)
            taken.add(name)
            gates.gates.append({"class": name, "shape": kind, "points": points})
            self.dragmode = DRAG_MODES[kind]
        self.build_editor()
        await self.update()

    async def save(self) -> None:
        reason = self.refusal()
        if reason:
            ui.notify(reason, type="negative")
            return
        rule = self.saved_rule()
        saved = copy.deepcopy(self.saved)
        index = self.label.saved_at
        if index is None or index >= len(saved):
            saved.append(rule)
            index = len(saved) - 1
        else:
            saved[index] = rule
        try:
            self.exp = save_labels(self.exp, saved)
        except (OSError, ValueError) as error:
            ui.notify(f"Not saved: {_reason(error)}", type="negative")
            return
        new = self.label.saved_at is None
        self.label.column = rule["column"]
        self.label.saved_at = index
        self.opened = self.state()
        self.read_saved()
        self.refresh_tabs()
        ui.notify(f'Saved the label "{rule["column"]}" to experiment.json.', type="positive")
        if new:
            self.build_editor()
            await self.update()

    async def delete(self) -> None:
        if self.asking:
            return
        index = self.label.saved_at
        if index is None or index >= len(self.saved):
            if self.edited() and not await self.ask_discard():
                return
            await self.open(0 if self.saved else None)
            return
        self.asking = True
        try:
            sure = await _ask(f'Delete the label "{_text(self.saved[index].get("column"))}"?', "Delete")
        finally:
            self.asking = False
        if not sure:
            return
        saved = copy.deepcopy(self.saved)
        del saved[index]
        try:
            self.exp = save_labels(self.exp, saved)
        except (OSError, ValueError) as error:
            ui.notify(f"Not saved: {_reason(error)}", type="negative")
            return
        self.read_saved()
        self.label.saved_at = None
        await self.open(0 if self.saved else None)


@ui.page("/labels")
def labels_page() -> None:
    with page_frame("Labels & gates"):
        exp = STATE.experiment
        if exp is None:
            with ui.row().classes("items-center gap-1 text-sm"):
                ui.label("No experiment open.")
                ui.link("Open one on the Experiment page.", "/")
            return
        cells = load_cells(exp)
        if not cells.measured:
            empty_state(cells)
            return
        kept, problem = kept_cells(cells, exp)
        _LabelsPage(exp, cells, marker_menu(cells.features, read_info_file(exp)), kept, problem).build()
