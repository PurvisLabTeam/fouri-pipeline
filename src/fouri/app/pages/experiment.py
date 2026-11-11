"""The Experiment page: the experiment's paths and anchor round, its plate, and its rounds."""

from __future__ import annotations

import json
import keyword
import os
import platform
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePath
from typing import Any

import pandas as pd
from nicegui import events, run, ui

from fouri.app.cells import CONDITION, PLATE_COLUMN, PLATE_ROW, WELL, Cells, load_cells
from fouri.app.derived import FormulaCheck, check_formulas
from fouri.app.dialogs import pick_file, pick_folder
from fouri.app.names import Feature, describe
from fouri.app.plate import overall_state, plate
from fouri.app.shell import page_frame
from fouri.app.state import STATE, open_experiment
from fouri.config import DEFAULT_PATHS, Experiment, save_config
from fouri.io import read_info_file
from fouri.metadata import default_derived_features
from fouri.status import State

SECTION_TITLE = "text-[1.05rem] font-semibold"
STATE_COLORS = {State.COMPLETE: "#0F9E8E", State.PARTIAL: "#D97706", State.NOT_STARTED: "#A8A3BC"}
NOT_IMAGED = "border: 1px dashed var(--c-line)"
LEGEND = [
    ("complete", f"background: {STATE_COLORS[State.COMPLETE]}"),
    ("partial", f"background: {STATE_COLORS[State.PARTIAL]}"),
    ("not started", f"background: {STATE_COLORS[State.NOT_STARTED]}"),
    ("not imaged", NOT_IMAGED),
]
CAN_NOT_OPEN = (OSError, ValueError, AttributeError, TypeError)  # what loading a file that isn't an experiment raises
TYPE_IN_FULL = "Type the full path to experiment.json or its folder."

DERIVED_INTRO = (
    "Features calculated from the measured ones, such as a ratio. Formulas use column names and "
    "+ - * / ( ), and may use any derived feature listed above them."
)
COUNTED = "Computed for {cells:,} of {total:,} cells."
COUNTING = "Counting cells…"
SOME_WELLS = "Counted on {measured} of {wells} wells."
NO_WELLS = "Not checked: no wells measured yet."
CANT_READ_CELLS = "Not checked: can't read the measured cells: {reason}"
NEEDS_BOTH = "Give every row a name and a formula, or delete it."
NAMED_TWICE = "Duplicate row names: change one of the rows named {name}."
MEASURED_NAME = "{name} is already a measured column. Choose another name."
GROUP_NAME = "{name} is already a column. Choose another name."
FIX_RED = "Fix the formulas marked in red, then save."
SAVED_DERIVED = "Saved {count} to experiment.json."
GROUP_NAMES = frozenset({WELL, CONDITION, PLATE_ROW, PLATE_COLUMN})
LINE = "text-xs ml-1"


# -- how the paths are written ---------------------------------------------------------------


def _same(first: str, second: str) -> bool:
    return os.path.normcase(os.path.normpath(first)) == os.path.normcase(os.path.normpath(second))


def _in_full(value: str, base: str) -> str:
    """Return a typed path in full; one without a drive or root is taken as relative to ``base``."""
    path = os.path.expanduser(value.strip())
    pure = PurePath(path)
    return os.path.normpath(path if pure.drive or pure.root else os.path.join(base, path))


def _results_key(config: dict[str, Any]) -> str:
    """Return the key Experiment reads the results folder from on this PC."""
    if platform.system() != "Windows" and (config.get("mac_base_dir") or "").strip():
        return "mac_base_dir"
    return "base_dir"


def _inside(path: str, folder: str) -> bool:
    try:
        return _same(os.path.commonpath([path, folder]), folder)
    except ValueError:  # on different drives
        return False


def _path_entry(value: str, base_dir: str, default: str) -> str | None:
    """Return a path as written under ``paths``: None when base_dir's own, else relative or full."""
    if not value.strip():
        return None
    path = _in_full(value, base_dir)
    if _same(path, os.path.join(base_dir, default)):
        return None
    if _inside(path, base_dir):
        return PurePath(os.path.relpath(path, base_dir)).as_posix()
    return path


def config_to_save(exp: Experiment, *, data: str, info_file: str, results: str, anchor: int | None) -> dict[str, Any]:
    """Return the experiment's settings with the page's paths and anchor round, as Save writes them.

    The results folder goes to the key it is read from on this PC: ``mac_base_dir`` on a PC that
    isn't Windows when that is set, ``base_dir`` otherwise. That key is left out only when the
    folder would be read the same without it: the folder holding experiment.json, with no other
    key to take over, so on such a PC ``mac_base_dir`` stays while ``base_dir`` is set. The ND2
    folder and info_exp.csv go under ``paths``, each left out when its box is empty or it is the
    results folder's own ``data`` or ``info_exp.csv``, relative to the results folder when
    inside it, and in full otherwise. A path typed without a drive or root is relative to the
    results folder. ``anchor`` None keeps the saved anchor round.
    """
    config = dict(exp.config)
    home = os.path.dirname(exp.config_path)
    base_dir = _in_full(results, home) if results.strip() else home
    key = _results_key(config)
    config[key] = base_dir
    without = {name: value for name, value in config.items() if name != key}
    if _same(Experiment(without, exp.config_path).base_dir, base_dir):
        config = without
    paths = dict(config.get("paths") or {})
    for name, value in (("data", data), ("info_file", info_file)):
        entry = _path_entry(value, base_dir, DEFAULT_PATHS[name])
        if entry is None:
            paths.pop(name, None)
        else:
            paths[name] = entry
    if paths:
        config["paths"] = paths
    else:
        config.pop("paths", None)
    if anchor is not None:
        config["anchor_round"] = anchor
    return config


# -- reading what the page shows ---------------------------------------------------------------


def _reason(error: Exception) -> str:
    text = error.strerror if isinstance(error, OSError) and error.strerror else str(error)
    return text.rstrip(".")


def _read_rounds(exp: Experiment) -> tuple[pd.DataFrame | None, str]:
    """Return info_exp.csv's rounds, or None and what to tell the user when it can't be read."""
    try:
        return read_info_file(exp), ""
    except (OSError, ValueError) as error:
        reason = _reason(error)
        return None, f"Can't read info_exp.csv at {exp.info_file}: {reason}. Pick the right file above, then Save."


def _anchor(exp: Experiment, rounds: pd.DataFrame) -> tuple[int | None, str]:
    """Return the anchor round's index, or None and what to tell the user when it isn't a round."""
    try:
        anchor = exp.anchor_round
    except ValueError:
        anchor = None
    if anchor is None or anchor >= len(rounds):
        value = json.dumps(exp.config.get("anchor_round", 0))
        return None, (
            f"The anchor round in experiment.json ({value}) isn't one of the rounds in info_exp.csv. "
            "Pick one above, then Save."
        )
    return anchor, ""


def _plate_states(exp: Experiment) -> tuple[dict[str, State] | None, str]:
    """Return each well's overall state, or None and what to tell the user if one can't be read."""
    try:
        return overall_state(exp), ""
    except Exception as error:  # any well whose files can't be read stops the whole plate
        return None, f"Can't show the plate: {_reason(error)}"


def _hover_text(well: str, metadata: dict[str, Any], state: State) -> str:
    parts = [str(metadata[key]) for key in ("treatment", "treatment_time") if metadata.get(key) not in (None, "")]
    return f"{well}: {' · '.join([*parts, state.value])}"


def _count(number: int, word: str) -> str:
    return f"{number} {word}{'' if number == 1 else 's'}"


# -- the page ------------------------------------------------------------------------------------


def _path_row(label: str, value: str, hint: str) -> tuple[ui.input, ui.button]:
    with ui.row().classes("w-full items-end no-wrap gap-2"):
        box = ui.input(label, value=value).classes("grow").tooltip(hint)
        browse = ui.button(icon="folder_open").props("outline").tooltip("Browse")
    return box, browse


def _show_open_path(box: ui.input) -> None:
    box.value = STATE.experiment.config_path if STATE.experiment is not None else ""


def _open(path: str | Path, box: ui.input) -> None:
    """Open ``path`` as the experiment; else say why and put the open experiment's path back."""
    try:
        open_experiment(path)
    except CAN_NOT_OPEN as error:
        # no experiment.json there: load_config's own message, which names the path
        nothing_there = isinstance(error, FileNotFoundError) and error.strerror is None
        message = str(error) if nothing_there else f"Can't open {path} as an experiment: {_reason(error)}"
        ui.notify(message, type="negative")
        _show_open_path(box)
        return
    _content.refresh()


def _open_typed(box: ui.input) -> None:
    path = os.path.expanduser(box.value.strip())
    pure = PurePath(path)
    if not (pure.drive or pure.root):
        ui.notify(TYPE_IN_FULL, type="negative")
        _show_open_path(box)
        return
    _open(path, box)


async def _open_other(box: ui.input) -> None:
    chosen = await pick_file(box.value, suffix=".json")
    if chosen is not None:
        _open(chosen, box)


async def _browse_folder(box: ui.input) -> None:
    chosen = await pick_folder(box.value)
    if chosen is not None:
        box.value = str(chosen)


async def _browse_csv(box: ui.input) -> None:
    chosen = await pick_file(box.value, suffix=".csv")
    if chosen is not None:
        box.value = str(chosen)


def _message_card(text: str) -> None:
    with ui.card().classes("p-5 max-w-[480px]"), ui.row().classes("items-start no-wrap gap-3"):
        ui.icon("warning").classes("text-xl").style("color: var(--q-warning)")
        ui.label(text).classes("text-sm")


def _plate_card(exp: Experiment, states: dict[str, State]) -> None:
    colors = {well: STATE_COLORS[state] for well, state in states.items()}
    metadata = exp.well_metadata
    hover = {well: _hover_text(well, metadata.get(well, {}), state) for well, state in states.items()}
    with ui.card().classes("p-5"):
        ui.label("Plate").classes(SECTION_TITLE)
        with ui.element("div").classes("mt-2"):
            plate(exp, colors, hover=hover)
        with ui.row().classes("gap-3 text-xs muted mt-3"):
            for name, style in LEGEND:
                with ui.row().classes("items-center gap-1"):
                    ui.element("div").classes("w-3 h-3 rounded").style(style)
                    ui.label(name)


def _rounds_card(exp: Experiment, rounds: pd.DataFrame, anchor: int) -> None:
    channels = [str(channel) for channel in rounds.columns[1:]]
    names = rounds["round_name"].astype(str).tolist()
    with ui.card().classes("w-full max-w-[1000px] p-5 gap-2"):
        ui.label("Rounds").classes(SECTION_TITLE)
        ui.label(
            f"{_count(len(names), 'round')}, {_count(len(channels), 'channel')}, from {Path(exp.info_file).name}. "
            f"The anchor round is {names[anchor]}."
        ).classes("text-sm muted")
        with ui.grid(columns=len(channels) + 1).classes("gap-x-6 gap-y-1 items-center mt-1"):
            ui.label("Round").classes("text-sm font-semibold")
            for channel in channels:
                ui.label(channel).classes("text-sm font-semibold")
            for index, name in enumerate(names):
                ui.label(name + (" (anchor)" if index == anchor else "")).classes("text-sm font-mono")
                for channel in channels:
                    value = rounds[channel].iloc[index]
                    marker = str(value).strip() if pd.notna(value) and str(value).strip() else "—"
                    ui.label(marker).classes("text-sm" + (" muted" if marker in {"—", "DNA"} else " font-medium"))


def _as_typed(column: str) -> str:
    return column if column.isidentifier() and not keyword.iskeyword(column) else f"`{column}`"


def _option(column: str, feature: Feature | None) -> str:
    title = feature.title if feature is not None else column
    return column if title == column else f"{column} ({title})"


def _selection(args: Any) -> tuple[int, int] | None:
    if isinstance(args, list) and len(args) == 2 and all(isinstance(value, int) for value in args):
        return args[0], args[1]
    return None


@dataclass(eq=False)
class _Row:
    name: str
    formula: str
    cursor: tuple[int, int] | None = None
    box: ui.input | None = None


class _Derived:
    def __init__(self) -> None:
        self.config_path = ""
        self.rows: list[_Row] = []
        self.selected: _Row | None = None
        self.info: pd.DataFrame | None = None
        self.cells: Cells | None = None
        self.problem = ""
        self.loading = False
        self.generation = 0
        self.checks: list[FormulaCheck] = []
        self.lines: list[ui.label] = []

    def show(self, exp: Experiment, info: pd.DataFrame | None) -> None:
        if exp.config_path != self.config_path:
            self.config_path = exp.config_path
            self.rows = [_Row(str(spec["name"]), str(spec["formula"])) for spec in exp.derived_features]
            self.selected = None
        self.info = info
        self.cells = None
        self.problem = ""
        self.loading = True
        self.generation += 1
        with ui.card().classes("w-full max-w-[1000px] p-5 gap-2"):
            ui.label("Derived features").classes(SECTION_TITLE)
            ui.label(DERIVED_INTRO).classes("text-sm muted")
            self.wells_line = ui.label().classes("text-sm muted")
            self.body = ui.column().classes("w-full gap-1")
            self.search = ui.select(
                {}, label="Insert a column into the selected formula", with_input=True, on_change=self._insert
            ).classes("w-full")
            with ui.row().classes("gap-2 w-full"):
                ui.button("Add a feature", icon="add", on_click=self._add).props("outline")
                suggest = (
                    ui.button(
                        "Suggest from the channels",
                        icon="auto_awesome",
                        on_click=partial(self._suggest, exp.info_file),
                    )
                    .props("outline")
                    .tooltip("pRB / RB when both are in one round")
                )
                suggest.set_enabled(info is not None)
                ui.space()
                self.save_button = ui.button("Save to experiment.json", icon="save", on_click=self._save)
        self._draw_rows()
        ui.timer(0, partial(self._load, exp, self.generation), once=True)

    async def _load(self, exp: Experiment, generation: int) -> None:
        try:
            cells = await run.io_bound(load_cells, exp)
        except Exception as error:
            if generation != self.generation:
                return
            self.problem = CANT_READ_CELLS.format(reason=_reason(error))
        else:
            if generation != self.generation:
                return
            self.cells = cells
        self.loading = False
        self._update_search()
        self._recheck()

    def _measured(self) -> Cells | None:
        return self.cells if self.cells is not None and self.cells.measured else None

    def _draw_rows(self) -> None:
        self.body.clear()
        self.lines = []
        with self.body:
            for row in self.rows:
                with ui.row().classes("w-full items-center no-wrap gap-2"):
                    ui.input("Name", value=row.name, on_change=partial(self._renamed, row)).classes("w-48")
                    row.box = ui.input("Formula", value=row.formula, on_change=partial(self._typed, row)).classes(
                        "grow font-mono"
                    )
                    row.box.on("focus", partial(self._select, row), js_handler="() => emit()")
                    row.box.on(
                        "blur",
                        partial(self._left, row),
                        js_handler="(e) => emit(e.target.selectionStart, e.target.selectionEnd)",
                    )
                    ui.button(icon="delete", on_click=partial(self._delete, row)).props("flat dense round").tooltip(
                        "Delete"
                    )
                self.lines.append(ui.label().classes(LINE))
        self._recheck()

    def _line(self, row: _Row, check: FormulaCheck | None, total: int) -> tuple[str, str]:
        if not row.formula.strip():
            return "", "muted"
        if self.loading:
            return COUNTING, "text-warning"
        if check is None:
            return "", "muted"
        if check.cells is not None:
            return "✓ " + COUNTED.format(cells=check.cells, total=total), "text-positive"
        return "✗ " + check.problem, "text-negative" if check.stops_build else "text-warning"

    def _wells_text(self) -> str:
        if self.loading:
            return ""
        if self.problem:
            return self.problem
        cells = self._measured()
        if cells is None:
            return NO_WELLS
        if cells.missing:
            return SOME_WELLS.format(measured=len(cells.measured), wells=len(cells.measured) + len(cells.missing))
        return ""

    def _recheck(self) -> None:
        cells = self._measured()
        specs = [{"name": row.name, "formula": row.formula} for row in self.rows]
        self.checks = check_formulas(cells.base, specs) if cells is not None and not self.loading else []
        total = len(cells.base) if cells is not None else 0
        for index, (row, line) in enumerate(zip(self.rows, self.lines, strict=True)):
            check = self.checks[index] if self.checks else None
            text, color = self._line(row, check, total)
            line.set_text(text)
            line.classes(replace=f"{LINE} {color}")
            line.set_visibility(bool(text))
        wells = self._wells_text()
        self.wells_line.set_text(wells)
        self.wells_line.set_visibility(bool(wells))
        self.save_button.set_enabled(not self.loading)

    def _update_search(self) -> None:
        cells = self._measured()
        if cells is None:
            self.search.set_options([])
            return
        columns = cells.base.select_dtypes("number").columns.tolist()
        names = describe(columns, self.info) if self.info is not None else {}
        options = {column: _option(column, names.get(column)) for column in columns}
        for row in self.rows:
            if row.name.strip() and row.name not in options:
                options[row.name] = row.name
        self.search.set_options(options)

    def _typed(self, row: _Row, event: events.ValueChangeEventArguments) -> None:
        row.formula = event.value or ""
        self._recheck()

    def _renamed(self, row: _Row, event: events.ValueChangeEventArguments) -> None:
        row.name = event.value or ""
        self._recheck()
        self._update_search()

    def _select(self, row: _Row, _: events.GenericEventArguments) -> None:
        self.selected = row

    def _left(self, row: _Row, event: events.GenericEventArguments) -> None:
        row.cursor = _selection(event.args)

    def _insert(self, event: events.ValueChangeEventArguments) -> None:
        column = event.value
        if not column:
            return
        self.search.set_value(None)
        if not self.rows:
            return
        row = self.selected if self.selected in self.rows else self.rows[-1]
        text = _as_typed(str(column))
        end = len(row.formula)
        start, stop = row.cursor if row.cursor is not None else (end, end)
        start, stop = max(0, min(start, end)), max(0, min(stop, end))
        row.formula = row.formula[:start] + text + row.formula[stop:]
        position = start + len(text)
        row.cursor = (position, position)
        if row.box is not None:
            row.box.set_value(row.formula)
            ui.run_javascript(
                "setTimeout(() => {"
                f" const box = getHtmlElement({row.box.id}).querySelector('input');"
                f" box.focus(); box.setSelectionRange({position}, {position});"
                " }, 50);"
            )
        self._recheck()

    def _add(self) -> None:
        self.rows.append(_Row("", ""))
        self.selected = None
        self._draw_rows()
        self._update_search()

    def _delete(self, row: _Row) -> None:
        self.rows = [other for other in self.rows if other is not row]
        if self.selected is row:
            self.selected = None
        self._draw_rows()
        self._update_search()

    def _suggest(self, info_file: str) -> None:
        names = {row.name for row in self.rows}
        for spec in default_derived_features(info_file):
            if spec["name"] not in names:
                self.rows.append(_Row(spec["name"], spec["formula"]))
                names.add(spec["name"])
        self._draw_rows()
        self._update_search()

    def _refusal(self) -> str:
        if any(not row.name.strip() or not row.formula.strip() for row in self.rows):
            return NEEDS_BOTH
        seen: set[str] = set()
        for row in self.rows:
            if row.name in seen:
                return NAMED_TWICE.format(name=row.name)
            seen.add(row.name)
        cells = self._measured()
        for row in self.rows:
            if cells is not None and row.name in cells.base.columns:
                return MEASURED_NAME.format(name=row.name)
            if row.name in GROUP_NAMES:
                return GROUP_NAME.format(name=row.name)
        if any(check.stops_build for check in self.checks):
            return FIX_RED
        return ""

    def _save(self) -> None:
        refusal = self._refusal()
        if refusal:
            ui.notify(refusal, type="negative")
            return
        try:
            with open(self.config_path, encoding="utf-8") as file:
                config = json.load(file)
            config["derived_features"] = [{"name": row.name, "formula": row.formula} for row in self.rows]
            save_config(config, self.config_path)
        except (OSError, ValueError) as error:
            ui.notify(f"Not saved: {_reason(error)}", type="negative")
            return
        open_experiment(self.config_path)
        ui.notify(SAVED_DERIVED.format(count=_count(len(self.rows), "derived feature")), type="positive")


@ui.refreshable
def _content(derived: _Derived) -> None:
    exp = STATE.experiment
    with ui.row().classes("w-full gap-6 items-start"):
        with ui.card().classes("w-[560px] p-5 gap-3"):
            ui.label("Experiment").classes(SECTION_TITLE)
            if exp is not None and exp.name:
                ui.label(exp.name).classes("muted -mt-2")
            config_box, browse = _path_row(
                "experiment.json", exp.config_path if exp else "", "Wells, conditions and settings"
            )
            config_box.on("keydown.enter", lambda: _open_typed(config_box))
            browse.on_click(lambda: _open_other(config_box))
            if exp is None:
                return

            data_box, browse = _path_row(
                "Raw images (ND2 folder)", exp.data_dir, "One folder per round, as saved by the microscope"
            )
            browse.on_click(lambda: _browse_folder(data_box))
            info_box, browse = _path_row(
                "Round list (info_exp.csv)", exp.info_file, "The antibody in each channel of each round"
            )
            browse.on_click(lambda: _browse_csv(info_box))
            results_box, browse = _path_row("Results folder", exp.base_dir, "Output folder for every step")
            browse.on_click(lambda: _browse_folder(results_box))

            rounds, problem = _read_rounds(exp)
            anchor: int | None = None
            states: dict[str, State] | None = None
            if rounds is not None:
                anchor, problem = _anchor(exp, rounds)
            if not problem:
                states, problem = _plate_states(exp)
            with ui.row().classes("items-center gap-2 w-full"):
                anchor_box = None
                if rounds is not None:
                    names = rounds["round_name"].astype(str).tolist()
                    options = {index: name + (" (default)" if index == 0 else "") for index, name in enumerate(names)}
                    anchor_box = (
                        ui.select(options, value=anchor, label="Anchor round")
                        .classes("w-40")
                        .tooltip("The round all other rounds are aligned to")
                    )
                ui.space()

                def save() -> None:
                    current = STATE.experiment or exp
                    config = config_to_save(
                        current,
                        data=data_box.value,
                        info_file=info_box.value,
                        results=results_box.value,
                        anchor=anchor_box.value if anchor_box is not None else None,
                    )
                    try:
                        save_config(config, current.config_path)
                    except (OSError, ValueError) as error:
                        ui.notify(f"Not saved: {_reason(error)}", type="negative")
                        return
                    open_experiment(current.config_path)
                    ui.notify("Saved. The previous version is in experiment.json.bak.", type="positive")
                    _content.refresh()

                ui.button("Save to experiment.json", icon="save", on_click=save)

        if states is None:
            _message_card(problem)
        else:
            _plate_card(exp, states)
    if rounds is not None and anchor is not None and states is not None:
        _rounds_card(exp, rounds, anchor)
    derived.show(exp, rounds)


@ui.page("/")
def experiment_page() -> None:
    with page_frame("Experiment"):
        _content(_Derived())
