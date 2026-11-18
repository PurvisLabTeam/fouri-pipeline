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
from fouri.app.dialogs import ask, pick_file, pick_folder
from fouri.app.names import Feature, describe
from fouri.app.plate import DASHED, OUTLINED, overall_state, plate
from fouri.app.shell import page_frame
from fouri.app.state import STATE, open_experiment
from fouri.app.wells import (
    DETAILS,
    SAMPLE_ID,
    blank,
    fill_sample_ids,
    in_full,
    new_config,
    path_entry,
    same_path,
    save_details,
    save_wells,
    wells_table,
)
from fouri.config import CONFIG_NAME, DEFAULT_PATHS, Experiment, save_config
from fouri.io import read_info_file
from fouri.metadata import check_metadata, default_derived_features, plate_sort_key, read_metadata_table, wells_in_data
from fouri.plots import GROUP_COLORS
from fouri.status import State

SECTION_TITLE = "text-[1.05rem] font-semibold"
STATE_COLORS = {State.COMPLETE: "#0F9E8E", State.PARTIAL: "#D97706", State.NOT_STARTED: "#A8A3BC"}
LEGEND = [
    ("complete", f"background: {STATE_COLORS[State.COMPLETE]}"),
    ("partial", f"background: {STATE_COLORS[State.PARTIAL]}"),
    ("not started", f"background: {STATE_COLORS[State.NOT_STARTED]}"),
    ("not imaged", DASHED),
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

NEED_RESULTS = "Pick the results folder."
RESULTS_IN_FULL = r"Give the results folder in full, for example Z:\Experiments\2026-11-11."
CREATED = "Created experiment.json with {wells}. Fill in their conditions below."
CREATED_FOLDER = "Created {folder} and experiment.json in it with {wells}. Fill in their conditions below."

DETAIL_LABELS = {
    "experiment_key": "Experiment key",
    "experiment_name": "Name",
    "cell_type": "Cell type",
    "passage": "Passage",
    "date_completed": "Date completed",
    "notes": "Notes",
}
SAVED_DETAILS = "Saved the experiment details."

NO_ID_COLOR = "#A8A3BC"
ROW_KEY = "_row"
INCLUDE = "include"
SELECT = "select"
MODES = {INCLUDE: "Include or leave out wells", SELECT: "Select wells to fill in"}
NO_FILES = "{well} has no ND2 files, so it cannot be added."
NO_ND2_FOUND = "No ND2 files found in {folder}, so wells can be removed but not added."
NAME_IT = "Name the column."
THERE_ALREADY = 'There is already a column "{name}".'
PAGE_NAME = '"{name}" is a name the cell pages use.'
MEASURED_COLUMN = 'The cells already have a column "{name}".'
CELL_PAGE_NAMES = frozenset({"label", "included", "treatment time", CONDITION, PLATE_ROW, PLATE_COLUMN})
WHOLE_ID = "A sample ID is a whole number above 0."
IMPORT_ID = "Can't import: a sample ID is a whole number above 0, and {wells} {verb} others."
FIX_PROBLEMS = "Fix the problems in red, then save."
SAVED_WELLS = "Saved {wells} to experiment.json."
METADATA_CSV = "well_metadata.csv"
REPLACE_CSV = "Replace well_metadata.csv in the results folder?"
WROTE_CSV = "Wrote {path}. Fill it in, for example in Excel, then Import CSV."
IMPORTED = "Imported {wells} from {file}."


# -- how the paths are written ---------------------------------------------------------------


def _results_key(config: dict[str, Any]) -> str:
    """Return the key Experiment reads the results folder from on this PC."""
    if platform.system() != "Windows" and (config.get("mac_base_dir") or "").strip():
        return "mac_base_dir"
    return "base_dir"


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
    base_dir = in_full(results, home) if results.strip() else home
    key = _results_key(config)
    config[key] = base_dir
    without = {name: value for name, value in config.items() if name != key}
    if same_path(Experiment(without, exp.config_path).base_dir, base_dir):
        config = without
    paths = dict(config.get("paths") or {})
    for name, value in (("data", data), ("info_file", info_file)):
        entry = path_entry(value, base_dir, DEFAULT_PATHS[name])
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


async def _browse_folder(box: ui.input, start: str = "") -> None:
    chosen = await pick_folder(box.value or start)
    if chosen is not None:
        box.value = str(chosen)


async def _browse_csv(box: ui.input, start: str = "") -> None:
    chosen = await pick_file(box.value or start, suffix=".csv")
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


def _name_of(experiment: Experiment | None) -> str:
    return experiment.name if experiment is not None else ""


def _has_name(experiment: Experiment | None) -> bool:
    return bool(_name_of(experiment))


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _whole(value: object) -> int | None:
    text = str(value).strip()
    return int(text) if text.isascii() and text.isdigit() and int(text) > 0 else None


def _id_key(value: object) -> tuple[int, int, str]:
    return (0, value, "") if isinstance(value, int) else (1, 0, str(value))


@ui.refreshable
def _status_plate(problem: str, states: dict[str, State] | None) -> None:
    exp = STATE.experiment
    if exp is None:
        return
    if not problem and states is None:
        states, problem = _plate_states(exp)
    if states is None:
        _message_card(problem)
    else:
        _plate_card(exp, states)


async def _new_experiment() -> None:
    with ui.dialog() as dialog, ui.card().classes("w-[560px] p-5 gap-3"):
        ui.label("New experiment").classes(SECTION_TITLE)
        results_box, browse = _path_row(
            "Results folder (experiment.json goes here)", "", "Output folder for every step"
        )
        browse.on_click(lambda: _browse_folder(results_box))
        data_box, browse = _path_row("ND2 folder", "", "Left empty: the data folder in the results folder")
        browse.on_click(lambda: _browse_folder(data_box, results_box.value))
        info_box, browse = _path_row("info_exp.csv", "", "Left empty: info_exp.csv in the results folder")
        browse.on_click(lambda: _browse_csv(info_box, results_box.value))
        problem = ui.label().classes("text-sm text-negative")
        problem.set_visibility(False)

        def refuse(text: str) -> None:
            problem.set_text(text)
            problem.set_visibility(True)

        def create() -> None:
            typed = results_box.value.strip()
            if not typed:
                refuse(NEED_RESULTS)
                return
            path = os.path.expanduser(typed)
            pure = PurePath(path)
            if not (pure.drive or pure.root):
                refuse(RESULTS_IN_FULL)
                return
            folder = Path(os.path.normpath(path))
            try:
                config = new_config(folder, data=data_box.value, info_file=info_box.value)
            except ValueError as error:
                refuse(str(error))
                return
            except OSError as error:
                refuse(f"Not created: {_reason(error)}")
                return
            made = not folder.exists()
            try:
                folder.mkdir(parents=True, exist_ok=True)
                save_config(config, folder / CONFIG_NAME)
                open_experiment(folder / CONFIG_NAME)
            except (OSError, ValueError) as error:
                refuse(f"Not created: {_reason(error)}")
                return
            wells = _count(len(config["well_list"]), "well")
            dialog.submit(CREATED_FOLDER.format(folder=folder, wells=wells) if made else CREATED.format(wells=wells))

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat")
            ui.button("Create", icon="add", on_click=create)
    message = await dialog
    dialog.delete()
    if isinstance(message, str):
        ui.notify(message, type="positive")
        _content.refresh()


class _Details:
    def __init__(self) -> None:
        self.config_path = ""
        self.values: dict[str, str] = {}

    def show(self, exp: Experiment) -> None:
        if exp.config_path != self.config_path:
            self.config_path = exp.config_path
            self.values = {key: _text(exp.config.get(key)) for key in DETAILS}
        with ui.card().classes("w-full max-w-[1000px] p-5 gap-2"):
            ui.label("Experiment details").classes(SECTION_TITLE)
            with ui.grid(columns=2).classes("w-full gap-x-6 gap-y-1"):
                for key in DETAILS:
                    if key != "notes":
                        ui.input(DETAIL_LABELS[key], value=self.values[key], on_change=partial(self._typed, key))
            ui.textarea(
                DETAIL_LABELS["notes"], value=self.values["notes"], on_change=partial(self._typed, "notes")
            ).classes("w-full").props("autogrow")
            with ui.row().classes("w-full"):
                ui.space()
                ui.button("Save details", icon="save", on_click=self._save)

    def _typed(self, key: str, event: events.ValueChangeEventArguments) -> None:
        self.values[key] = event.value or ""

    def _save(self) -> None:
        exp = STATE.experiment
        if exp is None or exp.config_path != self.config_path:
            return
        try:
            save_details(exp, self.values)
        except (OSError, ValueError) as error:
            ui.notify(f"Not saved: {_reason(error)}", type="negative")
            return
        ui.notify(SAVED_DETAILS, type="positive")


class _Wells:
    def __init__(self) -> None:
        self.config_path = ""
        self.exp: Experiment | None = None
        self.rows: list[dict[str, Any]] = []
        self.columns: list[str] = []
        self.conditions: set[str] = set()
        self.kept: dict[str, dict[str, Any]] = {}
        self.selected: set[str] = set()
        self.mode = INCLUDE
        self.boxes: dict[str, str] = {}
        self.data_wells: list[str] = []
        self.apply_button: ui.button | None = None
        self.column_box: ui.input | None = None

    def load(self, exp: Experiment) -> None:
        table = wells_table(exp)
        columns = [str(column) for column in table.columns if column not in (WELL, SAMPLE_ID)]
        for column in exp.condition_columns:
            if column not in columns and column not in (WELL, SAMPLE_ID):
                columns.append(column)
                table[column] = ""
        self.config_path = exp.config_path
        self.columns = columns
        self.rows = [{str(key): value for key, value in row.items()} for row in table.to_dict("records")]
        self.conditions = set(exp.condition_columns)
        self.kept = {}
        self.selected = set()

    def show(self, exp: Experiment) -> None:
        if exp.config_path != self.config_path:
            self.load(exp)
        self.exp = exp
        try:
            self.data_wells = wells_in_data(exp.data_dir)
        except OSError:
            self.data_wells = []
        with ui.card().classes("w-full max-w-[1000px] p-5 gap-3"):
            with ui.row().classes("items-center w-full"):
                ui.label("Wells and metadata").classes(SECTION_TITLE)
                ui.space()
                ui.button("Import CSV", icon="upload_file", on_click=self._import).props("outline")
                ui.button("Export CSV", icon="download", on_click=self._export).props("outline")
            if not self.data_wells:
                ui.label(NO_ND2_FOUND.format(folder=exp.data_dir)).classes("text-sm text-warning")
            with ui.row().classes("items-center w-full gap-2"):
                ui.toggle(MODES, value=self.mode, on_change=self._mode_changed).props("no-caps")
                ui.space()
                self.select_buttons = ui.row().classes("gap-2")
                with self.select_buttons:
                    ui.button("All included", on_click=self._select_all).props("outline dense")
                    ui.button("Clear selection", on_click=self._clear_selection).props("outline dense")
                self.select_buttons.set_visibility(self.mode == SELECT)
            with ui.row().classes("w-full no-wrap gap-8 items-start"):
                self.plate_area = ui.column().classes("gap-3")
                self.boxes_area = ui.column().classes("gap-1 grow")
            self.table_area = ui.column().classes("w-full")
            self.findings_area = ui.column().classes("w-full gap-1")
            with ui.row().classes("w-full"):
                ui.space()
                ui.button("Save wells and metadata", icon="save", on_click=self._save)
        self._draw_boxes()
        self._draw_plate()
        self._draw_table()
        self._draw_findings()

    def _sort(self) -> None:
        self.rows.sort(key=lambda row: plate_sort_key(row[WELL]))

    def _ids(self) -> dict[str, object]:
        ids: dict[str, object] = {}
        for row in self.rows:
            ids.setdefault(row[WELL], row.get(SAMPLE_ID, ""))
        return ids

    def _hover(self, well: str) -> str:
        row = next(row for row in self.rows if row[WELL] == well)
        parts = [str(row[column]).strip() for column in self.columns if not blank(row.get(column))]
        if not blank(row.get(SAMPLE_ID)):
            parts.append(f"sample ID {row[SAMPLE_ID]}")
        return f"{well}: {' · '.join(parts)}" if parts else well

    def _draw_plate(self) -> None:
        exp = self.exp
        if exp is None:
            return
        ids = self._ids()
        numbers = sorted({value for value in ids.values() if not blank(value)}, key=_id_key)
        palette = {value: GROUP_COLORS[index % len(GROUP_COLORS)] for index, value in enumerate(numbers)}
        colors = {well: NO_ID_COLOR if blank(value) else palette[value] for well, value in ids.items()}
        outlined = [well for well in self.data_wells if well not in ids]
        hover = {well: self._hover(well) for well in ids}
        self.plate_area.clear()
        with self.plate_area:
            plate(
                exp,
                colors,
                hover=hover,
                on_click=self._click,
                outlined=outlined,
                selected=self.selected if self.mode == SELECT else None,
            )
            with ui.row().classes("gap-3 text-xs muted"):
                for name, style in (
                    ("included, colored by sample ID", f"background: {GROUP_COLORS[0]}"),
                    ("not included", OUTLINED),
                    ("no ND2 files", DASHED),
                ):
                    with ui.row().classes("items-center gap-1"):
                        ui.element("div").classes("w-3 h-3 rounded").style(style)
                        ui.label(name)
        self._update_apply()

    def _draw_boxes(self) -> None:
        self.boxes_area.clear()
        with self.boxes_area:
            for column in self.columns:
                with ui.row().classes("items-center no-wrap gap-3 w-full"):
                    ui.input(
                        column, value=self.boxes.get(column, ""), on_change=partial(self._box_typed, column)
                    ).classes("grow")
                    ui.checkbox("Condition", value=column in self.conditions, on_change=partial(self._ticked, column))
            self.apply_button = ui.button(icon="done_all", on_click=self._apply).props("outline").classes("mt-1")
            with ui.row().classes("items-end no-wrap gap-2 w-full mt-3"):
                self.column_box = ui.input("New column").classes("grow")
                self.column_box.on("keydown.enter", self._add_column)
                ui.button("Add a column", icon="add", on_click=self._add_column).props("outline")
        self._update_apply()

    def _update_apply(self) -> None:
        if self.apply_button is None:
            return
        count = len(self.selected)
        self.apply_button.set_text(f"Apply to {count} selected well{'' if count == 1 else 's'}")
        self.apply_button.set_enabled(count > 0)

    def _draw_table(self) -> None:
        columns = [WELL, SAMPLE_ID, *self.columns]
        definitions: list[dict[str, Any]] = [{"field": WELL, "headerName": WELL, "editable": False, "pinned": "left"}]
        definitions += [{"field": column, "headerName": column, "editable": True} for column in columns[1:]]
        data = [
            {**{column: "" if blank(row.get(column)) else row[column] for column in columns}, ROW_KEY: index}
            for index, row in enumerate(self.rows)
        ]
        options: dict[str, Any] = {
            "columnDefs": definitions,
            "rowData": data,
            "defaultColDef": {"cellDataType": False, "sortable": False, "resizable": True},
            "singleClickEdit": True,
            "stopEditingWhenCellsLoseFocus": True,
        }
        if len(data) <= 10:
            options["domLayout"] = "autoHeight"
        self.table_area.clear()
        with self.table_area:
            grid = ui.aggrid(options).classes("w-full")
            if len(data) > 10:
                grid.style("height: 360px")
            grid.on("cellValueChanged", self._edited)

    def _condition_list(self) -> list[str]:
        return [column for column in self.columns if column in self.conditions]

    def _table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=[WELL, SAMPLE_ID, *self.columns])

    def _issues(self) -> list[tuple[str, str]]:
        conditions = self._condition_list()
        filled = fill_sample_ids(self._table(), conditions)
        return list(check_metadata(filled, conditions, self.data_wells or None))

    def _draw_findings(self) -> None:
        self.findings_area.clear()
        with self.findings_area:
            for level, message in self._issues():
                color = "text-negative" if level == "error" else "text-warning"
                ui.label(message).classes(f"text-sm {color}")

    def _changed(self) -> None:
        self._draw_plate()
        self._draw_table()
        self._draw_findings()

    def _click(self, well: str) -> None:
        included = any(row[WELL] == well for row in self.rows)
        if self.mode == SELECT:
            if included:
                self.selected ^= {well}
                self._draw_plate()
            return
        if included:
            first = next(row for row in self.rows if row[WELL] == well)
            self.kept[well] = dict(first)
            self.rows = [row for row in self.rows if row[WELL] != well]
            self.selected.discard(well)
        elif well in self.data_wells:
            row = self.kept.pop(well, None) or {WELL: well, SAMPLE_ID: ""}
            for column in self.columns:
                row.setdefault(column, "")
            self.rows.append(row)
            self._sort()
        else:
            ui.notify(NO_FILES.format(well=well), type="warning")
            return
        self._changed()

    def _mode_changed(self, event: events.ValueChangeEventArguments) -> None:
        self.mode = event.value
        self.select_buttons.set_visibility(self.mode == SELECT)
        self._draw_plate()

    def _select_all(self) -> None:
        self.selected = {row[WELL] for row in self.rows}
        self._draw_plate()

    def _clear_selection(self) -> None:
        self.selected = set()
        self._draw_plate()

    def _box_typed(self, column: str, event: events.ValueChangeEventArguments) -> None:
        self.boxes[column] = event.value or ""

    def _ticked(self, column: str, event: events.ValueChangeEventArguments) -> None:
        if event.value:
            self.conditions.add(column)
        else:
            self.conditions.discard(column)
        self._draw_findings()

    def _apply(self) -> None:
        typed = {column: self.boxes.get(column, "").strip() for column in self.columns}
        filled = {column: value for column, value in typed.items() if value}
        for row in self.rows:
            if row[WELL] in self.selected:
                row.update(filled)
        self.selected = set()
        self._changed()

    def _edited(self, event: events.GenericEventArguments) -> None:
        args = event.args if isinstance(event.args, dict) else {}
        data = args.get("data") or {}
        column = args.get("colId")
        index = data.get(ROW_KEY)
        if not isinstance(index, int) or not 0 <= index < len(self.rows) or column not in (SAMPLE_ID, *self.columns):
            return
        row = self.rows[index]
        value = args.get("newValue")
        if column == SAMPLE_ID:
            if blank(value):
                row[SAMPLE_ID] = ""
            else:
                number = _whole(value)
                if number is None:
                    ui.notify(WHOLE_ID, type="negative")
                    self._draw_table()
                    return
                row[SAMPLE_ID] = number
        else:
            row[column] = "" if value is None else str(value).strip()
        self._draw_plate()
        self._draw_findings()

    async def _measured_columns(self) -> set[str]:
        exp = self.exp
        if exp is None:
            return set()
        try:
            cells = await run.io_bound(load_cells, exp)
        except Exception:
            return set()
        return {str(column) for column in cells.base.columns} if cells.measured else set()

    def _column_problem(self, name: str, measured: set[str]) -> tuple[str, str] | None:
        if name in self.columns or name in (WELL, SAMPLE_ID):
            return THERE_ALREADY.format(name=name), ""
        derived = {str(spec.get("name", "")) for spec in self.exp.derived_features} if self.exp else set()
        if name in CELL_PAGE_NAMES or name in derived:
            return PAGE_NAME.format(name=name), "Choose another."
        if name in measured:
            return MEASURED_COLUMN.format(name=name), "Choose another name."
        return None

    def _add_row_column(self, column: str) -> None:
        self.columns.append(column)
        for row in [*self.rows, *self.kept.values()]:
            row.setdefault(column, "")

    async def _add_column(self) -> None:
        if self.column_box is None:
            return
        name = (self.column_box.value or "").strip()
        if not name:
            ui.notify(NAME_IT, type="negative")
            return
        problem = self._column_problem(name, await self._measured_columns())
        if problem is not None:
            ui.notify(" ".join(part for part in problem if part), type="negative")
            return
        self._add_row_column(name)
        self._draw_boxes()
        self._changed()

    async def _import(self) -> None:
        exp = self.exp
        if exp is None:
            return
        chosen = await pick_file(exp.base_dir, suffix=".csv")
        if chosen is None:
            return
        try:
            table = read_metadata_table(chosen)
        except (OSError, ValueError) as error:
            ui.notify(f"Can't import: {_reason(error)}", type="negative")
            return
        given = [str(column) for column in table.columns]
        new_columns = [column for column in given if column not in (WELL, SAMPLE_ID) and column not in self.columns]
        measured = await self._measured_columns() if new_columns else set()
        for column in new_columns:
            problem = self._column_problem(column, measured)
            if problem is not None:
                ui.notify(f"Can't import: {problem[0]} Rename the column in the CSV.", type="negative")
                return
        if SAMPLE_ID in given:
            bad = sorted(
                {
                    well
                    for well, value in zip(table[WELL], table[SAMPLE_ID], strict=True)
                    if not blank(value) and _whole(value) is None
                },
                key=plate_sort_key,
            )
            if bad:
                verb = "has" if len(bad) == 1 else "have"
                ui.notify(IMPORT_ID.format(wells=_and(bad), verb=verb), type="negative")
                return
        for column in new_columns:
            self._add_row_column(column)
        listed = list(dict.fromkeys(str(well) for well in table[WELL]))
        replaced: list[dict[str, Any]] = []
        for well in listed:
            existing = next((dict(row) for row in self.rows if row[WELL] == well), None)
            stashed = self.kept.pop(well, None)
            start = existing or stashed or {WELL: well, SAMPLE_ID: ""}
            for column in self.columns:
                start.setdefault(column, "")
            for _, line in table[table[WELL] == well].iterrows():
                row = dict(start)
                for column in given:
                    if column == SAMPLE_ID:
                        if not blank(line[column]):
                            row[SAMPLE_ID] = _whole(line[column])
                    elif column != WELL:
                        row[column] = line[column]
                replaced.append(row)
        listed_set = set(listed)
        self.rows = [row for row in self.rows if row[WELL] not in listed_set] + replaced
        self._sort()
        ui.notify(IMPORTED.format(wells=_count(len(listed), "well"), file=chosen), type="positive")
        self._draw_boxes()
        self._changed()

    async def _export(self) -> None:
        exp = self.exp
        if exp is None:
            return
        path = Path(exp.base_dir) / METADATA_CSV
        if path.exists() and not await ask(REPLACE_CSV, "Replace"):
            return
        try:
            self._table().to_csv(path, index=False)
        except OSError as error:
            ui.notify(f"Not written: {_reason(error)}", type="negative")
            return
        ui.notify(WROTE_CSV.format(path=path), type="positive")

    def _save(self) -> None:
        exp = self.exp
        if exp is None:
            return
        if any(level == "error" for level, _ in self._issues()):
            ui.notify(FIX_PROBLEMS, type="negative")
            return
        try:
            saved = save_wells(exp, self._table(), self._condition_list())
        except (OSError, ValueError) as error:
            ui.notify(f"Not saved: {_reason(error)}", type="negative")
            return
        ui.notify(SAVED_WELLS.format(wells=_count(len(saved.well_list), "well")), type="positive")
        self.load(saved)
        self.exp = saved
        self._draw_boxes()
        self._changed()
        _status_plate.refresh(states=None)


@ui.refreshable
def _content(derived: _Derived, details: _Details, wells: _Wells) -> None:
    exp = STATE.experiment
    with ui.row().classes("w-full gap-6 items-start"):
        with ui.card().classes("w-[560px] p-5 gap-3"):
            ui.label("Experiment").classes(SECTION_TITLE)
            ui.label().classes("muted -mt-2").bind_text_from(
                STATE, "experiment", backward=_name_of
            ).bind_visibility_from(STATE, "experiment", backward=_has_name)
            with ui.row().classes("w-full items-end no-wrap gap-2"):
                config_box = (
                    ui.input("experiment.json", value=exp.config_path if exp else "")
                    .classes("grow")
                    .tooltip("Wells, conditions and settings")
                )
                browse = ui.button(icon="folder_open").props("outline").tooltip("Browse")
                ui.button("New experiment", icon="add", on_click=_new_experiment).props("outline")
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

        _status_plate(problem=problem, states=states)
    if rounds is not None and anchor is not None and states is not None:
        _rounds_card(exp, rounds, anchor)
    details.show(exp)
    wells.show(exp)
    derived.show(exp, rounds)


@ui.page("/")
def experiment_page() -> None:
    with page_frame("Experiment"):
        _content(_Derived(), _Details(), _Wells())
