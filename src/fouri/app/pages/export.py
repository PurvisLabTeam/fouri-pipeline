from __future__ import annotations

from typing import Any

import anndata as ad
import h5py
import pandas as pd
from nicegui import run, ui

from fouri.app.dialogs import ask
from fouri.app.embeddings import embedding_names
from fouri.app.exports import (
    BLANK_NAME,
    Counts,
    ExportFile,
    export_counts,
    export_settings,
    has_corrected,
    list_exports,
    name_problem,
    table_filters,
)
from fouri.app.names import describe, menu_order
from fouri.app.records import clock_text
from fouri.app.rules import switched_on_filters
from fouri.app.runner import Run, StepRequest
from fouri.app.shell import page_frame
from fouri.app.state import POLL_SECONDS, STATE, run_going
from fouri.app.steps import default
from fouri.config import Experiment
from fouri.export import FileFormat, Values
from fouri.io import adata_path, adata_sub_path, export_path, fullest_df_path, read_info_file
from fouri.parallel import error_in_words
from fouri.status import Step

SECTION_TITLE = "text-[1.05rem] font-semibold"
TITLE_LINE = "Write the included cells to a file to take elsewhere: AnnData for Python, CSV for anything else."
FORMATS = {FileFormat.ANNDATA.value: "AnnData (.h5ad)", FileFormat.CSV.value: "CSV"}
KINDS = {FileFormat.ANNDATA.value: "AnnData", FileFormat.CSV.value: "CSV"}
VALUES = {Values.MEASURED.value: "Measurements", Values.ZSCORES.value: "Z-scores", Values.BOTH.value: "Both"}
CORRECTED = "ComBat-corrected z-scores"
ALL_FEATURES = "All features"
SUBSAMPLED_ONLY = "With embeddings, the export holds only the subsampled cells."
PICK_FEATURE = "Pick at least one feature."
FILTERS_USED = "The cell table was built with the filters: {names}."
NO_FILTERS_USED = "The cell table was built with no filters."
NO_RECORD = "The run record doesn't say which filters built the cell table."
FILTERS_CHANGED = "The filters switched on now are different. Run Build cell table to export with them."
FILTERS_UNREADABLE = "The saved filters can't be read. Fix them on the Filters page."
GONE_FEATURES = "Not in the cell table any more, so left out:"
GONE_EMBEDDINGS = "Not in the subsample any more, so left out:"
NO_COMBAT_NOW = "No ComBat-corrected copy any more, so the z-scores are as built."
REPLACE = "Replace exports/{file}?"
DELETE = "Delete exports/{file}?"
DELETED = "Deleted exports/{file}."
NO_EXPORTS = "No exports yet."
NO_TABLE = "Run Build cell table first: the export reads its cell table."
LIST_COLUMNS = ("Name", "Kind", "Cells", "Size", "Written", "")
LIST_GRID = "minmax(0,2fr) auto auto auto auto auto"
SIZE_UNITS = (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024))


def _cells_text(number: int) -> str:
    return f"{number:,} cell" + ("" if number == 1 else "s")


def counts_text(counts: Counts, *, corrected: bool) -> str:
    inside = f"{counts.filtered:,} filtered"
    if corrected:
        inside += f"; {counts.uncorrected:,} with no corrected values"
    if counts.subsampled is None:
        return f"The export will hold {_cells_text(counts.cells)} ({inside})."
    noun = "cell" if counts.subsampled == 1 else "cells"
    return f"The export will hold {counts.cells:,} of the {counts.subsampled:,} subsampled {noun} ({inside})."


def size_text(size: int) -> str:
    for unit, scale in SIZE_UNITS:
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{size} B"


def _features(exp: Experiment) -> dict[str, str]:
    table = pd.read_csv(fullest_df_path(exp))
    with h5py.File(adata_path(exp), "r") as file:
        obs = {str(column) for column in file["obs"].attrs.get("column-order", [])}
    numbers = [column for column in table.select_dtypes(include="number").columns if column not in obs]
    info = read_info_file(exp)
    return menu_order(describe(numbers, info), info)


def _embeddings(exp: Experiment) -> dict[str, str] | None:
    path = adata_sub_path(exp)
    if not path.is_file():
        return None
    return embedding_names(ad.read_h5ad(path))


def _no_table() -> None:
    with ui.card().classes("p-8 items-center gap-3 max-w-xl mx-auto mt-10"):
        ui.icon("hourglass_empty").classes("text-5xl muted")
        ui.label(NO_TABLE).classes("text-sm text-center")
        ui.button("Go to Run", icon="play_circle", on_click=lambda: ui.navigate.to("/run"))


class _ExportPage:
    def __init__(self, exp: Experiment) -> None:
        self.exp = exp
        self.features = _features(exp)
        self.embeddings = _embeddings(exp) or {}
        self.combat = has_corrected(exp)
        self.wanted = 0
        self.counting = False
        self.count_problem: str | None = None
        self.blank_tried = False

    def build(self) -> None:
        ui.label("Export").classes("page-title")
        ui.label(TITLE_LINE).classes("text-sm muted")
        with ui.card().classes("w-full max-w-2xl p-5 gap-2"):
            ui.label("New export").classes(SECTION_TITLE)
            self.left_out = ui.column().classes("w-full gap-1")
            self.left_out.set_visibility(False)
            self.name_box = ui.input("Name", on_change=self.new_name).classes("w-full")
            self.name_box.without_auto_validation()
            self.kind = ui.toggle(FORMATS, value=str(default(Step.EXPORT, "file_format"))).props("no-caps")
            ui.label("Values").classes("text-sm font-medium mt-2")
            self.values = ui.toggle(VALUES, value=str(default(Step.EXPORT, "values")), on_change=self.new_values)
            self.values.props("no-caps")
            self.corrected = ui.switch(CORRECTED, value=False, on_change=self.count)
            self.corrected.set_visibility(self.combat)
            self.all_features = ui.switch(ALL_FEATURES, value=True, on_change=self.new_features).classes("mt-2")
            self.picked = (
                ui.select(
                    self.features,
                    value=[],
                    multiple=True,
                    with_input=True,
                    label="Features",
                    on_change=self.new_picked,
                )
                .props("use-chips")
                .classes("w-full")
            )
            self.feature_problem = ui.label(PICK_FEATURE).classes("text-xs text-negative -mt-1")
            self.picked_embeddings = (
                ui.select(
                    self.embeddings,
                    value=[],
                    multiple=True,
                    label="Embeddings (optional)",
                    on_change=self.new_embeddings,
                )
                .props("use-chips")
                .classes("w-full")
            )
            self.picked_embeddings.set_visibility(bool(self.embeddings))
            self.subsampled_line = ui.label(SUBSAMPLED_ONLY).classes("text-xs muted -mt-1")
            with ui.row().classes("items-center gap-2 mt-2"):
                self.spinner = ui.spinner(size="sm")
                self.counts_line = ui.label().classes("text-sm")
            self.filters_lines()
            self.button = ui.button("Export", icon="download", on_click=self.export).classes("mt-2")
        with ui.card().classes("w-full p-5 gap-2"):
            ui.label("Exports").classes(SECTION_TITLE)
            self.exports = ui.column().classes("w-full gap-0")
        self.new_name()
        self.follow_values()
        self.new_features()
        self.follow_embeddings()
        ui.timer(0.1, self.count, once=True)
        ui.timer(0.1, self.show_exports, once=True)
        ui.timer(POLL_SECONDS, self.follow_run)

    def filters_lines(self) -> None:
        built = table_filters(self.exp)
        if built is None:
            ui.label(NO_RECORD).classes("text-sm")
            return
        ui.label(FILTERS_USED.format(names=", ".join(built)) if built else NO_FILTERS_USED).classes("text-sm")
        try:
            now = switched_on_filters(self.exp)
        except ValueError:
            ui.label(FILTERS_UNREADABLE).classes("text-sm text-warning")
            return
        if set(now) != set(built):
            ui.label(FILTERS_CHANGED).classes("text-sm text-warning")

    def chosen_features(self) -> list[str]:
        chosen = set(self.picked.value or [])
        return [column for column in self.features if column in chosen]

    def chosen_embeddings(self) -> list[str]:
        chosen = set(self.picked_embeddings.value or [])
        return [key for key in self.embeddings if key in chosen]

    def is_corrected(self) -> bool:
        return self.combat and bool(self.corrected.value) and self.values.value != Values.MEASURED.value

    def name_shown(self) -> str | None:
        problem = name_problem(self.name_box.value or "")
        return None if problem == BLANK_NAME and not self.blank_tried else problem

    def refusal(self) -> str | None:
        problem = self.name_shown()
        if problem is not None:
            return problem
        if not self.all_features.value and not self.chosen_features():
            return PICK_FEATURE
        return self.count_problem

    def follow_run(self) -> None:
        allowed = not run_going() and self.refusal() is None
        if self.button.enabled != allowed:
            self.button.set_enabled(allowed)

    def new_name(self) -> None:
        if (self.name_box.value or "").strip():
            self.blank_tried = False
        self.name_box.error = self.name_shown()
        self.follow_run()

    def follow_values(self) -> None:
        if self.values.value == Values.MEASURED.value:
            self.corrected.value = False
            self.corrected.disable()
        else:
            self.corrected.enable()

    async def new_values(self) -> None:
        self.follow_values()
        await self.count()

    def new_picked(self) -> None:
        self.picked.run_method("updateInputValue", "")
        self.new_features()

    def new_features(self) -> None:
        self.picked.set_visibility(not self.all_features.value)
        self.feature_problem.set_visibility(not self.all_features.value and not self.chosen_features())
        self.follow_run()

    def follow_embeddings(self) -> None:
        self.subsampled_line.set_visibility(bool(self.chosen_embeddings()))

    async def new_embeddings(self) -> None:
        self.follow_embeddings()
        await self.count()

    async def count(self) -> None:
        self.wanted += 1
        if self.counting:
            return
        self.counting = True
        try:
            while True:
                asked = self.wanted
                corrected, embeddings = self.is_corrected(), self.chosen_embeddings()
                self.spinner.visible = True
                counts: Counts | None = None
                problem: str | None = None
                try:
                    counts = await run.io_bound(export_counts, self.exp, corrected=corrected, embeddings=embeddings)
                except (ValueError, OSError) as error:
                    problem = error_in_words(error)
                if asked == self.wanted:
                    self.show_counts(counts, problem, corrected=corrected)
                    return
        finally:
            self.counting = False
            self.spinner.visible = False

    def show_counts(self, counts: Counts | None, problem: str | None, *, corrected: bool) -> None:
        self.count_problem = problem
        if counts is None:
            self.counts_line.text = problem or ""
            self.counts_line.classes(replace="text-sm text-negative")
        else:
            self.counts_line.text = counts_text(counts, corrected=corrected)
            self.counts_line.classes(replace="text-sm")
        self.follow_run()

    def show_left_out(self, features: list[str], embeddings: list[str], combat_gone: bool) -> None:
        self.left_out.clear()
        with self.left_out:
            if features:
                ui.label("\n".join([GONE_FEATURES, *features])).classes("text-sm text-warning whitespace-pre-wrap")
            if embeddings:
                ui.label("\n".join([GONE_EMBEDDINGS, *embeddings])).classes("text-sm text-warning whitespace-pre-wrap")
            if combat_gone:
                ui.label(NO_COMBAT_NOW).classes("text-sm text-warning")
        self.left_out.set_visibility(bool(features or embeddings or combat_gone))

    def load(self, settings: dict[str, Any]) -> None:
        self.name_box.value = str(settings.get("name") or "")
        if settings.get("file_format") in FORMATS:
            self.kind.value = settings["file_format"]
        if settings.get("values") in VALUES:
            self.values.value = settings["values"]
        sent = settings.get("features")
        gone_features: list[str] = []
        if isinstance(sent, list):
            names = [str(name) for name in sent]
            gone_features = [name for name in names if name not in self.features]
            self.picked.value = [column for column in self.features if column in set(names)]
            self.all_features.value = False
        else:
            self.picked.value = []
            self.all_features.value = True
        sent = settings.get("embeddings")
        keys = [str(key) for key in sent] if isinstance(sent, list) else []
        self.picked_embeddings.value = [key for key in self.embeddings if key in set(keys)]
        combat_gone = settings.get("corrected") is True and not self.combat
        self.corrected.value = settings.get("corrected") is True and self.combat
        self.show_left_out(gone_features, [key for key in keys if key not in self.embeddings], combat_gone)

    async def show_exports(self) -> None:
        exports = await run.io_bound(list_exports, self.exp)
        settings = {export.path: export_settings(self.exp, export) for export in exports}
        self.exports.clear()
        with self.exports:
            if not exports:
                ui.label(NO_EXPORTS).classes("text-sm muted")
                return
            with ui.grid(columns=LIST_GRID).classes("w-full items-center gap-x-6 gap-y-1"):
                for title in LIST_COLUMNS:
                    ui.label(title).classes("text-xs muted font-medium")
                for export in exports:
                    self.export_row(export, settings[export.path])

    def export_row(self, export: ExportFile, settings: dict[str, Any] | None) -> None:
        ui.label(export.path.stem).classes("text-sm break-all")
        ui.label(KINDS[export.file_format]).classes("text-sm")
        ui.label("—" if export.cells is None else f"{export.cells:,}").classes("text-sm text-right tabular-nums")
        ui.label(size_text(export.size)).classes("text-sm text-right tabular-nums")
        ui.label(clock_text(export.written)).classes("text-sm")
        with ui.row().classes("gap-1 no-wrap"):
            load = ui.button("Load settings", on_click=lambda: self.load(settings or {})).props("flat dense no-caps")
            load.set_enabled(settings is not None)
            ui.button("Delete", on_click=lambda: self.delete(export)).props("flat dense no-caps color=negative")

    async def delete(self, export: ExportFile) -> None:
        if not await ask(DELETE.format(file=export.path.name), "Delete"):
            return
        try:
            export.path.unlink()
        except OSError as error:
            ui.notify(error_in_words(error), type="negative")
            return
        ui.notify(DELETED.format(file=export.path.name), type="positive")
        await self.show_exports()

    async def export(self) -> None:
        if not (self.name_box.value or "").strip():
            self.blank_tried = True
            self.new_name()
            return
        if run_going() or self.refusal() is not None:
            return
        name = self.name_box.value
        file_format = str(self.kind.value)
        path = export_path(self.exp, name, file_format)
        overwrite = path.exists()
        if overwrite and not await ask(REPLACE.format(file=path.name), "Replace"):
            return
        if run_going():
            return
        settings = {
            "name": name,
            "file_format": file_format,
            "values": str(self.values.value),
            "corrected": self.is_corrected(),
            "features": None if self.all_features.value else self.chosen_features(),
            "embeddings": self.chosen_embeddings(),
            "overwrite": overwrite,
        }
        try:
            started = Run(self.exp.config_path, [StepRequest(Step.EXPORT, None, settings)])
            started.start()
        except OSError as error:
            ui.notify(f"The run could not start: {error}", type="negative")
            return
        STATE.run = started
        self.show_left_out([], [], False)
        ui.navigate.to("/run")


@ui.page("/export")
def export_page() -> None:
    with page_frame("Export"):
        exp = STATE.experiment
        if exp is None:
            with ui.row().classes("items-center gap-1 text-sm"):
                ui.label("No experiment open.")
                ui.link("Open one on the Experiment page.", "/")
            return
        if not fullest_df_path(exp).is_file() or not adata_path(exp).is_file():
            _no_table()
            return
        _ExportPage(exp).build()
