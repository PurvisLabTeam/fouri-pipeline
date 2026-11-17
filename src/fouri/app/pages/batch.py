"""The Batch effects page."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
from nicegui import run, ui
from nicegui.events import ValueChangeEventArguments

from fouri.app.batches import (
    DEFAULT_BATCH,
    LiveNumbers,
    SavedCorrection,
    adjustment_figure,
    batch_choices,
    classifier_figure,
    keep_choices,
    live_numbers,
    matches,
    save_batch_setup,
    saved_batch_setup,
    saved_correction,
    setup_words,
    shares_figure,
    violin_figure,
)
from fouri.app.cells import Cells, load_cells
from fouri.app.shell import empty_state, page_frame
from fouri.app.state import STATE
from fouri.app.theme import restyle
from fouri.batch import NEAR_CHANCE, SHARE_OVER, default_keep
from fouri.config import Experiment
from fouri.metadata import natural_key
from fouri.parallel import error_in_words

SECTION_TITLE = "text-sm font-medium"
MODE_SECONDS = 0.25

KEEP_LINE = "ComBat removes differences between batches and keeps differences between these groups."
LOG_LINE = (
    "Each channel's nucleus mean, ring mean and nucleus total are corrected on a log scale; ratios and shape as "
    "measured."
)
SAVED = "Saved. Batch effects on the Run page uses this set-up."
CHANGED = "Not saved. Batch effects on the Run page uses the saved set-up: {setup}."
NONE_SAVED = "Save the set-up to run Batch effects on the Run page."
UNREADABLE = "The saved set-up can't be read. Save it again."
NOT_SAVED = "The set-up was not saved: {error}"
CONDITIONS_TITLE = "Conditions and their batches"
LEFT_OUT = "(one {batch} only; left out)"
WORKING = "Working out the numbers…"
FAILED = "The numbers could not be worked out: {error}"
NEAR = (
    "Batches told apart in {score:.0%} of cells, against {chance:.0%} by chance: there is no batch effect to correct, "
    "and correcting is not recommended."
)
UNEVEN = (
    "{condition} is {low:.0%} of {batch} {lower}'s cells but {high:.0%} of {batch} {higher}'s. ComBat may make "
    "conditions spread this unevenly appear further apart than they are."
)
GROUPS = "Corrected apart, sharing no condition: {groups}. The differences between these groups stay as they are."
BEFORE_TITLE = "Before correction"
AFTER_TITLE = "After correction"
AFTER_LOG_TITLE = "After correction, on log intensities"
CELLS_USED = "Cells used: {used:,} of {cells:,}, in {conditions}"
BATCH_MEDIAN = "Median variance explained by batch: {value:.2%}"
FEATURES_OVER = "Features over {over:.0%} variance explained by batch: {count} of {total}"
TOLD_APART = "Batches told apart: {score:.0%} of cells (chance {chance:.0%})"
NOT_SCORED = "Batches told apart: not scored, too few cells in a batch"
CONDITION_MEDIAN = "Median variance explained by condition: {value:.2%}"
NO_CONDITION_MEDIAN = "Median variance explained by condition: needs two or more conditions in two or more batches"
REFERENCE_SCORE = "ComBat on shuffled batches: {score:.0%} of cells"
REFERENCE_NOT_SCORED = "ComBat on shuffled batches: not scored, too few cells in a batch"
RUN_STEP = "Run Batch effects on the Run page to see the corrected numbers."
OTHER_SETUP = "The saved correction is {setup}. Save this set-up and run Batch effects to see its result."
UNDER_CARDS = "A good correction lowers the batch numbers and keeps the condition number."
SHARES_TABLE_TITLE = "Conditions within each batch (% of the batch's cells)"


def _count(number: int, noun: str) -> str:
    return f"{number:,} {noun}" + ("" if number == 1 else "s")


def _same(saved: Mapping[str, Any], setup: Mapping[str, Any]) -> bool:
    return (
        saved["batch"] == setup["batch"]
        and list(saved["keep"]) == list(setup["keep"])
        and bool(saved["log_intensities"]) == bool(setup["log_intensities"])
    )


def _reason(error: Exception) -> str:
    return error.strerror if isinstance(error, OSError) and error.strerror else str(error)


def _plot_config(name: str) -> dict[str, Any]:
    return {"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": name, "scale": 2}}


def _show_figure(fig: go.Figure, name: str, height: int) -> ui.plotly:
    restyle(fig, dark=STATE.dark)
    return (
        ui.plotly({**fig.to_plotly_json(), "config": _plot_config(name)})
        .classes("w-full")
        .style(f"height: {height}px")
    )


def _number_lines(
    batch_median: float,
    features_over: int,
    total: int,
    classifier: float | None,
    chance: float,
    condition_median: float | None,
) -> None:
    ui.label(BATCH_MEDIAN.format(value=batch_median)).classes("text-sm")
    ui.label(FEATURES_OVER.format(over=SHARE_OVER, count=features_over, total=total)).classes("text-sm")
    if classifier is None:
        ui.label(NOT_SCORED).classes("text-sm")
    else:
        ui.label(TOLD_APART.format(score=classifier, chance=chance)).classes("text-sm")
    if condition_median is None:
        ui.label(NO_CONDITION_MEDIAN).classes("text-sm muted")
    else:
        ui.label(CONDITION_MEDIAN.format(value=condition_median)).classes("text-sm")


def _warning(text: str) -> None:
    with ui.row().classes("items-start gap-2 no-wrap"):
        ui.icon("warning").classes("text-lg text-warning shrink-0")
        ui.label(text).classes("text-sm")


class _BatchPage:
    def __init__(self, exp: Experiment, cells: Cells) -> None:
        self.exp = exp
        self.names = {column: feature.title for column, feature in cells.features.items()}
        try:
            self.saved_setup = saved_batch_setup(exp)
            self.unreadable = False
        except ValueError:
            self.saved_setup, self.unreadable = None, True
        choices = batch_choices(exp)
        if self.saved_setup is not None:
            saved_batch = self.saved_setup["batch"]
            self.batch = saved_batch if saved_batch in choices else DEFAULT_BATCH
            wanted = set(self.saved_setup["keep"])
            self.log_intensities = bool(self.saved_setup["log_intensities"])
        else:
            self.batch = DEFAULT_BATCH
            wanted = set(default_keep(exp))
            self.log_intensities = True
        self.keep = [column for column in keep_choices(exp, self.batch) if column in wanted]
        self.saved = saved_correction(exp)
        self.live: LiveNumbers | None = None
        self.live_batch = self.batch
        self.asked = 0
        self.working = False
        self.quiet = False
        self.dark = STATE.dark
        self.condition: str | None = None
        self.violins: dict[str, go.Figure] = {}

    def setup(self) -> dict[str, Any]:
        return {"batch": self.batch, "keep": list(self.keep), "log_intensities": self.log_intensities}

    def build(self) -> None:
        with ui.row().classes("w-full no-wrap items-start gap-6"):
            with ui.column().classes("w-[320px] shrink-0 gap-2"):
                ui.label("Batch effects").classes("page-title")
                ui.select(batch_choices(self.exp), value=self.batch, label="Batch", on_change=self.new_batch).classes(
                    "w-full"
                )
                self.keep_box = (
                    ui.select(
                        keep_choices(self.exp, self.batch),
                        value=list(self.keep),
                        multiple=True,
                        label="Biology to keep",
                        on_change=self.new_keep,
                    )
                    .props("use-chips")
                    .classes("w-full")
                )
                ui.label(KEEP_LINE).classes("text-xs muted -mt-1")
                ui.switch("Log-transform intensities", value=self.log_intensities, on_change=self.new_log)
                ui.label(LOG_LINE).classes("text-xs muted -mt-2")
                ui.button("Save set-up", icon="save", on_click=self.save).classes("mt-2")
                self.saved_line = ui.label().classes("text-xs")
                ui.label(CONDITIONS_TITLE).classes(f"{SECTION_TITLE} mt-3")
                self.conditions_box = ui.column().classes("w-full gap-1")
            with ui.column().classes("grow min-w-0 gap-3"):
                with ui.row().classes("items-center gap-2") as self.working_row:
                    ui.spinner(size="sm")
                    ui.label(WORKING).classes("text-sm muted")
                self.error_line = ui.label().classes("text-sm text-negative")
                self.error_line.set_visibility(False)
                self.results = ui.column().classes("w-full gap-3")
        self.show_saved_line()
        ui.timer(0, self.work_out, once=True)
        ui.timer(MODE_SECONDS, self.follow_mode)

    def show_saved_line(self) -> None:
        if self.unreadable:
            self.saved_line.text = UNREADABLE
            self.saved_line.classes(replace="text-xs text-negative")
            return
        self.saved_line.classes(replace="text-xs")
        if self.saved_setup is None:
            self.saved_line.text = NONE_SAVED
        elif _same(self.saved_setup, self.setup()):
            self.saved_line.text = SAVED
        else:
            words = setup_words(
                self.saved_setup["batch"], self.saved_setup["keep"], bool(self.saved_setup["log_intensities"])
            )
            self.saved_line.text = CHANGED.format(setup=words)

    async def new_batch(self, e: ValueChangeEventArguments[Any]) -> None:
        if not e.value or e.value == self.batch:
            return
        self.batch = e.value
        offered = keep_choices(self.exp, self.batch)
        self.keep = [column for column in offered if column in set(self.keep)]
        self.quiet = True
        try:
            self.keep_box.set_options(offered, value=list(self.keep))
        finally:
            self.quiet = False
        self.show_saved_line()
        await self.work_out()

    async def new_keep(self, e: ValueChangeEventArguments[Any]) -> None:
        if self.quiet:
            return
        chosen = set(e.value or [])
        keep = [column for column in keep_choices(self.exp, self.batch) if column in chosen]
        if list(e.value or []) != keep:
            self.quiet = True
            try:
                self.keep_box.value = list(keep)
            finally:
                self.quiet = False
        if keep == self.keep:
            return
        self.keep = keep
        self.show_saved_line()
        await self.work_out()

    def new_log(self, e: ValueChangeEventArguments[Any]) -> None:
        self.log_intensities = bool(e.value)
        self.show_saved_line()
        self.show()

    def save(self) -> None:
        setup = self.setup()
        try:
            self.exp = save_batch_setup(self.exp, setup)
        except (OSError, ValueError) as error:
            ui.notify(NOT_SAVED.format(error=_reason(error)), type="negative")
            return
        self.saved_setup = setup
        self.unreadable = False
        self.show_saved_line()

    async def work_out(self) -> None:
        self.asked += 1
        ticket = self.asked
        batch, keep = self.batch, list(self.keep)
        self.working = True
        self.working_row.set_visibility(True)
        self.error_line.set_visibility(False)
        self.results.clear()
        self.conditions_box.clear()
        try:
            live = await run.io_bound(live_numbers, self.exp, batch, keep)
        except Exception as error:
            if ticket == self.asked:
                self.working = False
                self.working_row.set_visibility(False)
                self.error_line.text = FAILED.format(error=error_in_words(error))
                self.error_line.set_visibility(True)
            return
        if ticket != self.asked:
            return
        self.working = False
        self.working_row.set_visibility(False)
        self.live, self.live_batch = live, batch
        self.show_conditions()
        self.show()

    def show_conditions(self) -> None:
        self.conditions_box.clear()
        if self.live is None:
            return
        with self.conditions_box:
            table = self.live.conditions
            for condition, batches, in_two in zip(table["condition"], table["batches"], table["usable"], strict=True):
                usable = bool(in_two)
                text = f"{condition}: {', '.join(str(batch) for batch in batches)}"
                if not usable:
                    text += f" {LEFT_OUT.format(batch=self.live_batch)}"
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.icon("check" if usable else "block").classes(
                        "text-base " + ("text-positive" if usable else "muted")
                    )
                    ui.label(text).classes("text-xs")

    def matched(self) -> SavedCorrection | None:
        if self.saved is not None and matches(self.saved, self.setup()):
            return self.saved
        return None

    def warnings(self, live: LiveNumbers) -> list[str]:
        texts = [live.problem] if live.problem else []
        metrics = live.metrics
        if (
            metrics is not None
            and metrics.classifier is not None
            and metrics.classifier - metrics.chance <= NEAR_CHANCE
        ):
            texts.append(NEAR.format(score=metrics.classifier, chance=metrics.chance))
        for condition, lower, higher in live.uneven:
            texts.append(
                UNEVEN.format(
                    condition=condition,
                    batch=self.live_batch,
                    lower=lower,
                    higher=higher,
                    low=cast(float, live.shares.at[lower, condition]),
                    high=cast(float, live.shares.at[higher, condition]),
                )
            )
        if len(live.groups) >= 2:
            texts.append(GROUPS.format(groups="; ".join(f"({', '.join(group)})" for group in live.groups)))
        return texts

    def before_card(self, live: LiveNumbers) -> None:
        ui.label(BEFORE_TITLE).classes(SECTION_TITLE)
        usable = int(live.conditions["usable"].sum())
        ui.label(CELLS_USED.format(used=live.used, cells=live.cells, conditions=_count(usable, "condition"))).classes(
            "text-sm"
        )
        metrics = live.metrics
        if metrics is None:
            return
        _number_lines(
            metrics.batch_median,
            metrics.features_over,
            len(metrics.shares),
            metrics.classifier,
            metrics.chance,
            metrics.condition_median,
        )

    def after_card(self, matched: SavedCorrection | None) -> None:
        if matched is None:
            ui.label(AFTER_TITLE).classes(SECTION_TITLE)
            if self.saved is None:
                ui.label(RUN_STEP).classes("text-sm muted")
            else:
                words = setup_words(self.saved.batch, self.saved.keep, bool(self.saved.log_transformed))
                ui.label(OTHER_SETUP.format(setup=words)).classes("text-sm muted")
            return
        ui.label(AFTER_LOG_TITLE if matched.log_transformed else AFTER_TITLE).classes(SECTION_TITLE)
        after = matched.after
        if not after:
            return
        _number_lines(
            float(after["batch_median"]),
            int(after["features_over"]),
            len(after["features"]),
            after.get("classifier"),
            float(after["chance"]),
            after.get("condition_median"),
        )
        reference = matched.reference.get("classifier")
        if reference is None:
            ui.label(REFERENCE_NOT_SCORED).classes("text-sm")
        else:
            ui.label(REFERENCE_SCORE.format(score=float(reference))).classes("text-sm")

    def shares_table(self, live: LiveNumbers) -> None:
        ui.label(SHARES_TABLE_TITLE).classes(SECTION_TITLE)
        conditions = [str(condition) for condition in live.shares.columns]
        columns = [{"name": "batch", "label": "Batch", "field": "batch", "align": "left"}]
        columns += [
            {"name": f"c{index}", "label": condition, "field": f"c{index}", "align": "right"}
            for index, condition in enumerate(conditions)
        ]
        rows = []
        for batch, shares in live.shares.iterrows():
            row: dict[str, Any] = {"batch": str(batch)}
            for index, value in enumerate(shares.to_numpy(dtype=np.float64)):
                row[f"c{index}"] = "" if np.isnan(value) else f"{value:.0%}"
            rows.append(row)
        ui.table(columns=columns, rows=rows, row_key="batch").props("flat dense").classes("w-full")

    def show(self) -> None:
        self.dark = STATE.dark
        self.results.clear()
        live = self.live
        if live is None or self.working:
            return
        with self.results:
            if live.why_not is not None:
                ui.label(live.why_not).classes("text-sm")
                return
            for text in self.warnings(live):
                _warning(text)
            matched = self.matched()
            with ui.row().classes("w-full no-wrap gap-4"):
                with ui.card().classes("w-1/2 p-4 gap-1"):
                    self.before_card(live)
                with ui.card().classes("w-1/2 p-4 gap-1"):
                    self.after_card(matched)
            ui.label(UNDER_CARDS).classes("text-xs muted -mt-1")
            _show_figure(shares_figure(live, matched, self.names), "batch-shares", 440)
            _show_figure(classifier_figure(live, matched), "batch-classifier", 380)
            if matched is not None:
                self.saved_figures(matched)
            self.shares_table(live)

    def saved_figures(self, saved: SavedCorrection) -> None:
        conditions = sorted(saved.conditions, key=natural_key)
        if conditions:
            if self.condition not in conditions:
                self.condition = conditions[0]
            ui.select(conditions, value=self.condition, label="Condition", on_change=self.new_condition).classes(
                "w-64"
            )
            self.violin_box = ui.column().classes("w-full gap-0")
            ui.timer(0, self.draw_violins, once=True)
        height = max(320, 140 + 32 * len(saved.adjustment.index))
        _show_figure(adjustment_figure(saved, self.names), "batch-adjustment", height)

    async def new_condition(self, e: ValueChangeEventArguments[Any]) -> None:
        if e.value and e.value != self.condition:
            self.condition = e.value
            await self.draw_violins()

    async def draw_violins(self) -> None:
        saved, condition = self.saved, self.condition
        if saved is None or condition is None:
            return
        fig = self.violins.get(condition)
        if fig is None:
            try:
                fig = await run.io_bound(violin_figure, self.exp, saved, condition, self.names)
            except Exception as error:
                self.violin_box.clear()
                with self.violin_box:
                    ui.label(FAILED.format(error=error_in_words(error))).classes("text-sm text-negative")
                return
            self.violins[condition] = fig
        if condition != self.condition or self.violin_box.is_deleted:
            return
        self.violin_box.clear()
        with self.violin_box:
            _show_figure(go.Figure(fig), "batch-violins", 440)

    def follow_mode(self) -> None:
        if self.dark != STATE.dark and not self.working:
            self.show()


@ui.page("/batch")
def batch_page() -> None:
    with page_frame("Batch effects"):
        exp = STATE.experiment
        if exp is None:
            with ui.row().classes("items-center gap-1 text-sm"):
                ui.label("No experiment open.")
                ui.link("Open one on the Experiment page.", "/")
            return
        cells = load_cells(exp)
        empty_state(cells)
        if not cells.measured:
            return
        _BatchPage(exp, cells).build()
