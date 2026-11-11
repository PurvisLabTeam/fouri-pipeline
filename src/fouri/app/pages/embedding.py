"""The Embedding page."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, cast

import anndata as ad
import pandas as pd
import plotly.graph_objects as go
from nicegui import run, ui
from nicegui.events import ValueChangeEventArguments

from fouri.app.cells import CONDITION, Cells, add_groups, load_cells
from fouri.app.embeddings import (
    DEFAULT_SETUP,
    LONGEST,
    RECORDS,
    SETTING_LABELS,
    SHORTEST,
    box_shape,
    data_proportions,
    embedding_keys,
    embedding_names,
    read_number,
    read_values,
    record_rows,
    run_seconds,
    runs_at_once,
    sample_names,
    sample_number,
    save_setup,
    saved_setup,
    setup_runs,
    time_text,
    total_seconds,
)
from fouri.app.names import Feature, describe, marker_menu, menu_order
from fouri.app.shell import empty_state, page_frame
from fouri.app.state import STATE
from fouri.app.steps import default
from fouri.app.theme import on_white_panel, restyle
from fouri.config import Experiment
from fouri.embedding import Method, Run, rank_features, run_key
from fouri.io import adata_sub_path, read_info_file
from fouri.parallel import error_in_words
from fouri.plots import plot_embedding
from fouri.status import Step

MARKER, GROUP = "Marker", "Group"
SECTION_TITLE = "text-sm font-medium"
AXIS_NAMES = ("Axis 1", "Axis 2", "Axis 3")
SLIDER_STEP = 0.1
CUBE = (1.0, 1.0, 1.0)
MODE_SECONDS = 0.25
PLOT_MARGIN = {"l": 40, "r": 20, "t": 50, "b": 40}
WHITE_PANEL = "background: #FFFFFF; border-radius: 12px"
NO_PANEL = "background: transparent"
SCALE_NOTE = "Markers are colored by their z-score; the color scale runs from the 1st to the 99th percentile."

VIEW_TAB = "View embeddings"
SETUP_TAB = "Set up an embedding"
INDEPENDENT = "Leave out DNA and EdU to keep them as an independent check on the embedding."
VALUES_HINT = "Enter one or more values separated by commas (e.g. 40, 80); one embedding is made per combination."
RANKING_TITLE = "Feature ranking: Laplacian score on the picked features' neighbor graph (lower = smoother)"
NO_SUBSAMPLE = (
    "No subsample yet, so no ranking. Save the set-up and run Embedding on the Run page: it makes the "
    "subsample first, then these embeddings."
)
NO_SAMPLE_IDS = "The wells in experiment.json have no sample IDs, so there are no samples to pick."
PICK_FEATURE = "Pick at least one feature."
PICK_SAMPLE = "Pick at least one sample."
NONE_TICKED = "No embeddings ticked."
CANT_READ = "The saved set-up can't be read: {reason} Save to replace it."
RANKING_FAILED = "Can't rank the features: {error}"
K_TITLE = "Neighbors (k) for the ranking"
TOP_N_TITLE = "Top N"
RANKING_COLUMNS = [
    {"name": "place", "label": "Place", "field": "place", "align": "right"},
    {"name": "feature", "label": "Feature", "field": "feature", "align": "left"},
    {"name": "score", "label": "Score", "field": "score", "align": "right"},
    {"name": "picked", "label": "Picked", "field": "picked", "align": "center"},
]
RUN_COLUMNS = [
    {"name": "method", "label": "Method", "field": "method", "align": "left"},
    {"name": "dims", "label": "Dims", "field": "dims", "align": "left"},
    {"name": "top_n", "label": "Top N", "field": "top_n", "align": "right"},
    {"name": "settings", "label": "Settings", "field": "settings", "align": "left", "style": "white-space: normal"},
    {"name": "about", "label": "About", "field": "about", "align": "right"},
]


def _read(exp: Experiment) -> ad.AnnData | None:
    path = adata_sub_path(exp)
    return ad.read_h5ad(path) if path.is_file() else None


def _dims(adata: ad.AnnData, key: str) -> int:
    return int(adata.uns[RECORDS][key]["dims"])


def _on_step(value: float) -> float:
    return round(round(value / SLIDER_STEP) * SLIDER_STEP, 1)


@dataclass
class _View:
    key: str
    mode: str
    marker: str
    measure: str
    group: str
    shape: list[float] = field(default_factory=lambda: list(CUBE))
    figure: go.Figure | None = None
    dark: bool = False
    quiet: bool = False


def _empty_state() -> None:
    with ui.card().classes("p-8 items-center gap-3 max-w-xl mx-auto mt-10"):
        ui.icon("bubble_chart").classes("text-5xl muted")
        ui.label("No embedding yet").classes("text-[1.05rem] font-semibold")
        ui.label("Set them up in the Set up an embedding tab, then run Embedding on the Run page.").classes(
            "text-sm muted text-center"
        )


def _plot_config(key: str) -> dict[str, Any]:
    return {"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": key, "scale": 2}}


def _view_tab(adata: ad.AnnData, exp: Experiment) -> None:
    keys = embedding_keys(adata)
    obs = cast(pd.DataFrame, adata.obs).copy()
    groups = add_groups(obs, exp)
    adata.obs = obs
    info = read_info_file(exp)
    menu = marker_menu(describe(list(adata.var_names), info), info)
    first_marker = next(iter(menu))
    view = _View(
        key=keys[0],
        mode=GROUP,
        marker=first_marker,
        measure=next(iter(menu[first_marker])),
        group=CONDITION if CONDITION in groups else groups[0],
    )

    with ui.row().classes("w-full no-wrap items-start gap-6"):
        with ui.column().classes("w-80 shrink-0 gap-2"):
            ui.select(
                embedding_names(adata), value=view.key, label="Embedding", on_change=lambda e: choose(e.value)
            ).classes("w-full")

            ui.label("Color by").classes(f"{SECTION_TITLE} mt-2")
            ui.toggle([MARKER, GROUP], value=view.mode, on_change=lambda e: color_by(e.value))
            with ui.row().classes("w-full no-wrap gap-1") as marker_row:
                marker_box = ui.select(list(menu), value=view.marker, label="Marker", with_input=True).classes(
                    "w-7/12"
                )
                measure_box = ui.select(list(menu[view.marker]), value=view.measure, label="measure").classes("w-5/12")
            group_box = ui.select(groups, value=view.group, label="Group").classes("w-full")
            note = ui.label(SCALE_NOTE).classes("text-xs muted")

            with ui.column().classes("w-full gap-0 mt-3") as shape_box:
                ui.label("Box shape").classes(SECTION_TITLE)
                sliders = []
                for axis, name in enumerate(AXIS_NAMES):
                    with ui.row().classes("w-full items-center no-wrap gap-2"):
                        ui.label(name).classes("text-xs muted w-12 shrink-0")
                        sliders.append(
                            ui.slider(
                                min=SHORTEST,
                                max=LONGEST,
                                step=SLIDER_STEP,
                                value=view.shape[axis],
                                on_change=lambda e, axis=axis: stretch(axis, e.value),
                            )
                            .props("label")
                            .classes("grow")
                        )
                with ui.row().classes("gap-2"):
                    ui.button("Cube", on_click=lambda: set_shape(CUBE)).props("outline dense")
                    ui.button("Data proportions", on_click=lambda: set_shape(data_proportions(adata, view.key))).props(
                        "outline dense"
                    )

            ui.label("Record").classes(f"{SECTION_TITLE} mt-3")
            record_box = ui.column().classes("w-full gap-1")
        plot = ui.plotly({"data": [], "layout": {}}).classes("grow min-w-0 h-[82vh]")

    def show_controls() -> None:
        marker_row.set_visibility(view.mode == MARKER)
        note.set_visibility(view.mode == MARKER)
        group_box.set_visibility(view.mode == GROUP)
        shape_box.set_visibility(_dims(adata, view.key) == 3)

    def show_record() -> None:
        record_box.clear()
        with record_box:
            for label, value in record_rows(adata, view.key):
                with ui.row().classes("w-full no-wrap gap-2 text-sm"):
                    ui.label(label).classes("muted w-28 shrink-0")
                    ui.label(value).classes("min-w-0 break-words whitespace-pre-line")

    def draw() -> None:
        color = menu[view.marker][view.measure] if view.mode == MARKER else view.group
        fig = plot_embedding(adata, view.key, color)
        fig.update_layout(margin=PLOT_MARGIN)
        if _dims(adata, view.key) == 3:
            box_shape(fig, *view.shape)
        white = on_white_panel(fig, dark=STATE.dark)
        restyle(fig, dark=STATE.dark)
        view.figure, view.dark = fig, STATE.dark
        plot.style(replace=WHITE_PANEL if white else NO_PANEL)
        plot.update_figure({**fig.to_plotly_json(), "config": _plot_config(view.key)})

    def set_sliders(shape: tuple[float, ...]) -> None:
        view.quiet = True
        try:
            for slider, length in zip(sliders, shape, strict=True):
                slider.value = _on_step(length)
        finally:
            view.quiet = False
        view.shape = [float(slider.value) for slider in sliders]

    def apply_shape() -> None:
        if view.figure is None or _dims(adata, view.key) != 3:
            return
        box_shape(view.figure, *view.shape)
        ratio = view.figure.layout.scene.aspectratio
        plot.run_plot_method(
            "relayout", {"scene.aspectmode": "manual", "scene.aspectratio": {"x": ratio.x, "y": ratio.y, "z": ratio.z}}
        )

    def stretch(axis: int, length: float | None) -> None:
        if view.quiet or length is None:
            return
        view.shape[axis] = float(length)
        apply_shape()

    def set_shape(shape: tuple[float, ...]) -> None:
        set_sliders(shape)
        apply_shape()

    def choose(key: str | None) -> None:
        if not key or key == view.key:
            return
        view.key = key
        set_sliders(CUBE)
        show_controls()
        show_record()
        draw()

    def color_by(mode: str | None) -> None:
        if not mode or mode == view.mode:
            return
        view.mode = mode
        show_controls()
        draw()

    def new_marker(e: ValueChangeEventArguments[Any]) -> None:
        if not e.value or e.value == view.marker:
            return
        options = list(menu[e.value])
        view.marker = e.value
        view.measure = view.measure if view.measure in options else options[0]
        measure_box.set_options(options, value=view.measure)
        draw()

    def new_measure(e: ValueChangeEventArguments[Any]) -> None:
        if e.value and e.value != view.measure:
            view.measure = e.value
            draw()

    def new_group(e: ValueChangeEventArguments[Any]) -> None:
        if e.value and e.value != view.group:
            view.group = e.value
            draw()

    def follow_mode() -> None:
        if view.figure is not None and view.dark != STATE.dark:
            draw()

    marker_box.on_value_change(new_marker)
    measure_box.on_value_change(new_measure)
    group_box.on_value_change(new_group)
    ui.timer(MODE_SECONDS, follow_mode)

    show_controls()
    show_record()
    draw()


@dataclass
class _Box:
    title: str
    name: str
    element: ui.input
    values: list[Any] = field(default_factory=list)
    error: str | None = None
    refusal: str | None = None


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def _text(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values)


def _count(number: int, noun: str) -> str:
    return f"{number:,} {noun}" + ("" if number == 1 else "s")


def _listed(heading: str, items: list[str]) -> str:
    return "\n".join([heading, *items])


def _left_out(features: list[str], samples: list[int], where: str) -> str:
    return _listed(f"Not in {where}, so left out:", [*features, *(f"sample {sample}" for sample in samples)])


def _not_saved(error: Exception) -> str:
    text = error.strerror if isinstance(error, OSError) and error.strerror else str(error)
    return f"Not saved: {text.rstrip('.')}"


class _SetupTab:
    def __init__(
        self, exp: Experiment, adata: ad.AnnData | None, features: dict[str, Feature], table: pd.DataFrame
    ) -> None:
        self.exp = exp
        self.adata = adata
        self.options = menu_order(features, read_info_file(exp))
        self.samples = sample_names(table, exp)
        numbers: Counter[int | None] = Counter()
        if "sample_ID" in table:
            numbers.update(sample_number(text) for text in table["sample_ID"].astype(str))
        self.cells_in = {sample: count for sample, count in numbers.items() if sample is not None}
        self.where = "the subsample" if adata is not None else "the measured cells"
        self.cells_per_sample = int(default(Step.EMBEDDING, "cells_per_sample"))
        self.order: list[str] = []
        self.keys: list[str] = []
        self.seconds: dict[str, float] = {}
        self.wanted = 0
        self.ranking = False

        try:
            saved = saved_setup(exp)
            self.unreadable = ""
        except ValueError as error:
            saved, self.unreadable = None, CANT_READ.format(reason=error)
        setup = copy.deepcopy(saved if saved is not None else DEFAULT_SETUP)
        chosen = set(setup["picked"])
        self.picked = [column for column in self.options if column in chosen]
        self.left_out = ""
        if saved is None:
            self.picked_samples = list(self.samples)
        else:
            self.picked_samples = [sample for sample in self.samples if sample in set(setup["samples"])]
            missing_features = [feature for feature in setup["picked"] if feature not in self.options]
            missing_samples = [sample for sample in setup["samples"] if sample not in self.samples]
            if missing_features or missing_samples:
                self.left_out = _left_out(missing_features, missing_samples, self.where)
        self.setup = setup
        self.unticked: set[str] = set(setup["skipped"])
        self.dims = {method: set(setup["methods"][method.value]["dims"]) for method in Method}
        self.k: int | None = setup["ranking_k"]
        self.seed: int | None = setup["seed"]
        self.k_error: str | None = None
        self.seed_error: str | None = None

    def build(self) -> None:
        setup = self.setup
        with ui.row().classes("w-full no-wrap items-start gap-6"):
            with ui.column().classes("w-[440px] shrink-0 gap-2"):
                self.unreadable_line = ui.label(self.unreadable).classes("text-sm text-negative")
                self.unreadable_line.set_visibility(bool(self.unreadable))
                self.left_out_line = ui.label(self.left_out).classes("text-sm text-warning whitespace-pre-wrap")
                self.left_out_line.set_visibility(bool(self.left_out))
                ui.select(
                    self.options,
                    value=list(self.picked),
                    multiple=True,
                    with_input=True,
                    label="Features for the neighbor graph",
                    on_change=self.new_picked,
                ).props("use-chips").classes("w-full")
                ui.label(INDEPENDENT).classes("text-xs muted -mt-1")
                if self.samples:
                    ui.select(
                        self.samples,
                        value=list(self.picked_samples),
                        multiple=True,
                        label="Samples",
                        on_change=self.new_samples,
                    ).props("use-chips").classes("w-full")
                else:
                    ui.label(NO_SAMPLE_IDS).classes("text-sm text-warning")
                with ui.row().classes("w-full no-wrap gap-2 items-start"):
                    self.k_box = ui.number(K_TITLE, value=setup["ranking_k"], step=1, on_change=self.new_k)
                    self.k_box.without_auto_validation().classes("w-1/2")
                    top_n = ui.input(
                        "Top-ranked features to use (one or more)",
                        value=_text(setup["top_n"]),
                        on_change=self.new_top_n,
                    )
                    top_n.without_auto_validation().classes("w-1/2")
                self.top_n = _Box(TOP_N_TITLE, "top_n", top_n)
                self.used = ui.column().classes("w-full gap-1")

                ui.label("Embeddings to make").classes(f"{SECTION_TITLE} mt-3")
                self.boxes: dict[Method, list[_Box]] = {}
                for method in Method:
                    spec = setup["methods"][method.value]
                    with ui.row().classes("w-full no-wrap items-start gap-3"):
                        with ui.column().classes("w-16 shrink-0 gap-0 pt-2"):
                            ui.label(method.value).classes("text-sm font-medium")
                            for dims in (2, 3):
                                ui.checkbox(
                                    f"{dims}D",
                                    value=dims in self.dims[method],
                                    on_change=lambda e, method=method, dims=dims: self.new_dims(method, dims, e.value),
                                ).props("dense")
                        with ui.row().classes("grow gap-x-2 gap-y-0"):
                            self.boxes[method] = []
                            for name, label in SETTING_LABELS[method].items():
                                element = ui.input(label, value=_text(spec["settings"][name]))
                                element.without_auto_validation().classes("w-[31%]")
                                box = _Box(f"{method} {label}", name, element)
                                element.on_value_change(lambda _, box=box: self.new_setting(box))
                                self.boxes[method].append(box)
                ui.label(VALUES_HINT).classes("text-xs muted")
                self.seed_box = ui.number("Seed", value=setup["seed"], step=1, on_change=self.new_seed)
                self.seed_box.without_auto_validation().classes("w-28")
                self.count_line = ui.label().classes("text-sm")
                ui.button("Save set-up", icon="save", on_click=self.save)

            with ui.column().classes("grow min-w-0 gap-2"):
                if self.adata is None:
                    ui.label(NO_SUBSAMPLE).classes("text-sm muted")
                else:
                    ui.label(RANKING_TITLE).classes(SECTION_TITLE)
                    with ui.row().classes("items-center gap-2"):
                        self.spinner = ui.spinner(size="sm")
                        self.status = ui.label().classes("text-sm muted whitespace-pre-wrap")
                    self.ranking_table = (
                        ui.table(columns=RANKING_COLUMNS, rows=[], row_key="place", pagination=12)
                        .props("flat dense")
                        .classes("w-full")
                    )
                ui.label("Embeddings to compute (untick to skip)").classes(f"{SECTION_TITLE} mt-4")
                self.runs_table = (
                    ui.table(
                        columns=RUN_COLUMNS,
                        rows=[],
                        row_key="key",
                        selection="multiple",
                        pagination=10,
                        on_select=self.new_ticks,
                    )
                    .props("flat dense")
                    .classes("w-full")
                )

        self.check_k()
        self.check_seed()
        self.check_top_n()
        for method in Method:
            for box in self.boxes[method]:
                self.check(box)
        self.list_runs()
        if self.adata is not None:
            self.spinner.visible = False
            ui.timer(0.1, self.rank, once=True)

    def check_k(self) -> None:
        self.k, problem = read_number(self.k_box.value, "ranking_k")
        self.k_error = None if problem is None else f"{K_TITLE}: {problem}"
        self.k_box.error = None if problem is None else _sentence(problem)

    def check_seed(self) -> None:
        self.seed, problem = read_number(self.seed_box.value, "seed")
        self.seed_error = None if problem is None else f"Seed: {problem}"
        self.seed_box.error = None if problem is None else _sentence(problem)

    def check_top_n(self) -> None:
        box = self.top_n
        values, problem = read_values(box.element.value, "top_n")
        box.values = sorted(values)
        if problem is not None:
            box.error, box.refusal = _sentence(problem), f"{TOP_N_TITLE}: {problem}"
        else:
            high = [n for n in box.values if n > len(self.picked)]
            picked = _count(len(self.picked), "picked feature")
            box.error = box.refusal = f"Top N {high[0]} is more than the {picked}." if high else None
        box.element.error = box.error

    def check(self, box: _Box) -> None:
        box.values, problem = read_values(box.element.value, box.name)
        box.error = None if problem is None else _sentence(problem)
        box.refusal = None if problem is None else f"{box.title}: {problem}"
        box.element.error = box.error

    async def new_picked(self, e: ValueChangeEventArguments[Any]) -> None:
        e.sender.run_method("updateInputValue", "")
        chosen = set(e.value or [])
        self.picked = [column for column in self.options if column in chosen]
        self.check_top_n()
        self.list_runs()
        self.show_used()
        await self.rank()

    async def new_samples(self, e: ValueChangeEventArguments[Any]) -> None:
        chosen = set(e.value or [])
        self.picked_samples = [sample for sample in self.samples if sample in chosen]
        self.list_runs()
        await self.rank()

    async def new_k(self) -> None:
        self.check_k()
        await self.rank()

    def new_seed(self) -> None:
        self.check_seed()

    def new_top_n(self) -> None:
        self.check_top_n()
        self.show_used()
        self.list_runs()

    def new_setting(self, box: _Box) -> None:
        self.check(box)
        self.list_runs()

    def new_dims(self, method: Method, dims: int, on: bool) -> None:
        if on:
            self.dims[method].add(dims)
        else:
            self.dims[method].discard(dims)
        self.list_runs()

    def chosen_methods(self) -> dict[str, Any]:
        return {
            method.value: {
                "dims": sorted(self.dims[method]),
                "settings": {box.name: list(box.values) for box in self.boxes[method]},
            }
            for method in Method
        }

    def cell_count(self) -> int:
        if self.adata is None:
            return self.cells_per_sample * len(self.picked_samples)
        return sum(self.cells_in.get(sample, 0) for sample in self.picked_samples)

    def settings_text(self, one: Run) -> str:
        labels = SETTING_LABELS[one.method]
        return ", ".join(f"{labels.get(name, name)} {value}" for name, value in one.settings.items())

    def list_runs(self) -> None:
        top_n = [n for n in self.top_n.values if n <= len(self.picked)]
        runs = setup_runs({"top_n": top_n, "methods": self.chosen_methods()})
        cells = self.cell_count()
        self.seconds = {}
        rows: list[dict[str, Any]] = []
        for one in runs:
            key = run_key(one)
            self.seconds[key] = run_seconds(one, cells)
            rows.append(
                {
                    "key": key,
                    "method": one.method.value,
                    "dims": f"{one.dims}D",
                    "top_n": one.top_n,
                    "settings": self.settings_text(one),
                    "about": time_text(self.seconds[key]),
                }
            )
        self.keys = [row["key"] for row in rows]
        self.runs_table.rows = rows
        self.runs_table.selected = [row for row in rows if row["key"] not in self.unticked]
        self.runs_table.update()
        self.show_count()

    def ticked(self) -> list[str]:
        return [key for key in self.keys if key not in self.unticked]

    def new_ticks(self) -> None:
        selected = {row["key"] for row in self.runs_table.selected}
        self.unticked = (self.unticked - set(self.keys)) | {key for key in self.keys if key not in selected}
        self.show_count()

    def show_count(self) -> None:
        ticked = self.ticked()
        if not ticked:
            self.count_line.text = NONE_TICKED
            return
        parallel = runs_at_once(len(ticked))
        total = total_seconds([self.seconds[key] for key in ticked], parallel)
        cells = "" if self.adata is not None else f" at {self.cells_per_sample:,} cells per sample"
        self.count_line.text = (
            f"{_count(len(ticked), 'embedding')}, about {time_text(total)}{cells} ({parallel} in parallel on this PC)."
        )

    def show_used(self) -> None:
        self.used.clear()
        if not self.order:
            return
        chosen = set(self.picked)
        in_order = [feature for feature in self.order if feature in chosen]
        shown = 0
        with self.used:
            for n in self.top_n.values:
                if n > len(in_order):
                    break
                names = [self.options.get(feature, feature) for feature in in_order[shown:n]]
                heading = f"Top {n}:" if shown == 0 else f"Top {n} adds:"
                ui.label(_listed(heading, names)).classes("text-xs whitespace-pre-wrap")
                shown = n

    def ranking_problem(self) -> str | None:
        if not self.picked:
            return PICK_FEATURE
        if not self.picked_samples:
            return PICK_SAMPLE
        return self.k_error

    async def rank(self) -> None:
        if self.adata is None:
            return
        self.wanted += 1
        if self.ranking:
            return
        self.ranking = True
        try:
            while True:
                asked = self.wanted
                problem, k = self.ranking_problem(), self.k
                if problem is not None or k is None:
                    self.show_ranking(None, problem or "")
                    return
                self.spinner.visible = True
                try:
                    scores = await run.io_bound(
                        rank_features, self.adata, list(self.picked), list(self.picked_samples), k
                    )
                except Exception as error:
                    if asked == self.wanted:
                        self.show_ranking(None, RANKING_FAILED.format(error=error_in_words(error)))
                        return
                    continue
                if asked == self.wanted:
                    self.show_ranking(scores, None)
                    return
        finally:
            self.ranking = False
            self.spinner.visible = False

    def show_ranking(self, scores: pd.Series | None, problem: str | None) -> None:
        if scores is None:
            self.order = []
            self.status.text = problem or ""
            self.ranking_table.rows = []
        else:
            self.order = [str(feature) for feature in scores.index]
            chosen = set(self.picked)
            places = sorted(self.order.index(feature) + 1 for feature in self.picked if feature in self.order)
            self.status.text = (
                f"{_count(self.cell_count(), 'cell')}. Picked features rank {', '.join(map(str, places))} "
                f"of {len(self.order)}."
            )
            self.ranking_table.rows = [
                {
                    "place": place,
                    "feature": self.options.get(str(feature), str(feature)),
                    "score": f"{float(score):.4f}",
                    "picked": "✓" if str(feature) in chosen else "",
                }
                for place, (feature, score) in enumerate(scores.items(), start=1)
            ]
        self.ranking_table.update()
        self.show_used()

    def refusal(self) -> str | None:
        if not self.picked:
            return PICK_FEATURE
        if not self.picked_samples:
            return PICK_SAMPLE
        if self.k_error is not None:
            return self.k_error
        for box in [self.top_n, *(box for method in Method for box in self.boxes[method])]:
            if box.refusal is not None:
                return box.refusal
        if self.seed_error is not None:
            return self.seed_error
        return None if self.ticked() else NONE_TICKED

    def save(self) -> None:
        refusal = self.refusal()
        if refusal is not None:
            ui.notify(refusal, type="negative")
            return
        setup = {
            "picked": list(self.picked),
            "samples": list(self.picked_samples),
            "ranking_k": self.k,
            "top_n": list(self.top_n.values),
            "methods": self.chosen_methods(),
            "skipped": [key for key in self.keys if key in self.unticked],
            "seed": self.seed,
            "values": "original",
        }
        try:
            self.exp = save_setup(self.exp, setup)
        except (OSError, ValueError) as error:
            ui.notify(_not_saved(error), type="negative")
            return
        self.unreadable_line.set_visibility(False)
        self.left_out_line.set_visibility(False)
        ticked = len(self.ticked())
        made = _count(ticked, "embedding")
        ui.notify(f"Saved the embedding set-up: {made}, {len(self.keys) - ticked} skipped.", type="positive")


def _setup_tab(exp: Experiment, adata: ad.AnnData | None) -> None:
    if adata is not None:
        features = describe(list(adata.var_names), read_info_file(exp))
        table = cast(pd.DataFrame, adata.obs)
    else:
        cells: Cells = load_cells(exp)
        empty_state(cells)
        if not cells.measured:
            return
        features, table = cells.features, cells.table
    _SetupTab(exp, adata, features, table).build()


@ui.page("/embedding")
def embedding_page() -> None:
    with page_frame("Embedding"):
        exp = STATE.experiment
        if exp is None:
            with ui.row().classes("items-center gap-1 text-sm"):
                ui.label("No experiment open.")
                ui.link("Open one on the Experiment page.", "/")
            return
        adata = _read(exp)
        made = adata is not None and bool(embedding_keys(adata))
        with ui.tabs().props("dense align=left no-caps") as tabs:
            view_tab = ui.tab(VIEW_TAB)
            setup_tab = ui.tab(SETUP_TAB)
        with ui.tab_panels(tabs, value=view_tab if made else setup_tab).classes("w-full bg-transparent"):
            with ui.tab_panel(view_tab):
                if adata is not None and made:
                    _view_tab(adata, exp)
                else:
                    _empty_state()
            with ui.tab_panel(setup_tab):
                _setup_tab(exp, adata)
