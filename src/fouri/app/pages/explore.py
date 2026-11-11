"""The Explore page."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from nicegui import run, ui
from nicegui.events import ValueChangeEventArguments

from fouri.app.cells import CONDITION, Cells, load_cells
from fouri.app.figures import (
    BINNED_X,
    BINNED_Y,
    COLORED,
    COLORED_BY_VALUE,
    GROUPED,
    SAMPLED,
    SMALL_PLOTS,
    SPLIT,
    TWO_AXES,
    Choices,
    Kind,
    explore_figure,
)
from fouri.app.names import MEASURES, marker_menu
from fouri.app.rules import kept_cells
from fouri.app.shell import empty_state, page_frame
from fouri.app.state import STATE
from fouri.app.theme import on_white_panel, restyle
from fouri.io import read_info_file

GROUP, MARKER = "Group", "Marker"
NONE = "(none)"
HIDE_OUTLIERS = "Hide outliers (outside 0.5\N{EN DASH}99.5 %)"
NUCLEUS_TOTAL, NUCLEUS_MEAN = MEASURES["total_nuc_protein"], MEASURES["nuc_mean"]
SECTION_TITLE = "text-sm font-medium"
MODE_SECONDS = 0.25
TYPING_MS = 500
FEWEST_BINS, MOST_BINS, BINS_STEP = 5, 500, 5
FEWEST_POINTS, POINTS_STEP = 100, 1000
WHITE_PANEL = "background: #FFFFFF; border-radius: 12px"
NO_PANEL = "background: transparent"

Menu = dict[str, dict[str, str]]


@dataclass
class _View:
    kind: Kind
    x: str
    y: str
    color_mode: str
    color_group: str | None
    color_value: str
    group: str
    split: str | None
    well: str
    small_plots: bool = False
    log_x: bool = False
    log_y: bool = False
    hide_outliers: bool = True
    bins_x: int = 100
    bins_y: int = 100
    max_points: int = 20000
    every_cell: bool = False
    figure: go.Figure | None = None
    dark: bool = False
    drawing: int = 0

    def choices(self) -> Choices:
        by_value = self.kind in COLORED_BY_VALUE and self.color_mode == MARKER
        return Choices(
            kind=self.kind,
            x=self.x,
            y=self.y,
            color=self.color_value if by_value else self.color_group,
            group=self.group,
            split=self.split,
            small_plots=self.small_plots,
            log_x=self.log_x,
            log_y=self.log_y,
            hide_outliers=self.hide_outliers,
            bins_x=self.bins_x,
            bins_y=self.bins_y,
            max_points=self.max_points,
            well=self.well,
        )


def _column(menu: Menu, marker: str, measure: str) -> str:
    measures = menu[marker]
    return measures.get(measure) or next(iter(measures.values()))


def _place(menu: Menu, column: str) -> tuple[str, str]:
    for marker, measures in menu.items():
        for measure, name in measures.items():
            if name == column:
                return marker, measure
    raise KeyError(column)


def _opening(cells: Cells, menu: Menu) -> _View:
    markers = list(menu)
    second = markers[1] if len(markers) > 1 else markers[0]
    group = CONDITION if CONDITION in cells.groups else cells.groups[0]
    return _View(
        kind=Kind.HISTOGRAM,
        x=_column(menu, markers[0], NUCLEUS_TOTAL),
        y=_column(menu, second, NUCLEUS_MEAN),
        color_mode=GROUP,
        color_group=group,
        color_value=_column(menu, markers[0], NUCLEUS_MEAN),
        group=group,
        split=None,
        well=cells.measured[0],
    )


def _plot_config(kind: Kind) -> dict[str, Any]:
    name = "explore-" + kind.value.lower().replace(" ", "-")
    return {"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": name, "scale": 2}}


def _feature_picker(menu: Menu, label: str, column: str, on_pick: Callable[[str], Awaitable[None]]) -> ui.row:
    marker, measure = _place(menu, column)
    with ui.row().classes("w-full no-wrap gap-1") as row:
        marker_box = ui.select(list(menu), value=marker, label=label, with_input=True).classes("w-7/12")
        measure_box = ui.select(list(menu[marker]), value=measure, label="measure").classes("w-5/12")

    async def new_marker(e: ValueChangeEventArguments[Any]) -> None:
        if not e.value:
            return
        options = list(menu[e.value])
        keep = measure_box.value if measure_box.value in options else options[0]
        measure_box.set_options(options, value=keep)
        await on_pick(menu[e.value][keep])

    async def new_measure(e: ValueChangeEventArguments[Any]) -> None:
        if e.value and marker_box.value and e.value in menu[marker_box.value]:
            await on_pick(menu[marker_box.value][e.value])

    marker_box.on_value_change(new_marker)
    measure_box.on_value_change(new_measure)
    return row


def _explorer(cells: Cells, menu: Menu, kept: pd.Series | None, problem: str | None) -> None:
    view = _opening(cells, menu)

    async def draw() -> None:
        view.drawing += 1
        ticket = view.drawing
        spinner.set_visibility(True)
        try:
            shown = None if view.every_cell else kept
            fig, text = await run.io_bound(explore_figure, cells, view.choices(), shown)
        finally:
            if ticket == view.drawing:
                spinner.set_visibility(False)
        if ticket != view.drawing:
            return
        view.figure = fig
        line.set_text(text)
        show()

    def show() -> None:
        if view.figure is None:
            return
        fig = go.Figure(view.figure)
        white = on_white_panel(fig, dark=STATE.dark)
        restyle(fig, dark=STATE.dark)
        view.dark = STATE.dark
        plot.style(replace=WHITE_PANEL if white else NO_PANEL)
        plot.update_figure({**fig.to_plotly_json(), "config": _plot_config(view.kind)})

    async def change(name: str, value: Any) -> None:
        if getattr(view, name) == value:
            return
        setattr(view, name, value)
        show_controls()
        await draw()

    def on(name: str, convert: Callable[[Any], Any] = lambda value: value) -> Callable[..., Awaitable[None]]:
        async def handler(e: ValueChangeEventArguments[Any]) -> None:
            if e.value is not None:
                await change(name, convert(e.value))

        return handler

    def chosen(value: str) -> str | None:
        return None if value == NONE else value

    async def pick_x(column: str) -> None:
        await change("x", column)

    async def pick_y(column: str) -> None:
        await change("y", column)

    async def pick_color(column: str) -> None:
        await change("color_value", column)

    with ui.row().classes("w-full no-wrap items-start gap-6"):
        with ui.column().classes("w-80 shrink-0 gap-2"):
            ui.label("Plot").classes(SECTION_TITLE)
            ui.radio([kind.value for kind in Kind], value=view.kind.value, on_change=on("kind", Kind)).props("dense")

            _feature_picker(menu, "Marker (x)", view.x, pick_x)
            y_row = _feature_picker(menu, "Marker (y)", view.y, pick_y)

            with ui.column().classes("w-full gap-2") as color_box:
                ui.label("Color by").classes(f"{SECTION_TITLE} mt-2")
                mode_toggle = ui.toggle([MARKER, GROUP], value=view.color_mode, on_change=on("color_mode"))
                color_marker_row = _feature_picker(menu, "Marker", view.color_value, pick_color)
                color_group_box = ui.select(
                    [NONE, *cells.groups],
                    value=view.color_group or NONE,
                    label="Group",
                    on_change=on("color_group", chosen),
                ).classes("w-full")

            group_box = ui.select(cells.groups, value=view.group, label="Group by", on_change=on("group")).classes(
                "w-full mt-2"
            )
            split_box = ui.select(
                [NONE, *cells.groups], value=NONE, label="Split each group by", on_change=on("split", chosen)
            ).classes("w-full")
            small_plots_box = ui.checkbox("One plot per group", value=view.small_plots, on_change=on("small_plots"))
            well_box = ui.select(cells.measured, value=view.well, label="Well", on_change=on("well")).classes("w-full")

            ui.checkbox("log10 x", value=view.log_x, on_change=on("log_x"))
            log_y_box = ui.checkbox("log10 y", value=view.log_y, on_change=on("log_y"))
            ui.checkbox(HIDE_OUTLIERS, value=view.hide_outliers, on_change=on("hide_outliers"))
            ui.switch("Show all cells", value=view.every_cell, on_change=on("every_cell"))
            with ui.row().classes("w-full no-wrap gap-2"):
                bins_x_box = (
                    ui.number(
                        "Bins (x)",
                        value=view.bins_x,
                        min=FEWEST_BINS,
                        max=MOST_BINS,
                        step=BINS_STEP,
                        format="%d",
                        on_change=on("bins_x", int),
                    )
                    .props(f"debounce={TYPING_MS}")
                    .classes("w-1/2")
                )
                bins_y_box = (
                    ui.number(
                        "Bins (y)",
                        value=view.bins_y,
                        min=FEWEST_BINS,
                        max=MOST_BINS,
                        step=BINS_STEP,
                        format="%d",
                        on_change=on("bins_y", int),
                    )
                    .props(f"debounce={TYPING_MS}")
                    .classes("w-1/2")
                )
            max_points_box = (
                ui.number(
                    "Max points shown",
                    value=view.max_points,
                    min=FEWEST_POINTS,
                    step=POINTS_STEP,
                    format="%d",
                    on_change=on("max_points", int),
                )
                .props(f"debounce={TYPING_MS}")
                .classes("w-full")
            )

        with ui.column().classes("grow min-w-0 gap-1"):
            if problem:
                ui.label(problem).classes("text-sm text-warning")
            plot = ui.plotly({"data": [], "layout": {}}).classes("w-full h-[80vh]")
            with ui.row().classes("items-center gap-2"):
                spinner = ui.spinner(size="sm")
                line = ui.label().classes("text-sm muted")

    def show_controls() -> None:
        kind = view.kind
        by_value = kind in COLORED_BY_VALUE and view.color_mode == MARKER
        y_row.set_visibility(kind in TWO_AXES)
        color_box.set_visibility(kind in COLORED)
        mode_toggle.set_visibility(kind in COLORED_BY_VALUE)
        color_marker_row.set_visibility(by_value)
        color_group_box.set_visibility(not by_value)
        group_box.set_visibility(kind in GROUPED)
        split_box.set_visibility(kind in SPLIT)
        small_plots_box.set_visibility(kind in SMALL_PLOTS)
        well_box.set_visibility(kind is Kind.WELL_MAP)
        log_y_box.set_visibility(kind in TWO_AXES)
        bins_x_box.set_visibility(kind in BINNED_X)
        bins_y_box.set_visibility(kind in BINNED_Y)
        max_points_box.set_visibility(kind in SAMPLED)

    def follow_mode() -> None:
        if view.figure is not None and view.dark != STATE.dark:
            show()

    ui.timer(MODE_SECONDS, follow_mode)
    show_controls()
    ui.timer(0, draw, once=True)


@ui.page("/explore")
def explore_page() -> None:
    with page_frame("Explore"):
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
        kept, problem = kept_cells(cells, exp)
        _explorer(cells, marker_menu(cells.features, read_info_file(exp)), kept, problem)
