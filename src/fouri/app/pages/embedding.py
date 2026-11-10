"""The Embedding page."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import anndata as ad
import pandas as pd
import plotly.graph_objects as go
from nicegui import ui
from nicegui.events import ValueChangeEventArguments

from fouri.app.cells import CONDITION, add_groups
from fouri.app.embeddings import (
    LONGEST,
    RECORDS,
    SHORTEST,
    box_shape,
    data_proportions,
    embedding_keys,
    embedding_names,
    record_rows,
)
from fouri.app.names import describe, marker_menu
from fouri.app.shell import page_frame
from fouri.app.state import STATE
from fouri.app.theme import on_white_panel, restyle
from fouri.config import Experiment
from fouri.io import adata_sub_path, read_info_file
from fouri.plots import plot_embedding

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


def _read(exp: Experiment) -> ad.AnnData | None:
    path = adata_sub_path(exp)
    if not path.is_file():
        return None
    adata = ad.read_h5ad(path)
    return adata if embedding_keys(adata) else None


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
        ui.label("Run the Embedding step to compute PHATE or UMAP embeddings.").classes("text-sm muted text-center")
        ui.button("Go to Run", icon="play_circle", on_click=lambda: ui.navigate.to("/run"))


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
                    ui.label(value).classes("min-w-0 break-words")

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
        if adata is None:
            _empty_state()
            return
        with ui.tabs().props("dense align=left no-caps") as tabs:
            view_tab = ui.tab("View embeddings")
        with ui.tab_panels(tabs, value=view_tab).classes("w-full bg-transparent"), ui.tab_panel(view_tab):
            _view_tab(adata, exp)
