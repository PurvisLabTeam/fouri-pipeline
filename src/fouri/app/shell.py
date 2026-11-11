"""The frame every page is built in: the header and the list of pages on the left."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import ui

from fouri.app.runner import Run
from fouri.app.state import STATE
from fouri.app.theme import apply_theme
from fouri.config import Experiment
from fouri.run import STEP_NAMES

if TYPE_CHECKING:
    from fouri.app.cells import Cells

# address, name in the list, Material icon
PAGES = [
    ("/", "Experiment", "science"),
    ("/run", "Run", "play_circle"),
    ("/status", "Status", "grid_view"),
    ("/review", "Review in napari", "visibility"),
    ("/explore", "Explore", "scatter_plot"),
    ("/filters", "Filters", "filter_alt"),
    ("/labels", "Labels & gates", "label"),
    ("/batch", "Batch effects", "compare_arrows"),
    ("/embedding", "Embedding", "bubble_chart"),
    ("/export", "Export", "download"),
    ("/models", "Models", "model_training"),
    ("/history", "History", "history"),
]


def _experiment_name(experiment: Experiment | None) -> str:
    if experiment is None:
        return "No experiment open"
    return experiment.name or experiment.key or Path(experiment.config_path).parent.name


def _mode_icon(dark: bool) -> str:
    return "light_mode" if dark else "dark_mode"


def running_line(run: Run | None) -> str:
    """Return "Step 2 of 4: Segment nuclei" for the step running; nothing when no run is going."""
    if run is None or run.finished:
        return ""
    for position, progress in enumerate(run.progress, start=1):
        if progress.outcome is None:
            return f"Step {position} of {len(run.progress)}: {STEP_NAMES[progress.step]}"
    return ""


@contextmanager
def page_frame(title: str) -> Generator[None]:
    """Build the header and the page list, and yield inside the page's content column.

    ``title`` is the page's name in the list, which is shown as the current page.
    """
    apply_theme()
    # each page opens in the mode chosen last, and a change is kept for the pages opened after it
    dark_mode = ui.dark_mode().bind_value(STATE, "dark")
    with ui.header().classes("items-center px-5 py-2 gap-3"):
        ui.html('<div class="brand">4i</div>', sanitize=False)
        with ui.column().classes("gap-0"):
            ui.label("4i pipeline").classes("font-semibold leading-tight")
            ui.label().classes("text-xs muted leading-tight").bind_text_from(
                STATE, "experiment", backward=_experiment_name
            )
        ui.space()
        ui.label().classes("text-sm font-medium").style("color: var(--c-primary)").bind_text_from(
            STATE, "run", backward=running_line
        )
        ui.button(on_click=dark_mode.toggle).props("flat round").bind_icon_from(
            STATE, "dark", backward=_mode_icon
        ).tooltip("Light / dark")
    with ui.left_drawer(value=True).props("width=230").classes("px-3 py-4"):
        for path, name, icon in PAGES:
            with ui.link(target=path).classes("no-underline w-full"):
                active = " active" if name == title else ""
                with ui.row().classes("nav-item items-center gap-3 px-3 py-2 w-full" + active):
                    ui.icon(icon).classes("text-lg")
                    ui.label(name)
    with ui.column().classes("w-full p-6 gap-4"):
        yield


def _missing_line(cells: Cells) -> str:
    """Return the line a cell page shows above its content; nothing when every well is measured."""
    if not cells.measured or not cells.missing:
        return ""
    wells = len(cells.measured) + len(cells.missing)
    return f"Showing {len(cells.measured)} of {wells} wells. Not measured yet: {', '.join(cells.missing)}."


def empty_state(cells: Cells) -> None:
    """Show what is missing above a cell page: no well measured, or the wells not measured yet."""
    if not cells.measured:
        with ui.card().classes("p-8 items-center gap-3 max-w-xl mx-auto mt-10"):
            ui.icon("hourglass_empty").classes("text-5xl muted")
            ui.label("No cells measured yet").classes("text-[1.05rem] font-semibold")
            ui.label("Run Measure cells on at least one well to plot cells here.").classes("text-sm muted text-center")
            ui.button("Go to Run", icon="play_circle", on_click=lambda: ui.navigate.to("/run"))
        return
    line = _missing_line(cells)
    if line:
        with ui.row().classes("items-center gap-2 text-sm muted"):
            ui.icon("info")
            ui.label(line)
