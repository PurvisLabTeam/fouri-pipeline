"""The Review in napari page."""

from __future__ import annotations

from dataclasses import dataclass

from nicegui import background_tasks, ui

from fouri.app.editors import EDITORS, READY, editor_command, run_editor
from fouri.app.shell import page_frame
from fouri.app.state import STATE
from fouri.metadata import plate_sort_key

SECTION_TITLE = "text-[1.05rem] font-semibold"
INTRO = (
    "A run never waits for hand work: run the steps, review here afterwards, "
    "then rerun Measure cells with Overwrite on for the wells you changed."
)
NO_WELLS = "No well to pick: the experiment has no wells. Add them on the Experiment page."


@dataclass
class EditorCard:
    running: bool = False
    problem: str = ""


CARDS = {editor: EditorCard() for editor in EDITORS}


async def _run(editor: str, config: str, well: str) -> None:
    card = CARDS[editor]
    try:
        code, last = await run_editor(editor_command(editor, config, well))
    except OSError as error:
        card.problem = f"The editor could not start: {error}"
    else:
        if code != 0:
            card.problem = last or f"The editor stopped unexpectedly (exit code {code})."
    finally:
        card.running = False


def _open(editor: str, config: str, well: str | None) -> None:
    card = CARDS[editor]
    if card.running or not well:
        return
    card.running, card.problem = True, ""
    background_tasks.create(_run(editor, config, well), name=f"{editor} editor")


def _card(editor: str, config: str, wells: list[str]) -> None:
    title, description = EDITORS[editor]
    card = CARDS[editor]
    with ui.card().classes("w-full p-5 gap-2"):
        ui.label(title).classes(SECTION_TITLE)
        ui.label(description).classes("text-sm muted")
        if editor not in READY:
            ui.label("Not available yet.").classes("text-sm muted")
            return
        if not wells:
            ui.label(NO_WELLS).classes("text-sm muted")
            return
        with ui.row().classes("items-center gap-3"):
            picker = ui.select(wells, value=wells[0], label="Well").classes("w-32")
            ui.button(
                "Open", icon="open_in_new", on_click=lambda: _open(editor, config, picker.value)
            ).bind_enabled_from(card, "running", backward=lambda running: not running)
        ui.label().classes("text-sm text-negative break-words").bind_text_from(card, "problem").bind_visibility_from(
            card, "problem", backward=bool
        )


@ui.page("/review")
def review_page() -> None:
    with page_frame("Review in napari"):
        ui.label("Review in napari").classes("page-title")
        exp = STATE.experiment
        if exp is None:
            with ui.row().classes("items-center gap-1 text-sm"):
                ui.label("No experiment open.")
                ui.link("Open one on the Experiment page.", "/")
            return
        ui.label(INTRO).classes("text-sm muted")
        wells = sorted(exp.well_list, key=plate_sort_key)
        for editor in EDITORS:
            _card(editor, str(exp.config_path), wells)
