from __future__ import annotations

from nicegui import ui

from fouri.app.cells import BLANK, _condition
from fouri.app.records import clock_text
from fouri.app.shell import page_frame
from fouri.app.state import STATE
from fouri.config import Experiment, load_config
from fouri.run import STEP_NAMES
from fouri.status import PER_WELL_STEPS, OutputStatus, State, Step, WellStatus, experiment_status, step_status

SECTION_TITLE = "text-[1.05rem] font-semibold"
STATE_ICONS = {
    State.COMPLETE: ("check_circle", "#0F9E8E"),
    State.PARTIAL: ("timelapse", "#D97706"),
    State.NOT_STARTED: ("radio_button_unchecked", "#A8A3BC"),
    State.BLOCKED: ("error", "#DC2626"),
}
COUNTS_LINE = (
    "Counts are rounds done out of the rounds in info_exp.csv; alignment leaves out the anchor round. "
    "After a correction in Review, rerun Measure cells for that well with Overwrite on."
)

PerWell = dict[Step, dict[str, WellStatus]]
Whole = dict[Step, OutputStatus]


def _reason(error: Exception) -> str:
    if isinstance(error, OSError) and error.strerror:
        return f"{error.strerror}: {error.filename}" if error.filename else error.strerror
    return str(error).rstrip(".")


def _read(config_path: str) -> tuple[Experiment, PerWell, Whole]:
    exp = load_config(config_path, announce=False)
    per_well = {step: step_status(exp, step) for step in PER_WELL_STEPS}
    return exp, per_well, experiment_status(exp)


def _state_icon(state: State) -> None:
    icon, color = STATE_ICONS[state]
    ui.icon(icon).classes("text-lg").style(f"color: {color}").tooltip(state.value)


def _well_cell(status: WellStatus) -> None:
    with ui.row().classes("items-center gap-1 no-wrap"):
        _state_icon(status.state)
        text = status.note if status.state == State.BLOCKED else f"{status.done}/{status.total}"
        ui.label(text).classes("text-xs muted")


def _output_text(status: OutputStatus) -> str:
    if status.updated is None:
        return status.state.value
    return " · ".join([f"written {clock_text(status.updated)}", *([status.note] if status.note else [])])


def _wells_table(exp: Experiment, per_well: PerWell) -> None:
    wells = list(per_well[PER_WELL_STEPS[0]])
    with ui.grid(columns=len(PER_WELL_STEPS) + 2).classes("gap-x-8 gap-y-2 items-center mt-2"):
        for heading in ["Well", "Condition", *(STEP_NAMES[step] for step in PER_WELL_STEPS)]:
            ui.label(heading).classes("font-semibold text-sm")
        for well in wells:
            ui.label(well).classes("font-mono")
            ui.label(_condition(exp.well_metadata.get(well, {}), exp.condition_columns) or BLANK).classes(
                "text-sm muted"
            )
            for step in PER_WELL_STEPS:
                _well_cell(per_well[step][well])


def _whole_experiment(whole: Whole) -> None:
    ui.label("Whole experiment").classes(SECTION_TITLE + " mt-4")
    for step, status in whole.items():
        with ui.row().classes("items-center gap-2"):
            _state_icon(status.state)
            ui.label(STEP_NAMES[step])
            ui.label(_output_text(status)).classes("text-xs muted")


@ui.refreshable
def _content(config_path: str) -> None:
    with ui.card().classes("w-full p-5 gap-2"):
        with ui.row().classes("w-full items-center"):
            ui.label("Steps completed per well").classes(SECTION_TITLE)
            ui.space()
            ui.button("Refresh", icon="refresh", on_click=lambda: _content.refresh()).props("flat")
        ui.label(COUNTS_LINE).classes("text-sm muted")
        try:
            exp, per_well, whole = _read(config_path)
        except Exception as error:
            ui.label(f"Can't show the status: {_reason(error)}")
            return
        _wells_table(exp, per_well)
        _whole_experiment(whole)


@ui.page("/status")
def status_page() -> None:
    with page_frame("Status"):
        exp = STATE.experiment
        if exp is None:
            with ui.row().classes("items-center gap-1 text-sm"):
                ui.label("No experiment open.")
                ui.link("Open one on the Experiment page.", "/")
            return
        _content(str(exp.config_path))
