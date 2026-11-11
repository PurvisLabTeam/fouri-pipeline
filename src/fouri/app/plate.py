"""The plate: every well position, the imaged wells in a color each, the rest dashed.

``plate`` knows nothing of what the colors mean, so the Experiment page can color wells by how
far they have got and the Run page by whether they are picked.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from string import ascii_uppercase

from nicegui import ui

from fouri.config import Experiment
from fouri.metadata import normalize_well, plate_sort_key
from fouri.status import PER_WELL_STEPS, State, step_status

# the smallest plate drawn is a 96-well plate, A to H and 1 to 12
SMALLEST_ROWS = 8
SMALLEST_COLUMNS = 12
DASHED = "border: 1px dashed var(--c-line)"
OUTLINED = "border: 2px solid var(--c-muted)"
PALER = "opacity: 0.3"


def plate_grid(wells: Iterable[str]) -> tuple[list[str], list[int]]:
    """Return the plate's row letters and column numbers: at least A to H and 1 to 12."""
    wells = [normalize_well(well) for well in wells]
    last_row = max((ascii_uppercase.index(well[0]) + 1 for well in wells), default=0)
    last_column = max((int(well[1:]) for well in wells), default=0)
    rows = list(ascii_uppercase[: max(SMALLEST_ROWS, last_row)])
    columns = list(range(1, max(SMALLEST_COLUMNS, last_column) + 1))
    return rows, columns


def _overall(states: list[State]) -> State:
    if all(state == State.COMPLETE for state in states):
        return State.COMPLETE
    if all(state == State.NOT_STARTED for state in states):
        return State.NOT_STARTED
    return State.PARTIAL


def overall_state(exp: Experiment) -> dict[str, State]:
    """Return each well's state across the four per-well steps, in plate order.

    A well is complete when every step is complete for it, not started when no step has started,
    and partial otherwise, a blocked step included.
    """
    per_step = [step_status(exp, step) for step in PER_WELL_STEPS]
    wells = sorted(per_step[0], key=plate_sort_key)
    return {well: _overall([status[well].state for status in per_step]) for well in wells}


def _click(on_click: Callable[[str], None], well: str) -> Callable[[], None]:
    return lambda: on_click(well)


def plate(
    exp: Experiment,
    colors: Mapping[str, str],
    *,
    hover: Mapping[str, str] | None = None,
    on_click: Callable[[str], None] | None = None,
    outlined: Collection[str] = (),
    selected: Collection[str] | None = None,
) -> None:
    """Draw the plate: imaged wells in their colors, the rest dashed; a click passes the well.

    ``hover`` gives the text shown when the pointer rests on an imaged well. ``on_click`` gets the
    name of any well clicked, imaged or not.
    """
    fills = {normalize_well(well): color for well, color in colors.items()}
    edged = {normalize_well(well) for well in outlined} - fills.keys()
    chosen = None if selected is None else {normalize_well(well) for well in selected}
    texts = {normalize_well(well): text for well, text in (hover or {}).items()}
    rows, columns = plate_grid([*exp.well_list, *fills, *edged])
    with ui.grid(columns=len(columns) + 1).classes("gap-1"):
        ui.label("")
        for column in columns:
            ui.label(str(column)).classes("text-xs text-center muted")
        for row in rows:
            ui.label(row).classes("text-xs muted")
            for column in columns:
                well = f"{row}{column:02d}"
                cell = ui.element("div").classes("w-7 h-7 rounded-md")
                if well in fills:
                    cell.style(f"background: {fills[well]}")
                    if chosen is not None and well not in chosen:
                        cell.style(PALER)
                elif well in edged:
                    cell.style(OUTLINED)
                else:
                    cell.style(DASHED)
                if well in texts:
                    cell.tooltip(texts[well])
                if on_click is not None:
                    cell.classes("cursor-pointer").on("click", _click(on_click, well))
