"""The plate: every well position, the imaged wells in a color each, the rest dashed.

``plate`` knows nothing of what the colors mean, so the Experiment page can color wells by how
far they have got and the Run page by whether they are picked.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterable, Mapping
from string import ascii_uppercase
from typing import Any

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
SECTION_TITLE = "text-[1.05rem] font-semibold"
PICKED = "var(--c-primary)"
NOT_PICKED = "var(--c-line)"


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


def _text(value: Any) -> str:
    """Return a metadata value as text, empty when missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def conditions(exp: Experiment) -> dict[str, list[str]]:
    """Return each condition's wells, in plate order; a condition is its condition columns' values.

    Columns no well has a value in are left out of the names.
    """
    wells = sorted({normalize_well(well) for well in exp.well_list}, key=plate_sort_key)
    metadata = {normalize_well(well): values for well, values in exp.well_metadata.items()}
    columns = [
        column for column in exp.condition_columns if any(_text(metadata.get(w, {}).get(column)) for w in wells)
    ]
    groups: dict[str, list[str]] = {}
    for well in wells:
        values = [_text(metadata.get(well, {}).get(column)) for column in columns]
        name = " · ".join(value for value in values if value) or "No condition"
        groups.setdefault(name, []).append(well)
    return groups


def wells_card(exp: Experiment, picked: dict[str, bool]) -> None:
    groups = conditions(exp)
    hover = {well: f"{well}: {name}" for name, wells in groups.items() for well in wells}
    with ui.card().classes("w-full p-4 gap-2"):
        with ui.row().classes("items-center w-full"):
            ui.label("Wells").classes(SECTION_TITLE)
            ui.space()
            count = ui.label().classes("text-sm muted")

        @ui.refreshable
        def drawn() -> None:
            colors = {well: PICKED if on else NOT_PICKED for well, on in picked.items()}
            plate(exp, colors, hover=hover, on_click=toggle)
            count.text = f"{sum(picked.values())} of {len(picked)} wells chosen"

        def pick(wells: set[str]) -> None:
            for well in picked:
                picked[well] = well in wells
            drawn.refresh()

        def toggle(well: str) -> None:
            if well not in picked:  # not imaged
                return
            picked[well] = not picked[well]
            condition.set_value(None)
            drawn.refresh()

        def pick_all(on: bool) -> None:
            pick(set(picked) if on else set())
            condition.set_value(None)

        def pick_condition(name: str | None) -> None:
            if name is not None:
                pick(set(groups[name]))

        drawn()
        with ui.row().classes("items-center gap-2"):
            ui.button("All", on_click=lambda: pick_all(True)).props("outline dense")
            ui.button("None", on_click=lambda: pick_all(False)).props("outline dense")
            condition = ui.select(
                list(groups), label="Condition only", on_change=lambda event: pick_condition(event.value)
            ).classes("w-56")
        ui.label("Click a well to include or exclude it. Dashed wells were not imaged in this experiment.").classes(
            "text-xs muted"
        )
