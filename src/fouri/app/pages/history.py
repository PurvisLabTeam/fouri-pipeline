from __future__ import annotations

import datetime as dt

from nicegui import ui

from fouri.app.records import RunRow, clock_text, read_runs
from fouri.app.shell import page_frame
from fouri.app.state import STATE, run_going

SECTION_TITLE = "text-[1.05rem] font-semibold"
COLUMNS = [
    {"name": name, "label": label, "field": name, "align": "left"}
    for name, label in [
        ("started", "Started"),
        ("steps", "Steps"),
        ("wells", "Wells"),
        ("result", "Result"),
        ("took", "Took"),
        ("notes", "Notes"),
    ]
]


def _took_text(seconds: float) -> str:
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    return f"{minutes} min {secs} s" if minutes else f"{secs} s"


def _table_row(key: int, row: RunRow) -> dict[str, object]:
    return {
        "key": key,
        "started": clock_text(dt.datetime.fromisoformat(row.started)),
        "steps": ", ".join(row.steps),
        "wells": ", ".join(row.wells),
        "result": row.result,
        "took": _took_text(row.seconds),
        "notes": "; ".join(row.notes),
    }


@ui.page("/history")
def history_page() -> None:
    with page_frame("History"):
        exp = STATE.experiment
        if exp is None:
            with ui.row().classes("items-center gap-1 text-sm"):
                ui.label("No experiment open.")
                ui.link("Open one on the Experiment page.", "/")
            return
        running = STATE.run.run_id if STATE.run is not None and run_going() else None
        runs = read_runs(exp, running=running)
        with ui.card().classes("w-full p-5 gap-2"):
            ui.label("Runs").classes(SECTION_TITLE)
            ui.label("Also saved in the results folder, with each run's settings and fouri version.").classes(
                "text-sm muted"
            )
            if not runs:
                ui.label("No runs recorded yet.")
                return
            rows = [_table_row(key, row) for key, row in enumerate(runs)]
            ui.table(columns=COLUMNS, rows=rows, row_key="key").props("flat wrap-cells").classes("w-full")
