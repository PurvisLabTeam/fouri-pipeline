"""The Run page: pick the steps, the wells and each step's settings, run them, and follow the run.

The page never runs a step: it starts a ``Run``, whose steps are programs of their own, and shows
what they report. The run's progress is read by the app's own timer, so leaving the page leaves
the run going. With no GPU found, Run asks before Segment nuclei runs on the CPU.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from typing import Any

from nicegui import ui

from fouri.app.gpu import cpu_minutes, images_to_segment
from fouri.app.plate import plate
from fouri.app.runner import Bar, Run, StepProgress, StepRequest
from fouri.app.shell import page_frame, running_line
from fouri.app.state import POLL_SECONDS, STATE, run_going, stop_run
from fouri.app.steps import DESCRIPTIONS, Setting, can_tick, default, settings_for, shown_settings, whole_number
from fouri.config import Experiment
from fouri.metadata import normalize_well, plate_sort_key
from fouri.run import STEP_NAMES
from fouri.status import PER_WELL_STEPS, Step

SECTION_TITLE = "text-[1.05rem] font-semibold"
PICKED = "var(--c-primary)"
NOT_PICKED = "var(--c-line)"
BADGE_STYLE = "background: var(--c-primary-soft); color: var(--c-primary)"
WORKER_CHOICES = {0: "Automatic", 1: "1", 2: "2", 4: "4", 8: "8", 12: "12"}  # 0 is Automatic
MAX_BARS = 3  # the step's own bars shown under the step bar
IDLE = "Nothing running."
# the step bar's colour and the line above it once a run has ended
ENDINGS = {
    "done": ("positive", "Finished: all wells complete."),
    "failed": (
        "negative",
        "Finished with errors (red lines in the log). Wells already done are skipped on the next run.",
    ),
    "stopped": ("warning", "Stopped. Completed wells are kept; run again to continue."),
}
FAILURE_LINE = "text-negative font-semibold"
GPU_LINE_SECONDS = 0.5  # how often the GPU line looks for the check's answer
CHECKING_GPU = "Checking for a GPU…"
NO_GPU = "No GPU found"
STILL_CHECKING = "Still checking for a GPU. Try again in a moment."
ON_CPU_LINE = "Segment nuclei runs on the CPU: no GPU found."


# -- what the page shows -------------------------------------------------------------------------


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


def open_bars(progress: StepProgress) -> list[Bar]:
    """Return the step's open bars from the outermost in, each the latest opened inside the last."""
    still_open = {bar.bar: bar for bar in progress.bars.values() if not bar.closed}
    chain: list[Bar] = []
    inside: int | None = None
    while len(chain) < MAX_BARS:
        if chain:
            candidates = [bar for bar in still_open.values() if bar.parent == inside]
        else:
            candidates = [bar for bar in still_open.values() if bar.parent is None or bar.parent not in still_open]
        if not candidates:
            break
        latest = max(candidates, key=lambda bar: bar.bar)
        chain.append(latest)
        inside = latest.bar
    return chain


def _share(bar: Bar) -> float:
    return min(max(bar.done / bar.total, 0.0), 1.0) if bar.total else 0.0


def step_share(progress: StepProgress) -> float:
    """Return how far a step has got by its latest outermost bar, which keeps its share once closed.

    A stopped step closes that bar while its last items finish, so the step bar holds its place.
    """
    outermost = [bar for bar in progress.bars.values() if bar.parent is None]
    return _share(max(outermost, key=lambda bar: bar.bar)) if outermost else 0.0


def _amount(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else f"{number:.1f}"


def duration(minutes: int) -> str:
    """Return a time in hours and minutes, the minutes left out when there are none.

    About 20 minutes, about 2 hours 20 minutes, about 6 hours.
    """
    hours, rest = divmod(minutes, 60)
    parts = [f"{hours} hour{'' if hours == 1 else 's'}"] if hours else []
    if rest or not hours:
        parts.append(f"{rest} minute{'' if rest == 1 else 's'}")
    return "about " + " ".join(parts)


def segments_on_cpu(run: Run) -> bool:
    """Return whether the run's Segment nuclei was told to run on the CPU."""
    return any(request.step == Step.SEGMENT and request.settings.get("allow_cpu") is True for request in run.requests)


def ending(run: Run) -> str:
    """Return how an ended run went: done, failed (a step failed) or stopped (stopped early)."""
    outcomes = [progress.outcome for progress in run.progress]
    if all(outcome == "done" for outcome in outcomes):
        return "done"
    return "failed" if "failed" in outcomes else "stopped"


# -- the page ------------------------------------------------------------------------------------


def _wells_card(exp: Experiment, picked: dict[str, bool]) -> None:
    groups = conditions(exp)
    hover = {well: f"{well}: {name}" for name, wells in groups.items() for well in wells}
    with ui.card().classes("w-full p-4 gap-2"):
        with ui.row().classes("items-center w-full"):
            ui.label("Wells").classes(SECTION_TITLE)
            ui.space()
            count = ui.label().classes("text-sm muted")

        @ui.refreshable
        def drawn() -> None:
            colours = {well: PICKED if on else NOT_PICKED for well, on in picked.items()}
            plate(exp, colours, hover=hover, on_click=toggle)
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


def _control(step: Step, setting: Setting, values: dict[str, Any]) -> None:
    if setting.control == "number":
        whole = whole_number(step, setting.name)
        box = ui.number(setting.label, precision=0 if whole else None)
        if whole:
            box.props("step=1")
        box.bind_value(values, setting.name).classes("w-56")
    elif setting.control == "switch":
        ui.switch(setting.label).bind_value(values, setting.name)
    else:  # text; no row is a choice yet
        ui.input(setting.label).bind_value(values, setting.name).classes("w-56")


def _gpu_line() -> None:
    """Show the GPU line under Segment nuclei: the GPU's name, or that none was found.

    Until the check answers, the line says it is checking.
    """
    with ui.row().classes("items-center gap-1 no-wrap mt-1"):
        icon = ui.icon("hourglass_empty").classes("text-base muted")
        label = ui.label(CHECKING_GPU).classes("text-xs muted")

    def show() -> None:
        if STATE.checking_gpu:
            return
        if STATE.gpu:
            icon.name = "memory"
            icon.classes(replace="text-base text-positive")
            label.text = f"GPU: {STATE.gpu}"
            label.classes(replace="text-xs")
        else:
            icon.name = "warning"
            icon.classes(replace="text-base text-warning")
            label.text = NO_GPU
            label.classes(replace="text-xs text-warning")
        timer.cancel()  # the GPU is checked once, so the line doesn't change again

    timer = ui.timer(GPU_LINE_SECONDS, show)


async def _ask_cpu(images: int) -> bool:
    """Ask before Segment nuclei runs on the CPU; return whether to go ahead."""
    count = f"{images} image{'' if images == 1 else 's'}"
    with ui.dialog() as dialog, ui.card().classes("p-5 gap-3 max-w-lg"):
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.icon("warning").classes("text-2xl text-warning")
            ui.label("No GPU found").classes(SECTION_TITLE)
        ui.label(f"Segment nuclei would segment {count} on the CPU, {duration(cpu_minutes(images))}.").classes(
            "text-sm"
        )
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
            ui.button("Run on the CPU", icon="play_arrow", on_click=lambda: dialog.submit(True))
    answer = await dialog  # None when closed with Escape or a click beside it, as Cancel
    dialog.delete()
    return answer is True


def _step_card(
    badge: int, step: Step, ticked: dict[Step, bool], overwrite: dict[Step, bool], values: dict[str, Any]
) -> None:
    with ui.card().classes("w-full p-3 gap-1"):
        with ui.row().classes("items-center no-wrap w-full gap-3"):
            ui.label(str(badge)).classes(
                "w-[26px] h-[26px] rounded-full grid place-items-center text-xs font-semibold shrink-0"
            ).style(BADGE_STYLE)
            with ui.column().classes("gap-0 grow"):
                box = ui.checkbox(STEP_NAMES[step]).bind_value(ticked, step).classes("font-medium -ml-2")
                if not can_tick(step):
                    box.disable()
                ui.label(DESCRIPTIONS[step]).classes("text-xs muted -mt-1")
                if step == Step.SEGMENT:
                    _gpu_line()
            if step in PER_WELL_STEPS:
                ui.switch("Overwrite").bind_value(overwrite, step).props("dense")
        settings = shown_settings(step)
        if settings:
            with ui.expansion("Settings").classes("w-full text-sm").props("dense"), ui.column().classes("gap-2 py-1"):
                for setting in settings:
                    _control(step, setting, values)


def _colour(bar: ui.linear_progress, name: str) -> None:
    if bar.props.get("color") != name:
        bar.props(f"color={name}")


def _progress_card(start: Callable[[], Awaitable[None]]) -> None:
    with ui.card().classes("grow p-5 gap-2"):
        ui.label("Progress").classes(SECTION_TITLE)
        step_line = ui.label(IDLE).classes("font-medium")
        # instant-feedback: bars jump to a new value, so a new run starts empty instead of sliding
        step_bar = ui.linear_progress(value=0, show_value=False).props("size=10px color=primary instant-feedback")
        rows = []
        for depth in range(MAX_BARS):
            with ui.row().classes("w-full items-center no-wrap gap-3") as row:
                label = ui.label().classes("text-sm muted w-44 shrink-0 truncate")
                bar = ui.linear_progress(value=0, show_value=False)
                bar.props(f"size={8 - 2 * depth}px color=primary instant-feedback").classes("grow")
                count = ui.label().classes("text-xs muted w-20 text-right")
            row.visible = False
            rows.append((row, label, bar, count))
        log = ui.log(max_lines=2000).classes("w-full h-96 text-xs")
        with ui.row().classes("mt-1 gap-2"):
            ui.button("Run", icon="play_arrow", on_click=start).bind_enabled_from(
                STATE, "run", backward=lambda run: run is None or run.finished
            )
            ui.button("Stop", icon="stop", color="negative", on_click=stop_run).props("outline").bind_enabled_from(
                STATE, "run", backward=lambda run: run is not None and not run.finished
            )
        shown: dict[str, Any] = {"run": None, "messages": [], "errors": set()}

        def show_log(run: Run) -> None:
            """Add the messages written since the last look, and each step's error in red."""
            for index, progress in enumerate(run.progress):
                for text in progress.messages[shown["messages"][index] :]:
                    log.push(text)
                shown["messages"][index] = len(progress.messages)
                if progress.outcome is not None and progress.error and index not in shown["errors"]:
                    log.push(progress.error, classes=FAILURE_LINE)
                    shown["errors"].add(index)

        def show_bars(bars: list[Bar]) -> None:
            for depth, (row, label, bar, count) in enumerate(rows):
                row.visible = depth < len(bars)
                if row.visible:
                    reported = bars[depth]
                    label.text = reported.desc
                    bar.value = _share(reported)
                    total = f" / {_amount(reported.total)}" if reported.total else ""
                    count.text = _amount(reported.done) + total

        def refresh() -> None:
            run = STATE.run
            if run is not shown["run"]:  # a new run: the log starts afresh
                log.clear()
                shown.update(run=run, messages=[0] * len(run.progress) if run else [], errors=set())
                if run is not None and segments_on_cpu(run):
                    log.push(ON_CPU_LINE)
            if run is None:
                return
            show_log(run)
            if not run.finished:
                position = next(index for index, p in enumerate(run.progress) if p.outcome is None)
                bars = open_bars(run.progress[position])
                step_line.text = running_line(run)
                step_bar.value = (position + step_share(run.progress[position])) / len(run.progress)
                _colour(step_bar, "primary")
                show_bars(bars)
                return
            how = ending(run)
            colour, text = ENDINGS[how]
            failed = [well for progress in run.progress for well in progress.failed]
            if how == "failed" and failed:
                text += f" Failed: {', '.join(failed)}."
            step_line.text = text
            step_bar.value = 1
            _colour(step_bar, colour)
            show_bars([])

        refresh()
        ui.timer(POLL_SECONDS, refresh)


def _content(exp: Experiment) -> None:
    wells = sorted({normalize_well(well) for well in exp.well_list}, key=plate_sort_key)
    picked = dict.fromkeys(wells, True)
    ticked = dict.fromkeys(Step, False)
    overwrite = dict.fromkeys(Step, False)
    values = {step: {s.name: default(step, s.name) for s in shown_settings(step)} for step in Step}
    workers = {"count": 0}

    async def start() -> None:
        if run_going():
            return
        chosen = [step for step in Step if ticked[step] and can_tick(step)]
        if not chosen:
            ui.notify("Tick at least one step.", type="warning")
            return
        wells_picked = [well for well, on in picked.items() if on]
        if not wells_picked and any(step in PER_WELL_STEPS for step in chosen):
            ui.notify("Pick at least one well.", type="warning")
            return
        on_cpu = False
        if Step.SEGMENT in chosen:
            if STATE.checking_gpu:
                ui.notify(STILL_CHECKING, type="warning")
                return
            if STATE.gpu is None:
                try:
                    images = images_to_segment(exp, wells_picked, overwrite=overwrite[Step.SEGMENT])
                except (OSError, ValueError) as error:
                    ui.notify(f"The run could not start: {error}", type="negative")
                    return
                if images > 0:
                    if not await _ask_cpu(images) or run_going():
                        return
                    on_cpu = True
        requests = [
            StepRequest(
                step,
                wells_picked if step in PER_WELL_STEPS else None,
                settings_for(step, values[step], overwrite=overwrite[step], workers=workers["count"] or None),
            )
            for step in chosen
        ]
        if on_cpu:  # this run only: the next run's settings start without it
            next(request for request in requests if request.step == Step.SEGMENT).settings["allow_cpu"] = True
        run = Run(exp.config_path, requests)
        try:
            run.start()
        except OSError as error:
            ui.notify(f"The run could not start: {error}", type="negative")
            return
        STATE.run = run

    with ui.row().classes("w-full gap-6 items-start no-wrap"):
        with ui.column().classes("w-[560px] shrink-0 gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("Steps").classes(SECTION_TITLE)
                ui.label("Wells already done are skipped unless Overwrite is on.").classes("text-sm muted")
            _wells_card(exp, picked)
            ui.select(WORKER_CHOICES, label="Wells processed in parallel").bind_value(workers, "count").classes("w-80")
            for badge, step in enumerate(Step, start=1):
                _step_card(badge, step, ticked, overwrite, values[step])
        _progress_card(start)


@ui.page("/run")
def run_page() -> None:
    with page_frame("Run"):
        exp = STATE.experiment
        if exp is None:
            with ui.row().classes("items-center gap-1 text-sm"):
                ui.label("No experiment open.")
                ui.link("Open one on the Experiment page.", "/")
            return
        _content(exp)
