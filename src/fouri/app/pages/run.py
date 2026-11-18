"""The Run page: pick the steps, the wells and each step's settings, run them, and follow the run.

The page never runs a step: it starts a ``Run``, whose steps are programs of their own, and shows
what they report. The run's progress is read by the app's own timer, so leaving the page leaves
the run going. With no GPU found, Run asks before Segment nuclei runs on the CPU.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from nicegui import ui
from tqdm import tqdm

from fouri.app.batches import batch_setup_line, saved_batch_setup
from fouri.app.embeddings import embedding_settings
from fouri.app.gpu import images_to_segment
from fouri.app.plate import wells_card
from fouri.app.rules import switched_on_filters, switched_on_labels
from fouri.app.runner import Bar, Run, StepProgress, StepRequest, time_left
from fouri.app.shell import page_frame, running_line
from fouri.app.state import POLL_SECONDS, STATE, run_going, stop_run
from fouri.app.steps import (
    DESCRIPTIONS,
    RUN_PAGE_STEPS,
    Setting,
    can_tick,
    default,
    models_left_out,
    settings_for,
    shown_settings,
    whole_number,
)
from fouri.config import Experiment, load_config
from fouri.metadata import normalize_well, plate_sort_key
from fouri.run import STEP_NAMES
from fouri.status import PER_WELL_STEPS, Step

SECTION_TITLE = "text-[1.05rem] font-semibold"
BADGE_STYLE = "background: var(--c-primary-soft); color: var(--c-primary)"
WORKER_CHOICES = {0: "Automatic", 1: "1", 2: "2", 4: "4", 8: "8", 12: "12"}  # 0 is Automatic
MAX_BARS = 3  # the step's own bars shown under the step bar
IDLE = "Nothing running."
# the step bar's color and the line above it once a run has ended
ENDINGS = {
    "done": ("positive", "Finished: all wells complete."),
    "failed": (
        "negative",
        "Finished with errors (red lines in the log). Wells already done are skipped on the next run.",
    ),
    "stopped": ("warning", "Stopped. Completed wells are kept; run again to continue."),
}
LOG_LINE = "whitespace-pre-wrap wrap-anywhere"
FAILURE_LINE = f"{LOG_LINE} text-negative font-semibold"
GPU_LINE_SECONDS = 0.5  # how often the GPU line looks for the check's answer
CHECKING_GPU = "Checking for a GPU…"
NO_GPU = "No GPU found"
STILL_CHECKING = "Still checking for a GPU. Try again in a moment."
ON_CPU_LINE = "Segment nuclei runs on the CPU: no GPU found."
NO_FILTERS = "No filters switched on."
FILTERS_UNREADABLE = "The saved filters can't be read. Fix them on the Filters page."
NO_LABELS = "No labels switched on."
LABELS_UNREADABLE = "The saved labels can't be read. Fix them on the Labels & gates page."
NO_EMBEDDING_SETUP = "Set up the embeddings on the Embedding page first."
EMBEDDING_UNREADABLE = "The embedding set-up can't be read. Save it again on the Embedding page."
ALL_SKIPPED = "Every embedding set up is skipped. Tick some on the Embedding page."
NO_BATCH_SETUP = "Set up batch correction on the Batch effects page first."
BATCH_UNREADABLE = "The Batch effects set-up can't be read. Save it again on the Batch effects page."
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")  # not %a, which follows the PC's language


# -- what the page shows -------------------------------------------------------------------------


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


def _step_bar(progress: StepProgress) -> Bar | None:
    """Return the step's latest outermost bar, the one for the whole step, kept once closed."""
    outermost = [bar for bar in progress.bars.values() if bar.parent is None]
    return max(outermost, key=lambda bar: bar.bar) if outermost else None


def step_share(progress: StepProgress) -> float:
    """Return how far a step has got by its latest outermost bar, which keeps its share once closed.

    A stopped step closes that bar while its last items finish, so the step bar holds its place.
    """
    bar = _step_bar(progress)
    return _share(bar) if bar is not None else 0.0


def step_time_left(progress: StepProgress, now: float, images: int | None = None) -> float | None:
    """Return the step's seconds left, or None before a rate exists.

    By the step's outermost bar, or, given Segment nuclei's count of images to segment, by image:
    each well's bar inside it moves once per image, and the step's bar opening starts the clock.
    """
    bar = _step_bar(progress)
    if bar is None:
        return None
    if images is None:
        return time_left(bar, now)
    inside = [inner for inner in progress.bars.values() if inner.parent == bar.bar]
    latest = max((inner.updated for inner in inside if inner.updated is not None), default=None)
    by_image = replace(bar, total=float(images), done=sum(inner.done for inner in inside), updated=latest)
    return time_left(by_image, now)


def _amount(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else f"{number:.1f}"


def time_left_text(seconds: float, now: float) -> str:
    """Return a bar's time left as tqdm writes it and the clock time it should be done.

    02:45 left, done about 2:32 PM; 9:41:10 left, done about Tue 7:05 AM, the weekday shown when it
    falls on another day.
    """
    done_at = dt.datetime.fromtimestamp(now + seconds)
    clock = f"{done_at.hour % 12 or 12}:{done_at.minute:02d} {'AM' if done_at.hour < 12 else 'PM'}"
    if done_at.date() != dt.datetime.fromtimestamp(now).date():
        clock = f"{WEEKDAYS[done_at.weekday()]} {clock}"
    return f"{tqdm.format_interval(seconds)} left, done about {clock}"


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


def _control(step: Step, setting: Setting, values: dict[str, Any]) -> None:
    if setting.control == "number":
        whole = whole_number(step, setting.name)
        box = ui.number(setting.label, precision=0 if whole else None)
        if whole:
            box.props("step=1")
        box.bind_value(values, setting.name).classes("w-56")
    elif setting.control == "switch":
        ui.switch(setting.label).bind_value(values, setting.name)
    elif setting.control == "choice" and setting.choices is not None:
        ui.select(setting.choices(), label=setting.label).bind_value(values, setting.name).classes("w-56")
        left_out = models_left_out() if setting.name == "model" else None
        if left_out:
            ui.label(left_out).classes("text-xs text-warning")
    else:  # text
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


def _filters_line(exp: Experiment) -> bool:
    try:
        names = switched_on_filters(exp)
    except ValueError:
        ui.label(FILTERS_UNREADABLE).classes("text-xs text-negative mt-1")
        return False
    ui.label(f"Filters switched on: {', '.join(names)}." if names else NO_FILTERS).classes("text-xs mt-1")
    return True


def _labels_line(exp: Experiment) -> bool:
    try:
        names = switched_on_labels(exp)
    except ValueError:
        ui.label(LABELS_UNREADABLE).classes("text-xs text-negative mt-1")
        return False
    ui.label(f"Labels switched on: {', '.join(names)}." if names else NO_LABELS).classes("text-xs mt-1")
    return True


def _embedding_line(exp: Experiment) -> bool:
    try:
        settings = embedding_settings(exp)
    except ValueError:
        ui.label(EMBEDDING_UNREADABLE).classes("text-xs text-negative mt-1")
        return False
    if settings is None:
        ui.label(NO_EMBEDDING_SETUP).classes("text-xs mt-1")
        return False
    count = len(settings["runs"])
    if not count:
        ui.label(ALL_SKIPPED).classes("text-xs mt-1")
        return False
    noun = "embedding" if count == 1 else "embeddings"
    ui.label(f"{count} {noun} set up on the Embedding page.").classes("text-xs mt-1")
    return True


def _batch_line(exp: Experiment) -> bool:
    try:
        setup = saved_batch_setup(exp)
    except ValueError:
        ui.label(BATCH_UNREADABLE).classes("text-xs text-negative mt-1")
        return False
    if setup is None:
        ui.label(NO_BATCH_SETUP).classes("text-xs mt-1")
        return False
    ui.label(batch_setup_line(setup)).classes("text-xs mt-1")
    return True


def _tickable(step: Step) -> bool:
    return step in (Step.BATCH, Step.EMBEDDING) or can_tick(step)


async def _ask_cpu(images: int) -> bool:
    """Ask before Segment nuclei runs on the CPU; return whether to go ahead."""
    count = f"{images} image{'' if images == 1 else 's'}"
    with ui.dialog() as dialog, ui.card().classes("p-5 gap-3 max-w-lg"):
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.icon("warning").classes("text-2xl text-warning")
            ui.label("No GPU found").classes(SECTION_TITLE)
        ui.label(f"Segment nuclei would segment {count} on the CPU.").classes("text-sm")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
            ui.button("Run on the CPU", icon="play_arrow", on_click=lambda: dialog.submit(True))
    answer = await dialog  # None when closed with Escape or a click beside it, as Cancel
    dialog.delete()
    return answer is True


def _step_card(
    exp: Experiment,
    badge: int,
    step: Step,
    ticked: dict[Step, bool],
    overwrite: dict[Step, bool],
    values: dict[str, Any],
) -> None:
    with ui.card().classes("w-full p-3 gap-1"):
        with ui.row().classes("items-center no-wrap w-full gap-3"):
            ui.label(str(badge)).classes(
                "w-[26px] h-[26px] rounded-full grid place-items-center text-xs font-semibold shrink-0"
            ).style(BADGE_STYLE)
            with ui.column().classes("gap-0 grow"):
                box = ui.checkbox(STEP_NAMES[step]).bind_value(ticked, step).classes("font-medium -ml-2")
                if not _tickable(step):
                    box.disable()
                ui.label(DESCRIPTIONS[step]).classes("text-xs muted -mt-1")
                if step == Step.SEGMENT:
                    _gpu_line()
                if step == Step.TABLE:
                    filters_read = _filters_line(exp)
                    labels_read = _labels_line(exp)
                    if not (filters_read and labels_read):
                        ticked[step] = False
                        box.disable()
                if step == Step.BATCH and not _batch_line(exp):
                    ticked[step] = False
                    box.disable()
                if step == Step.EMBEDDING and not _embedding_line(exp):
                    ticked[step] = False
                    box.disable()
            if step in PER_WELL_STEPS:
                ui.switch("Overwrite").bind_value(overwrite, step).props("dense")
        settings = shown_settings(step)
        if settings:
            with ui.expansion("Settings").classes("w-full text-sm").props("dense"), ui.column().classes("gap-2 py-1"):
                for setting in settings:
                    _control(step, setting, values)


def _color(bar: ui.linear_progress, name: str) -> None:
    if bar.props.get("color") != name:
        bar.props(f"color={name}")


def _progress_card(start: Callable[[], Awaitable[None]]) -> None:
    with ui.card().classes("grow p-5 gap-2"):
        ui.label("Progress").classes(SECTION_TITLE)
        step_line = ui.label(IDLE).classes("font-medium whitespace-pre-wrap")
        # instant-feedback: bars jump to a new value, so a new run starts empty instead of sliding
        step_bar = ui.linear_progress(value=0, show_value=False).props("size=10px color=primary instant-feedback")
        rows = []
        for depth in range(MAX_BARS):
            with ui.row().classes("w-full items-center no-wrap gap-3") as row:
                label = ui.label().classes("text-sm muted w-44 shrink-0 truncate")
                bar = ui.linear_progress(value=0, show_value=False)
                bar.props(f"size={8 - 2 * depth}px color=primary instant-feedback").classes("grow")
                # one line at any count, digits all one width: the row keeps its height
                count = ui.label().classes("text-xs muted w-28 shrink-0 text-right whitespace-nowrap tabular-nums")
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
                    log.push(text, classes=LOG_LINE)
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
                    log.push(ON_CPU_LINE, classes=LOG_LINE)
            if run is None:
                return
            show_log(run)
            if not run.finished:
                position = next(index for index, p in enumerate(run.progress) if p.outcome is None)
                bars = open_bars(run.progress[position])
                now = time.time()
                seconds = step_time_left(run.progress[position], now, run.requests[position].images)
                step_line.text = running_line(run) + ("" if seconds is None else f" · {time_left_text(seconds, now)}")
                step_bar.value = (position + step_share(run.progress[position])) / len(run.progress)
                _color(step_bar, "primary")
                show_bars(bars)
                return
            how = ending(run)
            color, text = ENDINGS[how]
            failed = [item for progress in run.progress for item in progress.failed]
            keys = any(progress.step is Step.EMBEDDING and progress.failed for progress in run.progress)
            if how == "failed" and failed:
                text += " Failed:\n" + "\n".join(failed) if keys else f" Failed: {', '.join(failed)}."
            step_line.text = text
            step_bar.value = 1
            _color(step_bar, color)
            show_bars([])

        refresh()
        ui.timer(POLL_SECONDS, refresh)


def _content(exp: Experiment) -> None:
    wells = sorted({normalize_well(well) for well in exp.well_list}, key=plate_sort_key)
    picked = dict.fromkeys(wells, True)
    ticked = dict.fromkeys(RUN_PAGE_STEPS, False)
    overwrite = dict.fromkeys(RUN_PAGE_STEPS, False)
    values = {step: {s.name: default(step, s.name) for s in shown_settings(step)} for step in RUN_PAGE_STEPS}
    workers = {"count": 0}

    async def start() -> None:
        if run_going():
            return
        chosen = [step for step in RUN_PAGE_STEPS if ticked[step] and _tickable(step)]
        if not chosen:
            ui.notify("Tick at least one step.", type="warning")
            return
        wells_picked = [well for well, on in picked.items() if on]
        if not wells_picked and any(step in PER_WELL_STEPS for step in chosen):
            ui.notify("Pick at least one well.", type="warning")
            return
        on_cpu = False
        images: int | None = None  # Segment nuclei's, for the question and its time left
        if Step.SEGMENT in chosen:
            if STATE.checking_gpu:
                ui.notify(STILL_CHECKING, type="warning")
                return
            try:
                images = images_to_segment(exp, wells_picked, overwrite=overwrite[Step.SEGMENT])
            except (OSError, ValueError) as error:
                if STATE.gpu is None:  # the question needs the count
                    ui.notify(f"The run could not start: {error}", type="negative")
                    return
                # with a GPU the run goes ahead, its time left by whole wells
            if STATE.gpu is None and images:
                if not await _ask_cpu(images) or run_going():
                    return
                on_cpu = True
        batch_setup: dict[str, Any] | None = None
        if Step.BATCH in chosen:
            try:
                batch_setup = saved_batch_setup(load_config(exp.config_path, announce=False))
            except (OSError, ValueError) as error:
                ui.notify(f"The run could not start: {error}", type="negative")
                return
            if batch_setup is None:
                ui.notify(NO_BATCH_SETUP, type="negative")
                return
        embedding: dict[str, Any] | None = None
        if Step.EMBEDDING in chosen:
            try:
                embedding = embedding_settings(load_config(exp.config_path, announce=False))
            except (OSError, ValueError) as error:
                ui.notify(f"The run could not start: {error}", type="negative")
                return
            if embedding is None or not embedding["runs"]:
                ui.notify(NO_EMBEDDING_SETUP if embedding is None else ALL_SKIPPED, type="negative")
                return
        filters: list[str] = []
        labels: list[str] = []
        if Step.TABLE in chosen:
            try:
                saved = load_config(exp.config_path, announce=False)
                filters = switched_on_filters(saved)
                labels = switched_on_labels(saved)
            except (OSError, ValueError) as error:
                ui.notify(f"The run could not start: {error}", type="negative")
                return
        requests = [
            StepRequest(
                step,
                wells_picked if step in PER_WELL_STEPS else None,
                settings_for(step, values[step], overwrite=overwrite[step], workers=workers["count"] or None),
                images=images if step == Step.SEGMENT else None,
            )
            for step in chosen
        ]
        if on_cpu:  # this run only: the next run's settings start without it
            next(request for request in requests if request.step == Step.SEGMENT).settings["allow_cpu"] = True
        if Step.TABLE in chosen:
            table_settings = next(request for request in requests if request.step == Step.TABLE).settings
            table_settings["filters"] = filters
            table_settings["labels"] = labels
        if batch_setup is not None:
            next(request for request in requests if request.step == Step.BATCH).settings.update(
                batch=batch_setup["batch"],
                keep=list(batch_setup["keep"]),
                log_intensities=batch_setup["log_intensities"],
            )
        if embedding is not None:
            next(request for request in requests if request.step == Step.EMBEDDING).settings.update(embedding)
        try:
            run = Run(exp.config_path, requests)
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
            wells_card(exp, picked)
            ui.select(WORKER_CHOICES, label="Wells processed in parallel").bind_value(workers, "count").classes("w-80")
            for badge, step in enumerate(RUN_PAGE_STEPS, start=1):
                _step_card(exp, badge, step, ticked, overwrite, values[step])
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
