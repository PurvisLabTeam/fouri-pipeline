"""The Experiment page: the experiment's paths and anchor round, its plate, and its rounds."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path, PurePath
from typing import Any

import pandas as pd
from nicegui import ui

from fouri.app.dialogs import pick_file, pick_folder
from fouri.app.plate import overall_state, plate
from fouri.app.shell import page_frame
from fouri.app.state import STATE, open_experiment
from fouri.config import DEFAULT_PATHS, Experiment, save_config
from fouri.io import read_info_file
from fouri.status import State

SECTION_TITLE = "text-[1.05rem] font-semibold"
STATE_COLORS = {State.COMPLETE: "#0F9E8E", State.PARTIAL: "#D97706", State.NOT_STARTED: "#A8A3BC"}
NOT_IMAGED = "border: 1px dashed var(--c-line)"
LEGEND = [
    ("complete", f"background: {STATE_COLORS[State.COMPLETE]}"),
    ("partial", f"background: {STATE_COLORS[State.PARTIAL]}"),
    ("not started", f"background: {STATE_COLORS[State.NOT_STARTED]}"),
    ("not imaged", NOT_IMAGED),
]
CAN_NOT_OPEN = (OSError, ValueError, AttributeError, TypeError)  # what loading a file that isn't an experiment raises
TYPE_IN_FULL = "Type the full path to experiment.json or its folder."


# -- how the paths are written ---------------------------------------------------------------


def _same(first: str, second: str) -> bool:
    return os.path.normcase(os.path.normpath(first)) == os.path.normcase(os.path.normpath(second))


def _in_full(value: str, base: str) -> str:
    """Return a typed path in full; one without a drive or root is taken as relative to ``base``."""
    path = os.path.expanduser(value.strip())
    pure = PurePath(path)
    return os.path.normpath(path if pure.drive or pure.root else os.path.join(base, path))


def _results_key(config: dict[str, Any]) -> str:
    """Return the key Experiment reads the results folder from on this PC."""
    if platform.system() != "Windows" and (config.get("mac_base_dir") or "").strip():
        return "mac_base_dir"
    return "base_dir"


def _inside(path: str, folder: str) -> bool:
    try:
        return _same(os.path.commonpath([path, folder]), folder)
    except ValueError:  # on different drives
        return False


def _path_entry(value: str, base_dir: str, default: str) -> str | None:
    """Return a path as written under ``paths``: None when base_dir's own, else relative or full."""
    if not value.strip():
        return None
    path = _in_full(value, base_dir)
    if _same(path, os.path.join(base_dir, default)):
        return None
    if _inside(path, base_dir):
        return PurePath(os.path.relpath(path, base_dir)).as_posix()
    return path


def config_to_save(exp: Experiment, *, data: str, info_file: str, results: str, anchor: int | None) -> dict[str, Any]:
    """Return the experiment's settings with the page's paths and anchor round, as Save writes them.

    The results folder goes to the key it is read from on this PC: ``mac_base_dir`` on a PC that
    isn't Windows when that is set, ``base_dir`` otherwise. That key is left out only when the
    folder would be read the same without it: the folder holding experiment.json, with no other
    key to take over, so on such a PC ``mac_base_dir`` stays while ``base_dir`` is set. The ND2
    folder and info_exp.csv go under ``paths``, each left out when its box is empty or it is the
    results folder's own ``data`` or ``info_exp.csv``, relative to the results folder when
    inside it, and in full otherwise. A path typed without a drive or root is relative to the
    results folder. ``anchor`` None keeps the saved anchor round.
    """
    config = dict(exp.config)
    home = os.path.dirname(exp.config_path)
    base_dir = _in_full(results, home) if results.strip() else home
    key = _results_key(config)
    config[key] = base_dir
    without = {name: value for name, value in config.items() if name != key}
    if _same(Experiment(without, exp.config_path).base_dir, base_dir):
        config = without
    paths = dict(config.get("paths") or {})
    for name, value in (("data", data), ("info_file", info_file)):
        entry = _path_entry(value, base_dir, DEFAULT_PATHS[name])
        if entry is None:
            paths.pop(name, None)
        else:
            paths[name] = entry
    if paths:
        config["paths"] = paths
    else:
        config.pop("paths", None)
    if anchor is not None:
        config["anchor_round"] = anchor
    return config


# -- reading what the page shows ---------------------------------------------------------------


def _reason(error: Exception) -> str:
    text = error.strerror if isinstance(error, OSError) and error.strerror else str(error)
    return text.rstrip(".")


def _read_rounds(exp: Experiment) -> tuple[pd.DataFrame | None, str]:
    """Return info_exp.csv's rounds, or None and what to tell the user when it can't be read."""
    try:
        return read_info_file(exp), ""
    except (OSError, ValueError) as error:
        reason = _reason(error)
        return None, f"Can't read info_exp.csv at {exp.info_file}: {reason}. Pick the right file above, then Save."


def _anchor(exp: Experiment, rounds: pd.DataFrame) -> tuple[int | None, str]:
    """Return the anchor round's index, or None and what to tell the user when it isn't a round."""
    try:
        anchor = exp.anchor_round
    except ValueError:
        anchor = None
    if anchor is None or anchor >= len(rounds):
        value = json.dumps(exp.config.get("anchor_round", 0))
        return None, (
            f"The anchor round in experiment.json ({value}) isn't one of the rounds in info_exp.csv. "
            "Pick one above, then Save."
        )
    return anchor, ""


def _plate_states(exp: Experiment) -> tuple[dict[str, State] | None, str]:
    """Return each well's overall state, or None and what to tell the user if one can't be read."""
    try:
        return overall_state(exp), ""
    except Exception as error:  # any well whose files can't be read stops the whole plate
        return None, f"Can't show the plate: {_reason(error)}"


def _hover_text(well: str, metadata: dict[str, Any], state: State) -> str:
    parts = [str(metadata[key]) for key in ("treatment", "treatment_time") if metadata.get(key) not in (None, "")]
    return f"{well}: {' · '.join([*parts, state.value])}"


def _count(number: int, word: str) -> str:
    return f"{number} {word}{'' if number == 1 else 's'}"


# -- the page ------------------------------------------------------------------------------------


def _path_row(label: str, value: str, hint: str) -> tuple[ui.input, ui.button]:
    with ui.row().classes("w-full items-end no-wrap gap-2"):
        box = ui.input(label, value=value).classes("grow").tooltip(hint)
        browse = ui.button(icon="folder_open").props("outline").tooltip("Browse")
    return box, browse


def _show_open_path(box: ui.input) -> None:
    box.value = STATE.experiment.config_path if STATE.experiment is not None else ""


def _open(path: str | Path, box: ui.input) -> None:
    """Open ``path`` as the experiment; else say why and put the open experiment's path back."""
    try:
        open_experiment(path)
    except CAN_NOT_OPEN as error:
        # no experiment.json there: load_config's own message, which names the path
        nothing_there = isinstance(error, FileNotFoundError) and error.strerror is None
        message = str(error) if nothing_there else f"Can't open {path} as an experiment: {_reason(error)}"
        ui.notify(message, type="negative")
        _show_open_path(box)
        return
    _content.refresh()


def _open_typed(box: ui.input) -> None:
    path = os.path.expanduser(box.value.strip())
    pure = PurePath(path)
    if not (pure.drive or pure.root):
        ui.notify(TYPE_IN_FULL, type="negative")
        _show_open_path(box)
        return
    _open(path, box)


async def _open_other(box: ui.input) -> None:
    chosen = await pick_file(box.value, suffix=".json")
    if chosen is not None:
        _open(chosen, box)


async def _browse_folder(box: ui.input) -> None:
    chosen = await pick_folder(box.value)
    if chosen is not None:
        box.value = str(chosen)


async def _browse_csv(box: ui.input) -> None:
    chosen = await pick_file(box.value, suffix=".csv")
    if chosen is not None:
        box.value = str(chosen)


def _message_card(text: str) -> None:
    with ui.card().classes("p-5 max-w-[480px]"), ui.row().classes("items-start no-wrap gap-3"):
        ui.icon("warning").classes("text-xl").style("color: var(--q-warning)")
        ui.label(text).classes("text-sm")


def _plate_card(exp: Experiment, states: dict[str, State]) -> None:
    colors = {well: STATE_COLORS[state] for well, state in states.items()}
    metadata = exp.well_metadata
    hover = {well: _hover_text(well, metadata.get(well, {}), state) for well, state in states.items()}
    with ui.card().classes("p-5"):
        ui.label("Plate").classes(SECTION_TITLE)
        with ui.element("div").classes("mt-2"):
            plate(exp, colors, hover=hover)
        with ui.row().classes("gap-3 text-xs muted mt-3"):
            for name, style in LEGEND:
                with ui.row().classes("items-center gap-1"):
                    ui.element("div").classes("w-3 h-3 rounded").style(style)
                    ui.label(name)


def _rounds_card(exp: Experiment, rounds: pd.DataFrame, anchor: int) -> None:
    channels = [str(channel) for channel in rounds.columns[1:]]
    names = rounds["round_name"].astype(str).tolist()
    with ui.card().classes("w-full max-w-[1000px] p-5 gap-2"):
        ui.label("Rounds").classes(SECTION_TITLE)
        ui.label(
            f"{_count(len(names), 'round')}, {_count(len(channels), 'channel')}, from {Path(exp.info_file).name}. "
            f"The anchor round is {names[anchor]}."
        ).classes("text-sm muted")
        with ui.grid(columns=len(channels) + 1).classes("gap-x-6 gap-y-1 items-center mt-1"):
            ui.label("Round").classes("text-sm font-semibold")
            for channel in channels:
                ui.label(channel).classes("text-sm font-semibold")
            for index, name in enumerate(names):
                ui.label(name + (" (anchor)" if index == anchor else "")).classes("text-sm font-mono")
                for channel in channels:
                    value = rounds[channel].iloc[index]
                    marker = str(value).strip() if pd.notna(value) and str(value).strip() else "—"
                    ui.label(marker).classes("text-sm" + (" muted" if marker in {"—", "DNA"} else " font-medium"))


@ui.refreshable
def _content() -> None:
    exp = STATE.experiment
    with ui.row().classes("w-full gap-6 items-start"):
        with ui.card().classes("w-[560px] p-5 gap-3"):
            ui.label("Experiment").classes(SECTION_TITLE)
            if exp is not None and exp.name:
                ui.label(exp.name).classes("muted -mt-2")
            config_box, browse = _path_row(
                "experiment.json", exp.config_path if exp else "", "Wells, conditions and settings"
            )
            config_box.on("keydown.enter", lambda: _open_typed(config_box))
            browse.on_click(lambda: _open_other(config_box))
            if exp is None:
                return

            data_box, browse = _path_row(
                "Raw images (ND2 folder)", exp.data_dir, "One folder per round, as saved by the microscope"
            )
            browse.on_click(lambda: _browse_folder(data_box))
            info_box, browse = _path_row(
                "Round list (info_exp.csv)", exp.info_file, "The antibody in each channel of each round"
            )
            browse.on_click(lambda: _browse_csv(info_box))
            results_box, browse = _path_row("Results folder", exp.base_dir, "Output folder for every step")
            browse.on_click(lambda: _browse_folder(results_box))

            rounds, problem = _read_rounds(exp)
            anchor: int | None = None
            states: dict[str, State] | None = None
            if rounds is not None:
                anchor, problem = _anchor(exp, rounds)
            if not problem:
                states, problem = _plate_states(exp)
            with ui.row().classes("items-center gap-2 w-full"):
                anchor_box = None
                if rounds is not None:
                    names = rounds["round_name"].astype(str).tolist()
                    options = {index: name + (" (default)" if index == 0 else "") for index, name in enumerate(names)}
                    anchor_box = (
                        ui.select(options, value=anchor, label="Anchor round")
                        .classes("w-40")
                        .tooltip("The round all other rounds are aligned to")
                    )
                ui.space()

                def save() -> None:
                    config = config_to_save(
                        exp,
                        data=data_box.value,
                        info_file=info_box.value,
                        results=results_box.value,
                        anchor=anchor_box.value if anchor_box is not None else None,
                    )
                    try:
                        save_config(config, exp.config_path)
                    except (OSError, ValueError) as error:
                        ui.notify(f"Not saved: {_reason(error)}", type="negative")
                        return
                    open_experiment(exp.config_path)
                    ui.notify("Saved. The previous version is in experiment.json.bak.", type="positive")
                    _content.refresh()

                ui.button("Save to experiment.json", icon="save", on_click=save)

        if states is None:
            _message_card(problem)
        else:
            _plate_card(exp, states)
    if rounds is not None and anchor is not None and states is not None:
        _rounds_card(exp, rounds, anchor)


@ui.page("/")
def experiment_page() -> None:
    with page_frame("Experiment"):
        _content()
