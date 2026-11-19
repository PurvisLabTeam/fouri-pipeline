"""What the whole app shares while it runs: the open experiment, light or dark, the GPU, the run.

The app is one window used by one person, so this is one object for the app, not one per
browser tab.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nicegui import app

from fouri.app.gpu import check_gpu
from fouri.app.runner import Run
from fouri.config import Experiment, load_config, save_config

LAST_EXPERIMENT = "last_experiment"  # key in NiceGUI's general storage, kept in the user's own app-data
POLL_SECONDS = 0.25  # how often the run's progress is read
APP_NOT_READ = "experiment.json's app entry can't be read; nothing was saved."


@dataclass
class AppState:
    """What the whole app shares: the experiment open, light or dark, the GPU found, the run."""

    experiment: Experiment | None = None
    dark: bool = False
    run: Run | None = None  # the run going, or the last one that ended
    gpu: str | None = None  # the GPU's name; None when none was found
    checking_gpu: bool = True  # the check made when the app opens has not answered yet
    kept: dict[str, dict[str, Any]] = field(default_factory=dict)


STATE = AppState()


async def find_gpu() -> None:
    """Check for a GPU in a program of its own and keep what it found.

    The app does this once, when it starts. A check that fails leaves ``gpu`` as None: no GPU found.
    """
    STATE.checking_gpu = True
    try:
        STATE.gpu = await check_gpu()
    finally:
        STATE.checking_gpu = False


def run_going() -> bool:
    """Return whether a run is going."""
    return STATE.run is not None and not STATE.run.finished


def poll_run() -> None:
    """Read the run's progress, starting its next step when one ends; the app calls this on a timer.

    It runs for the whole app rather than for one page, so a run goes on, and the header stays
    right, whichever page is open.
    """
    if STATE.run is not None and not STATE.run.finished:
        STATE.run.poll()


def stop_run() -> None:
    """Stop the run going, if any: the step ends as stopped after its current items."""
    if STATE.run is not None and not STATE.run.finished:
        STATE.run.stop()


def open_experiment(path: str | Path) -> Experiment:
    """Load an experiment.json or its folder as the open experiment, and record the path as last."""
    experiment = load_config(path, announce=False)
    if STATE.experiment is None or STATE.experiment.config_path != experiment.config_path:
        STATE.kept.clear()
    STATE.experiment = experiment
    app.storage.general[LAST_EXPERIMENT] = experiment.config_path
    return experiment


def save_app_value(exp: Experiment, key: str, value: Any) -> Experiment:
    with open(exp.config_path, encoding="utf-8") as file:
        config = json.load(file)
    app_value = config.get("app")
    if app_value is not None and not isinstance(app_value, dict):
        raise ValueError(APP_NOT_READ)
    config["app"] = {**(app_value or {}), key: value}
    save_config(config, exp.config_path)
    return open_experiment(exp.config_path)


def kept_choices(page: str) -> dict[str, Any]:
    return STATE.kept.setdefault(page, {})


def reopen_last() -> Experiment | None:
    """Open the experiment this user had open last; return None when its file is gone."""
    last = app.storage.general.get(LAST_EXPERIMENT)
    if not last or not Path(last).is_file():
        return None
    return open_experiment(last)
