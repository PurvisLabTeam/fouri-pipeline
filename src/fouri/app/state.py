"""What the whole app shares while it runs: the open experiment, the light/dark choice, the run.

The app is one window used by one person, so this is one object for the app, not one per
browser tab.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nicegui import app

from fouri.app.runner import Run
from fouri.config import Experiment, load_config

LAST_EXPERIMENT = "last_experiment"  # key in NiceGUI's general storage, kept in the user's own app-data
POLL_SECONDS = 0.25  # how often the run's progress is read


@dataclass
class AppState:
    """What the whole app shares: the experiment open, light or dark, the run; later the GPU."""

    experiment: Experiment | None = None
    dark: bool = True
    run: Run | None = None  # the run going, or the last one that ended


STATE = AppState()


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
    STATE.experiment = experiment
    app.storage.general[LAST_EXPERIMENT] = experiment.config_path
    return experiment


def reopen_last() -> Experiment | None:
    """Open the experiment this user had open last; return None when its file is gone."""
    last = app.storage.general.get(LAST_EXPERIMENT)
    if not last or not Path(last).is_file():
        return None
    return open_experiment(last)
