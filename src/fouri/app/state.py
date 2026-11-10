"""What the whole app shares while it runs: the open experiment and the light/dark choice.

The app is one window used by one person, so this is one object for the app, not one per
browser tab.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nicegui import app

from fouri.config import Experiment, load_config

LAST_EXPERIMENT = "last_experiment"  # key in NiceGUI's general storage, kept in the user's own app-data


@dataclass
class AppState:
    """What the whole app shares: the experiment open, light or dark, later the GPU and the run."""

    experiment: Experiment | None = None
    dark: bool = True


STATE = AppState()


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
