"""The steps as the Run page offers them: what each does, and the settings its card shows.

A setting's default is read from the step function's signature, and a card shows only the
settings its function takes today, so a parameter added to a step function appears here once it
has a row in ``SETTINGS``. The functions are looked up, never called: each step runs as its own
program.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from fouri.run import step_function
from fouri.status import PER_WELL_STEPS, Step


@dataclass(frozen=True)
class Setting:
    """One setting a step card shows: the parameter, its label, its control."""

    name: str
    label: str
    control: Literal["text", "number", "choice", "switch"]


SETTINGS: dict[Step, list[Setting]] = {
    Step.CONVERT: [Setting("align_signal", "Alignment signal", "text")],
    Step.SEGMENT: [
        Setting("diameter", "Cell diameter", "number"),
        Setting("flow_threshold", "Flow threshold", "number"),
        Setting("cellprob_threshold", "Cell probability threshold", "number"),
    ],
    Step.ALIGN: [Setting("downscale_factor", "Downscale factor", "number")],
    Step.MEASURE: [Setting("ring_width", "Ring width", "number"), Setting("ring_gap", "Ring gap", "number")],
    Step.TABLE: [],
    Step.EMBEDDING: [
        Setting("cells_per_sample", "Cells per sample", "number"),
        Setting("subsample_seed", "Subsample seed", "number"),
    ],
}

DESCRIPTIONS = {
    Step.CONVERT: "Read the ND2 files and save every channel as a TIFF",
    Step.SEGMENT: "Outline every nucleus with Cellpose",
    Step.ALIGN: "Align every round to the anchor round",
    Step.MEASURE: "Measure every cell in every channel; exclude cells touching an image edge or a blank area",
    Step.TABLE: "Combine the wells, add derived features, filters and labels, save the AnnData",
    Step.EMBEDDING: "Subsample each sample and compute the embeddings set up on the Embedding page",
}

# the steps whose n_workers counts wells: Segment nuclei has no pool, and Embedding's counts runs
WELL_WORKER_STEPS = (Step.CONVERT, Step.ALIGN, Step.MEASURE)


def _parameters(step: Step) -> dict[str, inspect.Parameter]:
    """Return the step function's parameters after the experiment."""
    parameters = list(inspect.signature(step_function(step)).parameters.values())[1:]
    return {parameter.name: parameter for parameter in parameters}


def shown_settings(step: Step) -> list[Setting]:
    """Return the settings a step's card shows: the rows whose parameter its function takes."""
    parameters = _parameters(step)
    return [setting for setting in SETTINGS[step] if setting.name in parameters]


def default(step: Step, name: str) -> Any:
    """Return a setting's default, from the step function's signature."""
    return _parameters(step)[name].default


def whole_number(step: Step, name: str) -> bool:
    """Return whether a number setting takes whole numbers: its default is one, and not a float."""
    parameter = _parameters(step)[name]
    value = parameter.default
    return isinstance(value, int) and not isinstance(value, bool) and "float" not in str(parameter.annotation)


def can_tick(step: Step) -> bool:
    """Return whether a card can be ticked: every setting its function takes has a default.

    Embedding needs the runs and features set up on the Embedding page, so it can't be ticked yet.
    """
    return all(parameter.default is not inspect.Parameter.empty for parameter in _parameters(step).values())


def _as_set(step: Step, name: str, value: Any) -> Any:
    """Return a box's value as the step takes it: a whole number as an int, 8 rather than 8.0."""
    if isinstance(value, float) and value.is_integer() and whole_number(step, name):
        return int(value)
    return value


def settings_for(step: Step, shown: Mapping[str, Any], *, overwrite: bool, workers: int | None) -> dict[str, Any]:
    """Return the settings object a step's card sends to its program."""
    parameters = _parameters(step)
    settings: dict[str, Any] = {}
    if step in PER_WELL_STEPS and "overwrite" in parameters:
        settings["overwrite"] = overwrite
    if workers is not None and step in WELL_WORKER_STEPS and "n_workers" in parameters:
        settings["n_workers"] = workers
    for setting in shown_settings(step):
        value = shown.get(setting.name)
        if value is None or (isinstance(value, str) and not value.strip()):  # an empty box keeps the default
            continue
        value = _as_set(step, setting.name, value.strip() if isinstance(value, str) else value)
        if value != parameters[setting.name].default:
            settings[setting.name] = value
    return settings
