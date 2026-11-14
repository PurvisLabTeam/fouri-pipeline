"""How far each step has got, read from the files on disk.

The app's Status page and Run page read these instead of keeping their own record, so they are
right after a crash, after a run on another PC, or after work done outside the app. Each check
follows the rule its step uses to skip work, counted against the rounds info_exp.csv lists now.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from fouri.config import Experiment
from fouri.io import (
    adata_path,
    adata_sub_path,
    cell_data_path,
    label_path,
    load_transforms,
    mask_path,
    mask_record_path,
    read_info_file,
    read_round_table,
    round_table_path,
)
from fouri.metadata import feature_prefix


class Step(StrEnum):
    """The steps a run can include, in their usual order."""

    CONVERT = "convert"
    SEGMENT = "segment"
    ALIGN = "align"
    ARTIFACTS = "artifacts"
    MEASURE = "measure"
    TABLE = "table"
    EMBEDDING = "embedding"
    EXPORT = "export"


PER_WELL_STEPS = (Step.CONVERT, Step.SEGMENT, Step.ALIGN, Step.ARTIFACTS, Step.MEASURE)


class State(StrEnum):
    """Where a step stands for one well, or for the whole experiment."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_STARTED = "not started"
    BLOCKED = "blocked"  # the step would stop here; the note says why


@dataclass(frozen=True)
class WellStatus:
    """One per-well step for one well: rounds done out of the rounds the step works on."""

    state: State
    done: int
    total: int
    note: str = ""


@dataclass(frozen=True)
class OutputStatus:
    """An experiment-wide step: its output file, when it was last written, and what it holds."""

    state: State
    path: Path
    updated: dt.datetime | None
    note: str = ""


def step_status(exp: Experiment, step: Step | str, wells: Sequence[str] | None = None) -> dict[str, WellStatus]:
    """Return each chosen well's state for one per-well step, in plate order."""
    step = Step(step)
    if step not in PER_WELL_STEPS:
        raise ValueError(f"{step} is not a per-well step; use experiment_status for it.")
    rounds = read_info_file(exp)
    check = _CHECKS[step]
    return {well: check(exp, well, rounds) for well in exp.pick_wells(wells)}


def experiment_status(exp: Experiment) -> dict[Step, OutputStatus]:
    """Return the state of the experiment-wide steps: the cell table and the embeddings."""
    return {Step.TABLE: _table(exp), Step.EMBEDDING: _embedding(exp)}


def _state(done: int, total: int) -> State:
    if done >= total:
        return State.COMPLETE
    return State.PARTIAL if done > 0 else State.NOT_STARTED


def _convert(exp: Experiment, well: str, rounds: pd.DataFrame) -> WellStatus:
    """Count converted rounds: the round table lists info_exp.csv's rounds, or the first of them."""
    names = rounds["round_name"].astype(str).tolist()
    if not round_table_path(exp, well).exists():
        return WellStatus(State.NOT_STARTED, 0, len(names))
    table = read_round_table(exp, well)
    converted = table.sort_values("round_index")["round_name"].astype(str).unique().tolist()
    if converted != names[: len(converted)]:
        return WellStatus(
            State.BLOCKED,
            0,
            len(names),
            f"converted rounds ({', '.join(converted)}) don't match info_exp.csv ({', '.join(names)})",
        )
    return WellStatus(_state(len(converted), len(names)), len(converted), len(names))


def _segment(exp: Experiment, well: str, rounds: pd.DataFrame) -> WellStatus:
    """Count segmented rounds: a label image exists for the round."""
    total = len(rounds)
    done = sum(label_path(exp, well, round_index).exists() for round_index in range(total))
    return WellStatus(_state(done, total), done, total)


def _align(exp: Experiment, well: str, rounds: pd.DataFrame) -> WellStatus:
    """Count aligned rounds, the anchor left out: the saved matrix isn't NaN."""
    anchor = exp.anchor_round
    total = len(rounds) - 1
    try:
        saved = load_transforms(exp, well)
    except FileNotFoundError:
        return WellStatus(_state(0, total), 0, total)
    if len(saved) > len(rounds):
        return WellStatus(State.BLOCKED, 0, total, f"{len(saved)} rounds saved, {len(rounds)} rounds in info_exp.csv")
    if anchor >= len(saved) or not np.allclose(saved[anchor], np.eye(3)):
        return WellStatus(State.BLOCKED, 0, total, f"saved with a different anchor round than round {anchor}")
    done = sum(not np.isnan(matrix).any() for round_index, matrix in enumerate(saved) if round_index != anchor)
    return WellStatus(_state(done, total), done, total)


def _measure(exp: Experiment, well: str, rounds: pd.DataFrame) -> WellStatus:
    """Count measured rounds: the cell data has a nucleus mean for every channel of the round."""
    total = len(rounds)
    path = cell_data_path(exp, well)
    if not path.exists():
        return WellStatus(State.NOT_STARTED, 0, total)
    columns = set(pd.read_csv(path, nrows=0).columns)
    channels = rounds.columns[1:]
    done = 0
    for _, row in rounds.iterrows():
        signals = [str(row[c]).strip() for c in channels if pd.notna(row[c]) and str(row[c]).strip()]
        done += all(f"{feature_prefix(row['round_name'], signal)}_nuc_mean" in columns for signal in signals)
    return WellStatus(_state(done, total), done, total)


def _record_readable(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and isinstance(record.get("edited"), bool)


def _artifacts(exp: Experiment, well: str, rounds: pd.DataFrame) -> WellStatus:
    total = len(rounds)
    done = sum(mask_path(exp, well, round_index).exists() for round_index in range(total))
    try:
        table = read_round_table(exp, well)
    except FileNotFoundError:
        if done:
            return WellStatus(State.BLOCKED, 0, total, "masks saved but no round table; rerun Convert images")
        return WellStatus(State.NOT_STARTED, 0, total)
    if "raw_file_name" not in table.columns:
        return WellStatus(State.BLOCKED, 0, total, "round table made by an older version; rerun Convert images")
    unreadable = [
        str(round_index)
        for round_index in range(total)
        if not _record_readable(mask_record_path(exp, well, round_index))
    ]
    if len(unreadable) == 1:
        return WellStatus(State.BLOCKED, 0, total, f"mask record of round {unreadable[0]} can't be read")
    if unreadable:
        return WellStatus(State.BLOCKED, 0, total, f"mask records of rounds {', '.join(unreadable)} can't be read")
    return WellStatus(_state(done, total), done, total)


_CHECKS = {
    Step.CONVERT: _convert,
    Step.SEGMENT: _segment,
    Step.ALIGN: _align,
    Step.ARTIFACTS: _artifacts,
    Step.MEASURE: _measure,
}


def _updated(path: Path) -> dt.datetime | None:
    return dt.datetime.fromtimestamp(path.stat().st_mtime) if path.exists() else None


def _table(exp: Experiment) -> OutputStatus:
    path = adata_path(exp)
    if not path.exists():
        return OutputStatus(State.NOT_STARTED, path, None)
    return OutputStatus(State.COMPLETE, path, _updated(path))


def _embedding(exp: Experiment) -> OutputStatus:
    path = adata_sub_path(exp)
    if not path.exists():
        return OutputStatus(State.NOT_STARTED, path, None)
    import anndata as ad

    sub = ad.read_h5ad(path, backed="r")
    count = len(sub.obsm)
    sub.file.close()
    if count == 0:
        return OutputStatus(State.PARTIAL, path, _updated(path), "subsampled, no embeddings yet")
    return OutputStatus(State.COMPLETE, path, _updated(path), f"{count} embedding{'s' if count != 1 else ''}")
