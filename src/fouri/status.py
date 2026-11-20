"""Status of each pipeline step, read from the files on disk.

`step_status` returns the status of a per-well step for each well. `experiment_status` returns the
status of each experiment-wide step. `measured_wells` divides the wells by whether Measure cells is
complete. The status of a per-well step is computed with the step's skip rule, against the rounds
listed in `info_exp.csv`.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from fouri.config import Experiment
from fouri.io import (
    COMBAT_LAYER,
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
    """Pipeline steps, listed in run order.

    Each member's value is the step's name for `python -m fouri.run`.

    Attributes
    ----------
    CONVERT : , default="convert"
        Convert images.
    SEGMENT : , default="segment"
        Segment nuclei.
    ALIGN : , default="align"
        Align rounds.
    ARTIFACTS : , default="artifacts"
        Find artifacts.
    MEASURE : , default="measure"
        Measure cells.
    TABLE : , default="table"
        Build cell table.
    BATCH : , default="batch"
        Batch effects.
    EMBEDDING : , default="embedding"
        Embedding.
    EXPORT : , default="export"
        Export.
    FINETUNE : , default="finetune"
        Fine-tune Cellpose.
    """

    CONVERT = "convert"
    SEGMENT = "segment"
    ALIGN = "align"
    ARTIFACTS = "artifacts"
    MEASURE = "measure"
    TABLE = "table"
    BATCH = "batch"
    EMBEDDING = "embedding"
    EXPORT = "export"
    FINETUNE = "finetune"


PER_WELL_STEPS = (Step.CONVERT, Step.SEGMENT, Step.ALIGN, Step.ARTIFACTS, Step.MEASURE)
WELL_STEPS = (*PER_WELL_STEPS, Step.FINETUNE)


class State(StrEnum):
    """State of a step for a single well or for the experiment.

    Attributes
    ----------
    COMPLETE : , default="complete"
        Every round is done, or the step's output exists.
    PARTIAL : , default="partial"
        Some but not all rounds are done, or the subsample does not have any embeddings.
    NOT_STARTED : , default="not started"
        No round is done, or the step's output does not exist.
    BLOCKED : , default="blocked"
        The step refuses the well. `WellStatus.note` states the reason.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_STARTED = "not started"
    BLOCKED = "blocked"  # the step would stop here; the note says why


@dataclass(frozen=True)
class WellStatus:
    """Status of a per-well step for a single well, read from the step's output files.

    Attributes
    ----------
    state : State
        The step's state for the well.
    done : int
        The number of rounds done. 0 when `state` is `BLOCKED`.
    total : int
        The number of rounds the step processes.
    note : str, default=""
        The reason for a `BLOCKED` state. Empty for any other state.
    """

    state: State
    done: int
    total: int
    note: str = ""


@dataclass(frozen=True)
class OutputStatus:
    """Status of an experiment-wide step, read from the step's output file.

    Attributes
    ----------
    state : State
        The step's state.
    path : Path
        The path to the step's output file.
    updated : dt.datetime | None
        The output file's last modification time, in local time. None when `state` is `NOT_STARTED`.
    note : str, default=""
        A description of the output (e.g., `3 embeddings`). May be empty.
    """

    state: State
    path: Path
    updated: dt.datetime | None
    note: str = ""


def measured_wells(exp: Experiment) -> tuple[list[str], list[str]]:
    """Return the wells for which Measure cells is complete, then the other wells.

    The state of each well is the one `step_status` returns for Measure cells. Each list is sorted
    by plate row, then by column.

    Parameters
    ----------
    exp : Experiment
        The Experiment that holds the cell data files.

    Returns
    -------
    measured : list[str]
        The wells in `well_list` whose Measure cells state is `COMPLETE`.
    not_measured : list[str]
        The other wells in `well_list`.
    """
    measured_stats = step_status(exp, Step.MEASURE)
    completed = [well for well, status in measured_stats.items() if status.state is State.COMPLETE]
    not_completed = [well for well, status in measured_stats.items() if status.state is not State.COMPLETE]

    return completed, not_completed


def step_status(exp: Experiment, step: Step | str, wells: Sequence[str] | None = None) -> dict[str, WellStatus]:
    """Return the status of a per-well step for each chosen well.

    The rounds done are counted, for each well, against the rounds listed in `info_exp.csv`:

    - Convert images: the rounds in the round table. `NOT_STARTED` when the round table file does
      not exist. `BLOCKED` when the round table's rounds are not the first rounds of `info_exp.csv`,
      in order.
    - Segment nuclei: the rounds with a label (segmentation mask) file.
    - Align rounds: the rounds whose saved transform does not contain NaN. The anchor round is
      excluded from `done` and `total`. A missing transforms file counts as no round done. `BLOCKED`
      when the transforms file holds more rounds than `info_exp.csv`, or when the anchor round's
      matrix is missing or is not the identity matrix.
    - Find artifacts: the rounds with an artifact mask file. `NOT_STARTED` when neither a round
      table file nor a mask file exists. `BLOCKED` when at least one mask file exists without a
      round table file, or when a mask record file cannot be read as a JSON object with a boolean
      `edited`.
    - Measure cells: the rounds for which the cell data file has a `<prefix>_nuc_mean` column for
      every signal of the round, `<prefix>` built by `feature_prefix` from the round name and the
      signal (e.g., `R0_DNA_nuc_mean`). `NOT_STARTED` when the cell data file does not exist.

    Otherwise, the state is `COMPLETE` when every round is done, `PARTIAL` when some rounds are
    done, and `NOT_STARTED` when no round is done. A `step` that is not a `Step` value raises
    `ValueError`. A well not in `well_list` raises `ValueError` from `Experiment.pick_wells`. A
    missing `info_exp.csv` file raises `FileNotFoundError`.

    Parameters
    ----------
    exp : Experiment
        The Experiment that holds the files.
    step : Step | str
        The step, as a `Step` or a `Step` value (e.g., `measure`).
    wells : Sequence[str] | None, default=None
        The wells to check. None for every well in `well_list`.

    Returns
    -------
    dict[str, WellStatus]
        The status of each well, keyed by well, sorted by plate row, then by column.

    Raises
    ------
    ValueError
        If `step` is not one of `PER_WELL_STEPS`.
    """
    step = Step(step)
    if step not in PER_WELL_STEPS:
        raise ValueError(f"{step} is not a per-well step; use experiment_status for it.")
    rounds = read_info_file(exp)
    check = _CHECKS[step]
    return {well: check(exp, well, rounds) for well in exp.pick_wells(wells)}


def experiment_status(exp: Experiment) -> dict[Step, OutputStatus]:
    """Return the status of Build cell table, Batch effects, and Embedding.

    - Build cell table: `COMPLETE` when the `standard_adata.h5ad` file exists.
    - Batch effects: `COMPLETE` when the `standard_adata.h5ad` file holds the layer `COMBAT_LAYER`.
      The note is `by <batch>`, `<batch>` being the batch column saved in `.uns["batch_effects"]`.
      The note is empty when the batch column is not saved.
    - Embedding: `COMPLETE` when the `standard_adata_sub.h5ad` file holds at least one embedding in
      `.obsm`. The note states the number of embeddings. `PARTIAL` when the file does not contain
      any embedding.

    Otherwise, the state is `NOT_STARTED`. The checks for Batch effects and Embedding raise
    `OSError` for a file that exists but cannot be opened as HDF5.

    Parameters
    ----------
    exp : Experiment
        The Experiment that holds the files.

    Returns
    -------
    dict[Step, OutputStatus]
        The status of each step, keyed by `Step.TABLE`, `Step.BATCH`, and `Step.EMBEDDING`.
    """
    return {Step.TABLE: _table(exp), Step.BATCH: _batch(exp), Step.EMBEDDING: _embedding(exp)}


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
    if not round_table_path(exp, well).exists():
        if done:
            return WellStatus(State.BLOCKED, 0, total, "masks saved but no round table; rerun Convert images")
        return WellStatus(State.NOT_STARTED, 0, total)
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


def _saved_batch(file: h5py.File) -> str | None:
    try:
        entry = file["uns/batch_effects/batch"]
        if not isinstance(entry, h5py.Dataset):
            return None
        batch = entry.asstr()[()]
    except (KeyError, TypeError, ValueError, OSError):
        return None
    return batch if isinstance(batch, str) and batch else None


def _batch(exp: Experiment) -> OutputStatus:
    path = adata_path(exp)
    if not path.exists():
        return OutputStatus(State.NOT_STARTED, path, None)
    with h5py.File(path, "r") as file:
        layers = file.get("layers")
        if not isinstance(layers, h5py.Group) or COMBAT_LAYER not in layers:
            return OutputStatus(State.NOT_STARTED, path, None)
        batch = _saved_batch(file)
    return OutputStatus(State.COMPLETE, path, _updated(path), f"by {batch}" if batch else "")


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
