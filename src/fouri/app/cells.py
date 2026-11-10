"""The cells the cell pages start from: every measured well's cells, with metadata and groups.

Explore, Filters, Labels & gates and Batch effects read the per-well cell data files as soon as
Measure cells has run for a well, so they do not wait for Build cell table.
"""

from __future__ import annotations

import json
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fouri.app.names import Feature, describe, natural_key
from fouri.config import Experiment
from fouri.features import add_derived_features, metadata_obs_columns
from fouri.io import cell_data_path, read_info_file
from fouri.metadata import normalize_well, plate_sort_key
from fouri.status import State, Step, step_status

# the groups added besides the wells' metadata
CONDITION = "condition"
WELL = "well"
PLATE_ROW = "plate row"
PLATE_COLUMN = "plate column"
CONDITION_JOIN = " · "  # between the condition columns' values: Palbo · 24h

ROW_NUMBER = "Unnamed: 0"  # the row-number column older cell data files have


@dataclass(frozen=True)
class Cells:
    """The cells of the measured wells, with their metadata, groups and derived features."""

    table: pd.DataFrame
    features: dict[str, Feature]
    groups: list[str]
    measured: list[str]
    missing: list[str]


# the last cells read, and what they were read from
_saved: tuple[tuple[Hashable, ...], Cells] | None = None


def _modified(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def _read_from(exp: Experiment) -> tuple[Hashable, ...]:
    """Return what the cells are read from: the experiment and each file's modification time."""
    return (
        exp.config_path,
        json.dumps(exp.config, sort_keys=True, default=str),
        _modified(Path(exp.info_file)),
        *(_modified(cell_data_path(exp, well)) for well in exp.well_list),
    )


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip()


def _condition(meta: dict[str, object], columns: list[str]) -> str:
    """Join a well's values in the condition columns, leaving out the ones it leaves blank."""
    return CONDITION_JOIN.join(str(meta[c]).strip() for c in columns if c in meta and not _blank(meta[c]))


def _in_natural_order(values: pd.Series) -> pd.Categorical:
    return pd.Categorical(values, categories=sorted(values.dropna().unique(), key=natural_key), ordered=True)


def _load(exp: Experiment) -> Cells:
    status = step_status(exp, Step.MEASURE)
    wells = sorted(status, key=plate_sort_key)
    measured = [well for well in wells if status[well].state == State.COMPLETE]
    missing = [well for well in wells if well not in measured]
    if not measured:
        return Cells(pd.DataFrame(), {}, [], measured, missing)

    frames = []
    for well in measured:
        frame = pd.read_csv(cell_data_path(exp, well)).drop(columns=ROW_NUMBER, errors="ignore")
        frame[WELL] = well
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)

    # the metadata under the names the AnnData uses (treatment_time is "treatment time")
    obs_names = dict(zip(exp.metadata_columns, metadata_obs_columns(exp), strict=True))
    metadata = exp.well_metadata
    added = {
        name: table[WELL].map({well: metadata[well].get(column) for well in measured})
        for column, name in obs_names.items()
    }
    table = pd.concat([table, pd.DataFrame(added, index=table.index)], axis=1)

    # add_derived_features copies the table into one block, so the columns after it add quickly
    table = add_derived_features(table, exp.derived_features)

    # a condition column no well has is left out, as when sample IDs are numbered
    conditions = [c for c in exp.condition_columns if c in obs_names]
    table[CONDITION] = table[WELL].map({well: _condition(metadata[well], conditions) for well in measured})
    table[PLATE_ROW] = table[WELL].map({well: normalize_well(well)[0] for well in measured})
    table[PLATE_COLUMN] = table[WELL].map({well: str(int(normalize_well(well)[1:])) for well in measured})
    groups = [*obs_names.values(), CONDITION, WELL, PLATE_ROW, PLATE_COLUMN]
    for name in groups:
        table[name] = _in_natural_order(table[name])

    features = describe(table.select_dtypes("number").columns.tolist(), read_info_file(exp))
    return Cells(table, features, groups, measured, missing)


def load_cells(exp: Experiment) -> Cells:
    """Read every measured well's cells and add their metadata, groups and derived features.

    A well is measured when its Measure cells status is complete; the others are ``missing``, in
    plate order. The groups are the metadata columns, the condition, the well, and its plate row
    and plate column, each an ordered category in natural order. The result is kept and read again
    only when experiment.json, info_exp.csv or a well's cell data file changes, so ``table`` is
    shared between calls: copy it before changing it.
    """
    global _saved
    read_from = _read_from(exp)
    if _saved is not None and _saved[0] == read_from:
        return _saved[1]
    cells = _load(exp)
    _saved = (read_from, cells)
    return cells
