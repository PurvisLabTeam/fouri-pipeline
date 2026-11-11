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
BLANK = "(blank)"

ROW_NUMBER = "Unnamed: 0"  # the row-number column older cell data files have


@dataclass(frozen=True)
class Cells:
    """The cells of the measured wells, with their metadata, groups and derived features."""

    table: pd.DataFrame
    base: pd.DataFrame
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
    blank = values.isna() | values.astype(str).str.strip().eq("")
    categories = sorted(values[~blank].unique(), key=natural_key)
    if blank.any():
        categories.append(BLANK)
    return pd.Categorical(values.where(~blank, BLANK), categories=categories, ordered=True)


def add_groups(table: pd.DataFrame, exp: Experiment) -> list[str]:
    wells = table[WELL].astype(str)
    present = list(dict.fromkeys(wells))
    metadata = exp.well_metadata
    meta = {well: metadata.get(well, {}) for well in present}

    # the metadata under the names the AnnData uses (treatment_time is "treatment time")
    obs_names = dict(zip(exp.metadata_columns, metadata_obs_columns(exp), strict=True))
    for column, name in obs_names.items():
        table[name] = wells.map({well: meta[well].get(column) for well in present})
    groups: list[str] = list(obs_names.values())

    conditions = [c for c in exp.condition_columns if any(not _blank(meta[well].get(c)) for well in present)]
    if len(conditions) >= 2:
        table[CONDITION] = wells.map({well: _condition(meta[well], conditions) for well in present})
        groups.append(CONDITION)

    table[WELL] = wells
    table[PLATE_ROW] = wells.map({well: normalize_well(well)[0] for well in present})
    table[PLATE_COLUMN] = wells.map({well: str(int(normalize_well(well)[1:])) for well in present})
    groups += [WELL, PLATE_ROW, PLATE_COLUMN]
    for name in groups:
        table[name] = _in_natural_order(table[name])
    return groups


def _as_built(base: pd.DataFrame, exp: Experiment) -> pd.DataFrame:
    base = base.drop(columns=ROW_NUMBER, errors="ignore")
    text = {column: str for column in metadata_obs_columns(exp) if column != "sample_ID" and column in base}
    base = base.astype(text)
    if "sample_ID" in base:
        base["sample_ID"] = base["sample_ID"].astype("category")
    return base.copy()


def _load(exp: Experiment) -> Cells:
    status = step_status(exp, Step.MEASURE)
    wells = sorted(status, key=plate_sort_key)
    measured = [well for well in wells if status[well].state == State.COMPLETE]
    missing = [well for well in wells if well not in measured]
    if not measured:
        return Cells(pd.DataFrame(), pd.DataFrame(), {}, [], measured, missing)

    obs_names = dict(zip(exp.metadata_columns, metadata_obs_columns(exp), strict=True))
    metadata = exp.well_metadata
    frames = []
    based = []
    for well in measured:
        frame = pd.read_csv(cell_data_path(exp, well))
        meta = metadata.get(well, {})
        values = {obs_names.get(column, column): [value] * len(frame) for column, value in meta.items()}
        based.append(pd.concat([frame, pd.DataFrame(values, index=frame.index)], axis=1))
        frame = frame.drop(columns=ROW_NUMBER, errors="ignore")
        frame[WELL] = well
        frames.append(frame)
    base = _as_built(pd.concat(based, ignore_index=True), exp)
    table = pd.concat(frames, ignore_index=True)

    # add_derived_features copies the table into one block, so the columns after it add quickly
    table = add_derived_features(table, exp.derived_features)
    groups = add_groups(table, exp)

    features = describe(table.select_dtypes("number").columns.tolist(), read_info_file(exp))
    return Cells(table, base, features, groups, measured, missing)


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
