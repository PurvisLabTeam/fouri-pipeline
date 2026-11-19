from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from anndata.io import read_elem

from fouri.app.cells import blank_as_missing, load_cells
from fouri.app.rules import kept_cells
from fouri.app.state import save_app_value
from fouri.batch import (
    PLATE_COLUMN,
    PLATE_ROW,
    BatchMetrics,
    batch_groups,
    batch_labels,
    batch_metrics,
    condition_labels,
    condition_shares,
    conditions_in_batches,
    uneven_conditions,
    used_conditions,
    why_not_combat,
)
from fouri.config import Experiment
from fouri.features import metadata_obs_columns, standardize_columns
from fouri.io import adata_path
from fouri.metadata import plate_sort_key

SETUP_KEY = "batch_effects"
DEFAULT_BATCH = "well"

RECORD_KEY = "batch_effects"
SAMPLE_ID = "sample_ID"
TEXT_KEYS = frozenset({"top", "features", "conditions", "batches"})


@dataclass(frozen=True)
class LiveNumbers:
    cells: int
    used: int
    problem: str | None
    conditions: pd.DataFrame
    why_not: str | None
    shares: pd.DataFrame
    uneven: list[tuple[str, str, str]]
    groups: list[list[str]]
    metrics: BatchMetrics | None
    blank_wells: dict[str, dict[str, int]]


@dataclass(frozen=True)
class SavedCorrection:
    batch: str
    keep: list[str]
    log_transformed: list[str]
    conditions: list[str]
    left_out: list[str]
    groups: list[list[str]]
    uneven: list[tuple[str, str, str]]
    before: dict[str, Any]
    after: dict[str, Any]
    reference: dict[str, Any]
    adjustment: pd.DataFrame


def batch_choices(exp: Experiment) -> list[str]:
    conditions = set(exp.condition_columns)
    technical = [
        name
        for column, name in zip(exp.metadata_columns, metadata_obs_columns(exp), strict=True)
        if column != SAMPLE_ID and column not in conditions
    ]
    return list(dict.fromkeys([DEFAULT_BATCH, PLATE_ROW, PLATE_COLUMN, *technical]))


def keep_choices(exp: Experiment, batch: str) -> list[str]:
    conditions = set(exp.condition_columns)
    return [
        name
        for column, name in zip(exp.metadata_columns, metadata_obs_columns(exp), strict=True)
        if column != SAMPLE_ID and column in conditions and name != batch
    ]


def _text_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


_KEYS: dict[str, tuple[Callable[[Any], bool], str]] = {
    "batch": (lambda value: isinstance(value, str), "text"),
    "keep": (_text_list, "a list of column names"),
    "log_intensities": (lambda value: isinstance(value, bool), "true or false"),
}


def saved_batch_setup(exp: Experiment) -> dict[str, Any] | None:
    app = exp.config.get("app")
    if app is None:
        return None
    if not isinstance(app, dict):
        raise ValueError('"app" in experiment.json is not a set of settings.')
    setup = app.get(SETUP_KEY)
    if setup is None:
        return None
    if not isinstance(setup, dict):
        raise ValueError(f'"{SETUP_KEY}" is not a set of settings.')
    for key, (fits, kind) in _KEYS.items():
        if key not in setup:
            raise ValueError(f'"{key}" is missing from "{SETUP_KEY}".')
        if not fits(setup[key]):
            raise ValueError(f'"{key}" in "{SETUP_KEY}" should be {kind}.')
    return copy.deepcopy(setup)


def save_batch_setup(exp: Experiment, setup: Mapping[str, Any]) -> Experiment:
    return save_app_value(exp, SETUP_KEY, copy.deepcopy(dict(setup)))


def setup_words(batch: str, keep: Sequence[str], log_intensities: bool) -> str:
    kept = ", ".join(keep) if keep else "no column"
    return f"by {batch}{' on log intensities' if log_intensities else ''}, keeping {kept}"


def batch_setup_line(setup: Mapping[str, Any]) -> str:
    words = setup_words(setup["batch"], setup["keep"], setup["log_intensities"])
    return f"{words[:1].upper()}{words[1:]}."


def _zscored(table: pd.DataFrame) -> tuple[npt.NDArray[np.float64], list[str]]:
    standardized = standardize_columns(table)
    unusable = {
        column for column in table.columns if standardized[column].isna().any() or table[column].nunique() <= 1
    }
    features = [column for column in table.columns if column not in unusable]
    return standardized[features].to_numpy(dtype=np.float64), features


def live_numbers(exp: Experiment, batch: str, keep: Sequence[str]) -> LiveNumbers:
    cells = load_cells(exp)
    kept, problem = kept_cells(cells, exp)
    table = cells.table if kept is None else cells.table.loc[kept]
    table = table.reset_index(drop=True)
    columns = [column for column in dict.fromkeys([batch, *keep]) if column in table.columns]
    table = table.assign(**{column: blank_as_missing(table[column]) for column in columns})
    batches = batch_labels(table, batch)
    conditions = condition_labels(table, list(keep))
    blanks = {batch: batches.isna()} | {column: table[column].isna() for column in keep if column in table.columns}
    blank_wells = {column: wells for column, blank in blanks.items() if (wells := _blank_wells(table, blank))}
    in_batches = conditions_in_batches(conditions, batches)
    _, used_mask = used_conditions(conditions, batches)
    used = int(used_mask.sum())
    why_not = why_not_combat(conditions, batches)
    if why_not is not None:
        return LiveNumbers(len(table), used, problem, in_batches, why_not, pd.DataFrame(), [], [], None, blank_wells)
    values, features = _zscored(table[list(cells.features)])
    shares = condition_shares(conditions, batches)
    return LiveNumbers(
        cells=len(table),
        used=used,
        problem=problem,
        conditions=in_batches,
        why_not=None,
        shares=shares,
        uneven=uneven_conditions(shares),
        groups=batch_groups(conditions, batches),
        metrics=batch_metrics(values, features, conditions, batches),
        blank_wells=blank_wells,
    )


def _blank_wells(table: pd.DataFrame, blank: pd.Series) -> dict[str, int]:
    counts = table["well"].astype(str)[blank.to_numpy(dtype=bool)].value_counts()
    return {str(well): int(counts[well]) for well in sorted(counts.index, key=plate_sort_key)}


def _texts(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(item) for item in np.asarray(value, dtype=object).ravel()]


def _plain(value: Any, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {str(name): _plain(item, str(name)) for name, item in value.items()}
    if key in TEXT_KEYS:
        return _texts(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OUS":
            return _texts(value)
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _groups(record: Mapping[str, Any]) -> list[list[str]]:
    groups: dict[int, list[str]] = {}
    for batch, group in zip(_texts(record["batches"]), np.asarray(record["values"]).ravel(), strict=True):
        groups.setdefault(int(group), []).append(batch)
    return list(groups.values())


def _uneven(value: Any) -> list[tuple[str, str, str]]:
    rows = np.asarray(value, dtype=object)
    if rows.size == 0:
        return []
    return [(str(row[0]), str(row[1]), str(row[2])) for row in rows.reshape(-1, 3)]


def saved_correction(exp: Experiment) -> SavedCorrection | None:
    path = adata_path(exp)
    if not path.exists():
        return None
    with h5py.File(path, "r") as file:
        uns = file.get("uns")
        if not isinstance(uns, h5py.Group) or RECORD_KEY not in uns:
            return None
        record: Any = read_elem(uns[RECORD_KEY])
    adjustment = record["adjustment"]
    return SavedCorrection(
        batch=str(record["batch"]),
        keep=_texts(record.get("keep")),
        log_transformed=_texts(record.get("log_transformed")),
        conditions=_texts(record.get("conditions")),
        left_out=_texts(record.get("left_out")),
        groups=_groups(record["groups"]),
        uneven=_uneven(record.get("uneven")),
        before=_plain(record.get("before", {})),
        after=_plain(record.get("after", {})),
        reference=_plain(record.get("reference", {})),
        adjustment=pd.DataFrame(
            np.asarray(adjustment["values"], dtype=np.float64).reshape(
                len(_texts(adjustment["batches"])), len(_texts(adjustment["features"]))
            ),
            index=pd.Index(_texts(adjustment["batches"]), name="batch"),
            columns=_texts(adjustment["features"]),
        ),
    )


def matches(saved: SavedCorrection, setup: Mapping[str, Any]) -> bool:
    return (
        saved.batch == setup["batch"]
        and saved.keep == list(setup["keep"])
        and bool(saved.log_transformed) == bool(setup["log_intensities"])
    )
