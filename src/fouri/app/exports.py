from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

from fouri.app.records import done_settings
from fouri.config import Experiment
from fouri.export import REFUSED_NAME_CHARACTERS, FileFormat
from fouri.io import COMBAT_LAYER, adata_path, adata_sub_path, fullest_df_path
from fouri.status import Step

KEY_COLUMNS = ("well", "label")
INCLUDED = "included"
BLANK_NAME = "Name the export."
REFUSED_CHARACTER = f"A name can't hold {' '.join(REFUSED_NAME_CHARACTERS)}"
BAD_ENDING = "A name can't end with a dot or a space."
EARLIER_TABLE = (
    "The subsample was made from an earlier cell table. Run Embedding with Overwrite to make a new one, "
    "or export without embeddings."
)
NO_COMBAT = "There is no batch-corrected copy yet. Run Batch effects first or export the z-scores as built."


@dataclass(frozen=True)
class ExportFile:
    path: Path
    file_format: str
    cells: int | None
    size: int
    written: dt.datetime


@dataclass(frozen=True)
class Counts:
    cells: int
    filtered: int
    uncorrected: int
    subsampled: int | None


def name_problem(name: str) -> str | None:
    if not name.strip():
        return BLANK_NAME
    if any(character in REFUSED_NAME_CHARACTERS for character in name):
        return REFUSED_CHARACTER
    if name.endswith((".", " ")):
        return BAD_ENDING
    return None


def has_corrected(exp: Experiment) -> bool:
    with h5py.File(adata_path(exp), "r") as file:
        layers = file.get("layers")
        return isinstance(layers, h5py.Group) and COMBAT_LAYER in layers


def _keys(wells: Any, labels: Any) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([np.asarray(wells), np.asarray(labels)], names=list(KEY_COLUMNS))


def _obs_keys(file: h5py.File) -> pd.MultiIndex:
    return _keys(read_elem(file["obs"]["well"]), read_elem(file["obs"]["label"]))


def _included(exp: Experiment) -> pd.Series:
    table = pd.read_csv(fullest_df_path(exp), usecols=lambda column: column in {*KEY_COLUMNS, INCLUDED})
    keys = _keys(table["well"], table["label"])
    if INCLUDED not in table.columns:
        return pd.Series(True, index=keys)
    return table[INCLUDED].astype(bool).set_axis(keys)


def _uncorrected(exp: Experiment) -> pd.Series:
    with h5py.File(adata_path(exp), "r") as file:
        layers = file.get("layers")
        if not isinstance(layers, h5py.Group) or COMBAT_LAYER not in layers:
            raise ValueError(NO_COMBAT)
        keys = _obs_keys(file)
        values: Any = read_elem(layers[COMBAT_LAYER])
    if hasattr(values, "toarray"):
        values = values.toarray()
    return pd.Series(np.isnan(np.asarray(values, dtype=np.float64)).all(axis=1), index=keys)


def export_counts(exp: Experiment, *, corrected: bool, embeddings: Sequence[str]) -> Counts:
    included = _included(exp)
    subsampled: int | None = None
    if embeddings:
        with h5py.File(adata_sub_path(exp), "r") as file:
            subsample = _obs_keys(file)
        if not subsample.difference(cast(pd.MultiIndex, included.index)).empty:
            raise ValueError(EARLIER_TABLE)
        chosen = included.loc[subsample]
        subsampled = len(subsample)
    else:
        chosen = included
    filtered = int((~chosen).sum())
    cells = chosen[chosen].index
    uncorrected = 0
    if corrected:
        missing = _uncorrected(exp).loc[cells]
        uncorrected = int(missing.sum())
    return Counts(cells=len(cells) - uncorrected, filtered=filtered, uncorrected=uncorrected, subsampled=subsampled)


def table_filters(exp: Experiment) -> list[str] | None:
    done = done_settings(exp, Step.TABLE)
    if not done:
        return None
    filters = done[-1].get("filters")
    return [str(name) for name in filters] if isinstance(filters, list) else []


def _cells(path: Path, file_format: str) -> int | None:
    try:
        if file_format == FileFormat.ANNDATA:
            with h5py.File(path, "r") as file:
                obs = file["obs"]
                return len(obs[obs.attrs["_index"]])
        return len(pd.read_csv(path, usecols=[0]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def list_exports(exp: Experiment) -> list[ExportFile]:
    folder = Path(exp.exports)
    if not folder.is_dir():
        return []
    formats = {f".{kind.value}": kind.value for kind in FileFormat}
    exports = []
    for path in folder.iterdir():
        file_format = formats.get(path.suffix)
        if file_format is None or not path.is_file():
            continue
        stat = path.stat()
        exports.append(
            ExportFile(
                path=path,
                file_format=file_format,
                cells=_cells(path, file_format),
                size=stat.st_size,
                written=dt.datetime.fromtimestamp(stat.st_mtime),
            )
        )
    return sorted(exports, key=lambda export: export.written, reverse=True)


def export_settings(exp: Experiment, export: ExportFile) -> dict[str, Any] | None:
    matching = [
        settings
        for settings in done_settings(exp, Step.EXPORT)
        if settings.get("name") == export.path.stem and settings.get("file_format") == export.file_format
    ]
    return matching[-1] if matching else None
