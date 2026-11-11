"""@TODO."""

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd

from fouri import io
from fouri.config import Experiment
from fouri.report import say


class Values(StrEnum):
    MEASURED = "measured"
    ZSCORES = "zscores"
    BOTH = "both"


class FileFormat(StrEnum):
    ANNDATA = "h5ad"
    CSV = "csv"


REFUSED_NAME_CHARACTERS = '\\/:*?"<>|'


def export_cells(
    exp: Experiment,
    name: str,
    *,
    file_format: str = FileFormat.ANNDATA,
    values: str = Values.ZSCORES,
    corrected: bool = False,
    features: Sequence[str] | None = None,
    embeddings: Sequence[str] = (),
    overwrite: bool = False,
) -> Path:
    """@TODO."""
    export_path, file_format = _validate_filename(exp, name, file_format=file_format, overwrite=overwrite)
    values = _validate_values(values, corrected=corrected)

    table_pd = pd.read_csv(io.fullest_df_path(exp))
    table_ad = ad.read_h5ad(io.adata_path(exp))
    cells_pd = _cell_keys(table_pd)
    cells_ad = _cell_keys(table_ad)

    if corrected and "combat" not in table_ad.layers:
        raise ValueError(
            "There is no batch-corrected copy yet. Run Batch effects first or export the z-scores as built."
        )

    chosen_features = _chosen_features(table_pd, table_ad.obs.columns, features)
    subsample = _subsample(exp, embeddings, cells_pd)
    chosen_cells, counts = _chosen_cells(table_pd, cells_pd, table_ad, cells_ad, subsample, corrected=corrected)
    measured, zscores, left_out = _export_values(
        table_pd, cells_pd, table_ad, cells_ad, chosen_cells, chosen_features, values=values, corrected=corrected
    )
    record = {
        "name": name,
        "file_format": str(file_format),
        "values": str(values),
        "corrected": corrected,
        "features": list(features) if features is not None else features,
        "embeddings": list(embeddings),
        "overwrite": overwrite,
        "cells": len(chosen_cells),
        **counts,
        "left_out": left_out,
    }

    final = _anndata(
        table_ad, cells_ad, chosen_cells, chosen_features, measured, zscores, subsample, embeddings, record
    )
    export_path.parent.mkdir(parents=True, exist_ok=True)

    with io.written_whole(export_path) as partial:
        match file_format:
            case FileFormat.ANNDATA:
                final.write_h5ad(partial)
            case FileFormat.CSV:
                csv = _csv(final, values, left_out)
                csv.to_csv(partial, index=False)

    n_final_cells = len(final)
    n_starting_cells = len(subsample) if subsample is not None else len(table_ad)
    filtered = counts["filtered"]
    uncorrected = counts["uncorrected"]
    message = (
        f"Exported {n_final_cells:,} of {n_starting_cells:,}{' subsampled' if embeddings else ''} cells with "
        f"{final.n_vars} feature(s){f' and {len(embeddings)} embedding(s)' if embeddings else ''} to "
        f"exports/{export_path.name}{':\n' if filtered > 0 or (corrected and uncorrected > 0) else '.'}"
    )
    if filtered > 0:
        message += f"\tExcluded by filters: {filtered:,}\n"
    if corrected and uncorrected > 0:
        message += f"\tExcluded because they had no batch correction: {uncorrected:,}"

    say(message)

    return export_path


def _csv(adata: ad.AnnData, values: Values, left_out: list[str]) -> pd.DataFrame:
    obs = cast(pd.DataFrame, adata.obs)
    measured = (
        adata.to_df() if values is Values.MEASURED else adata.to_df("measured") if "measured" in adata.layers else None
    )
    zscores = (
        None
        if values is Values.MEASURED
        else adata.to_df().drop(columns=left_out, errors="ignore").add_suffix("_zscore")
    )
    coordinates = []
    for key, array in adata.obsm.items():
        array = np.asarray(array, dtype=np.float64)
        embed_obsm = pd.DataFrame(
            array, index=adata.obs_names, columns=[f"{key}_{n}" for n in range(1, array.shape[1] + 1)]
        )
        coordinates.append(embed_obsm)

    return pd.concat([obs, measured, zscores, *coordinates], axis=1)


def _anndata(
    adata: ad.AnnData,
    adata_cells: pd.MultiIndex,
    chosen_cells: pd.MultiIndex,
    chosen_features: list[str],
    measured: pd.DataFrame | None,
    zscores: pd.DataFrame | None,
    subsample: ad.AnnData | None,
    embeddings: Sequence[str],
    record: dict[str, Any],
) -> ad.AnnData:
    chosen_idx = adata_cells.get_indexer(chosen_cells)
    obs = cast(pd.DataFrame, adata.obs)
    obs = obs.iloc[chosen_idx]
    obs = obs.drop(columns="included", errors="ignore")

    layers = {}
    match (measured, zscores):
        case (pd.DataFrame(), None):
            x = measured
        case (None, pd.DataFrame()):
            x = zscores
        case (pd.DataFrame(), pd.DataFrame()):
            x = zscores.reindex(columns=chosen_features)
            layers["measured"] = np.asarray(measured)
        case _:
            raise RuntimeError(
                "The export has neither measurements nor z-scores to write. This should never happen. "
                "Please report it as a bug."
            )

    var = cast(pd.DataFrame, adata.var.reindex(x.columns))
    final_adata = ad.AnnData(np.asarray(x), obs=obs, var=var, layers=layers)

    embed_coordinates, records = _embedding_coordinates(subsample, embeddings, chosen_cells)
    final_adata.obsm = embed_coordinates
    uns = dict(adata.uns)
    if records:
        uns["embeddings"] = records

    uns["export"] = record
    final_adata.uns = uns

    return final_adata


def _embedding_coordinates(
    subsample: ad.AnnData | None, embeddings: Sequence[str], chosen_cells: pd.MultiIndex
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, Any]]:
    coordinates: dict[str, npt.NDArray[np.float64]] = {}
    records: dict[str, Any] = {}
    if subsample is None:
        return coordinates, records

    subsample_keys = _cell_keys(subsample)
    for embedding in embeddings:
        table = pd.DataFrame(np.asarray(subsample.obsm[embedding], dtype=np.float64), index=subsample_keys)
        coordinates[embedding] = np.asarray(table.loc[chosen_cells], dtype=np.float64)
        records[embedding] = subsample.uns["embeddings"][embedding]

    return coordinates, records


def _export_values(
    table: pd.DataFrame,
    table_cells: pd.MultiIndex,
    adata: ad.AnnData,
    adata_cells: pd.MultiIndex,
    chosen_cells: pd.MultiIndex,
    features: Sequence[str],
    *,
    values: Values,
    corrected: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[str]]:

    measured: pd.DataFrame | None = None
    zscores: pd.DataFrame | None = None
    left_out: list[str] = []
    match values:
        case Values.MEASURED:
            measured = _measured(table, table_cells, chosen_cells, features)
        case Values.ZSCORES:
            zscores, left_out = _zscores(adata, adata_cells, chosen_cells, features, corrected=corrected)
        case Values.BOTH:
            measured = _measured(table, table_cells, chosen_cells, features)
            zscores, left_out = _zscores(adata, adata_cells, chosen_cells, features, corrected=corrected)
    return measured, zscores, left_out


def _measured(
    table: pd.DataFrame, table_cells: pd.MultiIndex, chosen_cells: pd.MultiIndex, features: Sequence[str]
) -> pd.DataFrame:
    return table.set_axis(table_cells).loc[chosen_cells, features]


def _zscores(
    adata: ad.AnnData,
    adata_cells: pd.MultiIndex,
    chosen_cells: pd.MultiIndex,
    features: Sequence[str],
    *,
    corrected: bool,
) -> tuple[pd.DataFrame, list[str]]:
    chosen_features = [f for f in features if f in adata.var_names]
    left_out = [f for f in features if f not in adata.var_names]
    if left_out:
        say(
            "The following chosen features were left out of z-scores "
            "because the cell table does not have z-scores for them:"
            f"\n{'\n'.join(left_out)}\n"
        )
    by_cell = _values_by_cell(adata, adata_cells, layer="combat" if corrected else None)
    return by_cell.loc[chosen_cells, chosen_features], left_out


def _values_by_cell(adata: ad.AnnData, cells: pd.MultiIndex, layer: str | None = None) -> pd.DataFrame:
    data = np.asarray(adata.X if layer is None else adata.layers[layer], dtype=np.float64)
    cols = adata.var_names
    return pd.DataFrame(data, index=cells, columns=cols)


def _chosen_cells(
    table: pd.DataFrame,
    table_cells: pd.MultiIndex,
    adata: ad.AnnData,
    adata_cells: pd.MultiIndex,
    subsample: ad.AnnData | None,
    *,
    corrected: bool,
) -> tuple[pd.MultiIndex, dict[str, int]]:
    included = (
        table["included"].set_axis(table_cells) if "included" in table.columns else pd.Series(True, index=table_cells)
    )
    if subsample is None:
        cells = included[included].index
        filtered = int((~included).sum())
    else:
        subsample_keys = _cell_keys(subsample)
        included_subsample = included.loc[subsample_keys]
        cells = subsample_keys[included_subsample]
        filtered = int((~included_subsample).sum())
        if filtered > 0:
            say(f"Excluded {filtered:,} subsampled cells that the current filters exclude.")

    if corrected:
        layer = _values_by_cell(adata, adata_cells, layer="combat")
        layer = layer.loc[cells]
        uncorrected_cells = layer.isna().all(axis=1)
        cells = cells[~uncorrected_cells]
        uncorrected = int(uncorrected_cells.sum())
    else:
        uncorrected = 0

    if uncorrected > 0:
        say(f"Dropped {uncorrected:,} cells with no ComBat-corrected values.")

    result = {"filtered": filtered, "uncorrected": uncorrected}

    return cast(pd.MultiIndex, cells), result


def _cell_keys(cells: pd.DataFrame | ad.AnnData) -> pd.MultiIndex:
    if isinstance(cells, ad.AnnData):
        cells = cast(pd.DataFrame, cells.obs)
    return pd.MultiIndex.from_arrays([cells["well"], cells["label"]], names=["well", "label"])


def _chosen_features(table: pd.DataFrame, obs_cols: pd.Index, features: Sequence[str] | None = None) -> list[str]:
    numeric = [col for col in table.select_dtypes(include="number").columns.to_list() if col not in obs_cols]
    if features is None:
        return numeric
    missing = [feature for feature in features if feature not in table.columns]
    if missing:
        raise ValueError(f"The cell table is missing the following chosen columns:\n{'\n'.join(missing)}\n")

    not_numeric = [feature for feature in features if feature not in numeric]
    if not_numeric:
        raise ValueError(f"The following chosen column(s) are not features:\n{'\n'.join(not_numeric)}\n")

    return [col for col in numeric if col in features]


def _subsample(exp: Experiment, embeddings: Sequence[str], table_cells: pd.MultiIndex) -> ad.AnnData | None:
    if len(embeddings) == 0:
        return None
    subsample_file = io.adata_sub_path(exp)
    if not subsample_file.exists():
        raise FileNotFoundError(
            "The subsample file does not exist yet, so there is no embedding to export. Run Embedding first."
        )
    adata_sub = ad.read_h5ad(subsample_file)
    missing = [k for k in embeddings if k not in adata_sub.obsm]
    if missing:
        raise ValueError(f"The subsample file has no embedding(s) named:\n{'\n'.join(missing)}\n")
    cell_keys = _cell_keys(adata_sub)
    missing_cells = cell_keys.difference(table_cells)
    if not missing_cells.empty:
        raise ValueError(
            "The subsample file has cells that the main table no longer has. It was probably made from an "
            "outdated table. Run Embedding with Overwrite to make a new subsample "
            f"(missing {len(missing_cells):,} cells)."
        )

    return adata_sub


def _validate_values(values: str, *, corrected: bool) -> Values:
    try:
        values = Values(values)
    except ValueError as e:
        raise ValueError(f"Invalid value type. Got {values}, expected 'measured', 'zscores', or 'both'.") from e

    if corrected and values not in (Values.ZSCORES, Values.BOTH):
        raise ValueError("Only z-scores can be batch corrected. Choose z-scores or both.")

    return values


def _validate_filename(exp: Experiment, name: str, *, file_format: str, overwrite: bool) -> tuple[Path, FileFormat]:
    try:
        file_format = FileFormat(file_format)
    except ValueError as e:
        raise ValueError(f"Invalid file format for export. Got {file_format}, expected 'h5ad' or 'csv'.") from e

    stripped = name.strip()
    if stripped == "":
        raise ValueError("Please provide a name for the export.")

    if (invalid_char := next((ch for ch in name if ch in REFUSED_NAME_CHARACTERS), None)) is not None:
        raise ValueError(
            f"Invalid character in file name: {invalid_char}. Export file name may not contain "
            f"any of these characters: {', '.join(REFUSED_NAME_CHARACTERS)}"
        )

    if name.endswith((" ", ".")):
        raise ValueError(f"Invalid file name: {name}. File name must not end with a space or a dot.")

    fpath = io.export_path(exp, name, file_format)
    if fpath.exists() and not overwrite:
        raise FileExistsError(
            f"Export file path: exports/{fpath.name} already exists. Choose another name or replace it."
        )

    return fpath, file_format
