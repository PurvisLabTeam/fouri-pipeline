"""@TODO."""

import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from fouri.config import Experiment
from fouri.features import MEASURES, metadata_obs_columns, standardize_columns
from fouri.io import COMBAT_LAYER, adata_path, adata_sub_path, fullest_df_path, read_round_table, written_whole
from fouri.metadata import config_to_table, feature_prefix
from fouri.report import say

POSITION_COLUMNS = ["centroid-0", "centroid-1", "bbox-0", "bbox-1", "bbox-2", "bbox-3", "orientation"]
OLD_COLS_TO_DROP = ["nuc_mask", "ring_mask"]

CHANNEL_COLUMNS = ["round_index", "channel_name", "excitation_nm", "emission_nm", "exposure_ms"]
MICROSCOPE_COLUMNS = ["pixel_size_um", "objective", "bit_depth"]
EXPERIMENT_KEYS = ["experiment_key", "experiment_name", "cell_type", "passage", "date_completed", "notes"]


def is_current_subsample(table_uns: Mapping[str, Any], subsample_uns: Mapping[str, Any]) -> bool:
    """@TODO."""
    table_id = table_uns.get("cell_table", {}).get("id")
    subsample_id = subsample_uns.get("subsample", {}).get("cell_table")

    return table_id is not None and subsample_id is not None and table_id == subsample_id


def check_subsample(table_uns: Mapping[str, Any], subsample_uns: Mapping[str, Any]) -> None:
    table_id = table_uns.get("cell_table", {}).get("id")
    if table_id is None:
        raise ValueError("The adata cell table was built by an older version. Build the cell table again.")
    if not is_current_subsample(table_uns, subsample_uns):
        raise ValueError(
            "The subsampled table was made from an outdated cell table. Rerun embedding with overwrite on."
        )


def build_adata(exp: Experiment, fullest_df: pd.DataFrame, labels: Sequence[str] = ()) -> ad.AnnData:
    """@TODO."""
    fullest_df = fullest_df.drop(columns=OLD_COLS_TO_DROP, errors="ignore")
    names = [f"C_{i}" for i in range(len(fullest_df))]

    obs_columns = ["label", "well", "included", *labels, *metadata_obs_columns(exp), *POSITION_COLUMNS]
    obs = fullest_df[[col for col in obs_columns if col in fullest_df.columns]].copy()
    obs.index = pd.Index(names)

    included = (
        fullest_df["included"] if "included" in fullest_df.columns else pd.Series(data=True, index=fullest_df.index)
    )
    numbered = fullest_df.drop(columns=obs.columns).select_dtypes("number")
    features = standardize_columns(numbered, included)
    features_included = features[included]
    numbered_included = numbered[included]
    unusable = [
        col for col in features.columns if features_included[col].isna().any() or numbered_included[col].nunique() <= 1
    ]
    if unusable:
        say(
            "Left out of the AnnData because some cells have unusable values or all cells have the same value:"
            + "".join(f"\n{column}" for column in unusable)
            + "\n"
        )
        features = features.drop(columns=unusable)
    features.index = pd.Index(names)

    round_tables = pd.concat([read_round_table(exp, well) for well in fullest_df["well"].unique()], ignore_index=True)
    adata = ad.AnnData(
        X=features.to_numpy(),
        obs=obs,
        var=_feature_table(exp, features.columns, round_tables),
        uns=_experiment_info(exp, round_tables),
    )

    adata.uns["cell_table"] = {"id": uuid.uuid4().hex}

    df_path = fullest_df_path(exp)
    ad_path = adata_path(exp)
    partial_df = df_path.with_name(f"{df_path.name}.partial")
    partial_adata = ad_path.with_name(f"{ad_path.name}.partial")
    try:
        fullest_df.to_csv(partial_df, index=False)
        adata.write_h5ad(partial_adata)
    except BaseException:
        partial_df.unlink(missing_ok=True)
        partial_adata.unlink(missing_ok=True)
        raise
    _replace_together(partial_df, df_path, partial_adata, ad_path)

    adata_sub = adata_sub_path(exp)
    if adata_sub.exists():
        sub = ad.read_h5ad(adata_sub)
        if COMBAT_LAYER in sub.layers:
            del sub.layers[COMBAT_LAYER]
            with written_whole(adata_sub) as partial:
                sub.write_h5ad(partial)
    say(f"Saved {adata.n_obs:,} cells and {adata.n_vars} features to {adata_path(exp)}")
    return adata


def _not_replaced(path: Path, error: OSError) -> OSError:
    return OSError(
        f"Could not replace {path.name}: {error}. Close any program that has it open and run Build cell table "
        "again; both files are as they were."
    )


def _replace_together(partial_df: Path, df_path: Path, partial_adata: Path, ad_path: Path) -> None:
    previous_df = df_path.with_name(f"{df_path.name}.previous")
    had_df = df_path.exists()
    try:
        if had_df:
            Path.replace(df_path, previous_df)
        try:
            Path.replace(partial_df, df_path)
        except OSError:
            if had_df:
                Path.replace(previous_df, df_path)
            raise
    except OSError as error:
        partial_df.unlink(missing_ok=True)
        partial_adata.unlink(missing_ok=True)
        raise _not_replaced(df_path, error) from error

    try:
        Path.replace(partial_adata, ad_path)
    except OSError as error:
        if had_df:
            Path.replace(previous_df, df_path)
        else:
            df_path.unlink()
        partial_adata.unlink(missing_ok=True)
        raise _not_replaced(ad_path, error) from error
    previous_df.unlink(missing_ok=True)


def _feature_table(exp: Experiment, features: Iterable[str], round_tables: pd.DataFrame) -> pd.DataFrame:
    """Describe each feature: its kind, and for channel features round, signal and acquisition."""
    channels = _same_in_every_well(round_tables, ["round_name", "signal"], CHANNEL_COLUMNS)
    rows = [
        {
            "feature": f"{feature_prefix(str(channel.round_name), str(channel.signal))}_{measure}",
            "kind": "channel",
            "round": channel.round_name,
            "signal": channel.signal,
            "measure": measure,
            **{col: getattr(channel, col) for col in CHANNEL_COLUMNS},
        }
        for channel in channels.itertuples()
        for measure in MEASURES
    ]
    table = pd.DataFrame(rows).set_index("feature").reindex(pd.Index(features))
    derived = {spec["name"] for spec in exp.derived_features}
    table["kind"] = [
        kind if isinstance(kind, str) else ("derived" if name in derived else "shape")
        for name, kind in zip(table.index, table["kind"], strict=True)
    ]
    # text as pandas strings: missing values stay missing, which anndata can save
    text = ["kind", "round", "signal", "measure", "channel_name"]
    return table.astype(dict.fromkeys(text, str) | dict.fromkeys(CHANNEL_COLUMNS[2:], float) | {"round_index": float})


def _experiment_info(exp: Experiment, round_tables: pd.DataFrame) -> dict[str, Any]:
    """Describe the experiment: setup-form details, formulas, plate, microscope, round tables."""
    microscope = _same_in_every_well(round_tables.assign(all="all"), ["all"], MICROSCOPE_COLUMNS)
    plate = config_to_table(exp.well_list, exp.well_metadata)
    text = [col for col in plate.columns if col != "sample_ID"]
    derived = exp.derived_features
    return {
        "experiment": {key: str(exp.config.get(key) or "") for key in EXPERIMENT_KEYS},
        "condition_columns": list(exp.condition_columns),
        "derived_features": pd.DataFrame(
            {"name": [spec["name"] for spec in derived], "formula": [spec["formula"] for spec in derived]},
            dtype=str,
        ),
        "plate": plate.astype(dict.fromkeys(text, str)),
        "microscope": {
            col: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for col, value in microscope.iloc[0].items()
            if col != "all"
        },
        "round_tables": round_tables.astype(dict.fromkeys(round_tables.select_dtypes(["object", "str"]).columns, str)),
    }


def _same_in_every_well(round_tables: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    """One row per group of `keys`; a column that differs between wells is left empty and named."""
    rows = []
    for key, group in round_tables.groupby(keys, sort=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        for col in columns:
            values = group[col].dropna().unique()
            if len(values) > 1:
                say(f"{' '.join(map(str, row.values()))}: {col} differs between wells, left empty")
            row[col] = values[0] if len(values) == 1 else None
        rows.append(row)
    return pd.DataFrame(rows)
