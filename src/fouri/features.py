"""Helpers for notebook 07 (cell data -> AnnData)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, zscore
from skimage.filters import threshold_multiotsu, threshold_otsu

from fouri.config import Experiment
from fouri.io import cell_data_path, read_round_table, round_table_path
from fouri.metadata import feature_prefix
from fouri.report import say

DEFAULT_SDS: Final[float] = 3.0

MEASURES: Final[list[str]] = ["nuc_mean", "ring_mean", "total_nuc_protein", "nuc_cyto_ratio"]
_CHANNEL_SUFFIXES = [*MEASURES, "covered"]

_SHAPE_IDENTITY: Final[list[str]] = [
    "label",
    "nuc_area",
    "orientation",
    "axis_major_length",
    "axis_minor_length",
    "ring_area",
    *[f"bbox-{i}" for i in range(4)],
    *[f"centroid-{i}" for i in range(2)],
]

# column names notebook 07 has always used in fullest_df / adata.obs
_OBS_NAMES = {"treatment_time": "treatment time"}

_NO_LABEL = np.array(None)

_OTSU_BINS = 1024


class Threshold(StrEnum):
    OTSU = "Otsu"
    OTSU_2CUT = "Otsu, 2 cut"
    VALLEY = "Valley"


def measured_columns(exp: Experiment) -> list[str]:
    """@TODO."""
    measured = []
    for well in exp.well_list:
        table_path = round_table_path(exp, well)

        if not table_path.exists():
            continue
        table = read_round_table(exp, well)
        channels = [
            f"{feature_prefix(round_name, signal)}_{suffix}"
            for round_name, signal in zip(table["round_name"], table["signal"], strict=True)
            for suffix in _CHANNEL_SUFFIXES
        ]
        measured.extend(channels)

    if measured == []:
        # No well was converted yet.
        return measured

    measured.extend(_SHAPE_IDENTITY)
    return sorted(dict.fromkeys(measured))


def reserved_names(exp: Experiment) -> set[str]:
    return {*measured_columns(exp), *metadata_obs_columns(exp), "sample_ID", "well", "label", "included"}


def build_fullest_df(exp: Experiment, wells: Sequence[str]) -> pd.DataFrame:
    """Concatenate every well's cell_data CSV and add its metadata from the config."""
    frames = []
    metadata = exp.well_metadata
    for well in wells:
        meta = metadata[well]
        df = pd.read_csv(cell_data_path(exp, well), sep=",")
        meta_df = pd.DataFrame(
            {_OBS_NAMES.get(c, c): [v] * len(df) for c, v in meta.items()},
            index=df.index,
        )
        frames.append(pd.concat([df, meta_df], axis=1))
        say(f"Well {well}: {len(frames[-1]):,} cells")
    fullest_df = pd.concat(frames, ignore_index=True)
    say(f"{len(fullest_df):,} cells in {len(frames)} wells")
    for column in metadata_obs_columns(exp):
        if column not in fullest_df.columns:
            fullest_df[column] = np.nan
    fullest_df = fullest_df.drop(columns="Unnamed: 0", errors="ignore")
    fullest_df = fullest_df.astype({col: str for col in metadata_obs_columns(exp) if col != "sample_ID"})
    fullest_df["sample_ID"] = fullest_df["sample_ID"].astype("category")
    return fullest_df


def threshold_labels(values: pd.Series, cutoffs: Sequence[float], classes: Sequence[str]) -> pd.Categorical:
    cutoffs = sorted(cutoffs)
    classes = list(classes)
    n_cutoffs = len(cutoffs)
    finite = np.isfinite(values)
    if n_cutoffs == 1:
        cutoff = cutoffs[0]
        return pd.Categorical(
            np.select([finite & (values < cutoff), finite & (values >= cutoff)], classes, default=_NO_LABEL),
            categories=classes,
        )
    if n_cutoffs == 2:
        cut_lo = cutoffs[0]
        cut_hi = cutoffs[1]
        return pd.Categorical(
            np.select(
                [
                    finite & (values < cut_lo),
                    finite & ((values >= cut_lo) & (values <= cut_hi)),
                    finite & (values > cut_hi),
                ],
                classes,
                default=_NO_LABEL,
            ),
            categories=classes,
        )
    raise ValueError(f"Invalid number of cutoffs: {n_cutoffs}. Use 1 or 2 values only.")


def middle_values(values: npt.NDArray[np.float64], sds: float = DEFAULT_SDS) -> npt.NDArray[np.float64]:
    z_scores = zscore(values, nan_policy="omit")
    mask = np.abs(z_scores) <= sds
    return values[mask]


def _smoothed_curve(values: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    smoothed = gaussian_kde(values)
    positions = np.linspace(start=np.min(values), stop=np.max(values), num=2000)
    heights = smoothed(positions)
    return positions, heights


def curve_range(values: npt.NDArray[np.float64], sds: float = DEFAULT_SDS) -> tuple[float, float]:
    trimmed = middle_values(values, sds)
    positions, heights = _smoothed_curve(trimmed)
    tallest = np.max(heights)
    positions = positions[heights >= 0.01 * tallest]
    return float(positions[0]), float(positions[-1])


def suggest_cutoffs(values: npt.NDArray[np.float64], sds: float = DEFAULT_SDS) -> dict[Threshold, list[float]]:
    values = middle_values(values, sds)

    cutoffs = {}
    cutoffs[Threshold.OTSU] = [float(threshold_otsu(values, nbins=_OTSU_BINS))]
    cutoffs[Threshold.OTSU_2CUT] = [float(val) for val in threshold_multiotsu(values, classes=3, nbins=_OTSU_BINS)]

    positions, heights = _smoothed_curve(values)
    peak_positions, _ = find_peaks(heights)
    peaks = sorted((int(peak) for peak in peak_positions), reverse=True, key=lambda peak: heights[peak])
    if len(peaks) > 1:
        left, right = sorted(peaks[:2])  # the two tallest, left to right
        cutoff_position = left + np.argmin(heights[left : right + 1])
        cutoffs[Threshold.VALLEY] = [float(positions[cutoff_position])]

    return cutoffs


def metadata_obs_columns(exp):
    """fullest_df column names that hold well metadata (go to adata.obs, not .X)."""
    return [_OBS_NAMES.get(c, c) for c in exp.metadata_columns]


def add_derived_features(df, derived_features):
    """Add columns defined as {"name": ..., "formula": ...} (pandas.eval syntax).

    Features are added in order, so a formula can use an earlier one. A feature
    whose formula uses a column that doesn't exist is skipped with a message,
    so experiments without e.g. an S6 antibody don't stop here.
    """
    df = df.copy()  # one contiguous block, so adding columns one by one stays fast
    for spec in derived_features:
        name, formula = spec["name"], spec["formula"]
        try:
            df[name] = df.eval(formula)
            say(f"added {name} = {formula}")
        except (pd.errors.UndefinedVariableError, KeyError) as e:
            say(f"SKIPPED {name} = {formula}  ({e})")
    return df


def standardize_columns(df: pd.DataFrame, include: pd.Series[bool] | None = None) -> pd.DataFrame:
    """z-score every column (sample standard deviation, as before)."""
    include = include if include is not None else pd.Series(True, index=df.index)
    selected = df[include]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Precision loss")
        standardized = stats.zmap(df, selected, ddof=1)
    return pd.DataFrame(data=standardized, columns=df.columns, index=df.index)
