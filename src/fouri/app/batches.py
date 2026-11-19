from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from anndata.io import read_elem
from plotly.subplots import make_subplots

from fouri.app.cells import blank_as_missing, load_cells
from fouri.app.rules import kept_cells
from fouri.app.state import open_experiment
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
from fouri.config import Experiment, save_config
from fouri.features import metadata_obs_columns, standardize_columns
from fouri.io import COMBAT_LAYER, adata_path
from fouri.metadata import natural_key, plate_sort_key
from fouri.plots import FONT

SETUP_KEY = "batch_effects"
DEFAULT_BATCH = "well"
VIOLIN_FEATURES = 5
VIOLIN_CELLS = 2_000
ADJUSTED_FEATURES = 15
BEFORE_COLOR = "#A8A3BC"
AFTER_COLOR = "#7C3AED"
REFERENCE_COLOR = "#0F9E8E"
ADJUSTMENT_SCALE = "RdBu_r"

RECORD_KEY = "batch_effects"
SAMPLE_ID = "sample_ID"
CHANCE_COLOR = "#6C6885"
CHANCE_MARGIN = 0.1
TEMPLATE = "simple_white"
TEXT_KEYS = frozenset({"top", "features", "conditions", "batches"})

SHARES_TITLE = "Variance explained by batch, 15 most affected features"
CLASSIFIER_TITLE = "Batches told apart, per condition"
VIOLIN_TITLE = "Values per batch, five most affected features"
ADJUSTMENT_TITLE = "ComBat's adjustment, 15 most adjusted features"
BEFORE, AFTER, REFERENCE, CHANCE = "before", "after", "ComBat on shuffled batches", "chance"


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
    with open(exp.config_path, encoding="utf-8") as file:
        config = json.load(file)
    app = config.get("app")
    config["app"] = {**(app if isinstance(app, dict) else {}), SETUP_KEY: copy.deepcopy(dict(setup))}
    save_config(config, exp.config_path)
    return open_experiment(exp.config_path)


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


def _layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        font_family=FONT,
        font_size=14,
        title_text=title,
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1, "xanchor": "right"},
    )
    return fig


def _by_name(entry: Mapping[str, Any], names_key: str, values_key: str) -> pd.Series:
    return pd.Series(np.asarray(entry[values_key], dtype=np.float64).ravel(), index=_texts(entry[names_key]))


def shares_figure(live: LiveNumbers, saved: SavedCorrection | None, names: Mapping[str, str]) -> go.Figure:
    fig = go.Figure()
    if live.metrics is not None:
        top = list(live.metrics.top)
        titles = [names.get(column, column) for column in top]
        fig.add_trace(
            go.Bar(
                name=BEFORE,
                x=titles,
                y=live.metrics.shares.reindex(top).to_numpy(),
                marker_color=BEFORE_COLOR,
                hovertemplate="%{x}: %{y:.2%}<extra>before</extra>",
            )
        )
        if saved is not None and saved.after:
            after = _by_name(saved.after, "features", "shares")
            fig.add_trace(
                go.Bar(
                    name=AFTER,
                    x=titles,
                    y=after.reindex(top).to_numpy(),
                    marker_color=AFTER_COLOR,
                    hovertemplate="%{x}: %{y:.2%}<extra>after</extra>",
                )
            )
    _layout(fig, SHARES_TITLE)
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category", tickangle=-40)
    fig.update_yaxes(tickformat=".1%", rangemode="tozero")
    return fig


def classifier_figure(live: LiveNumbers, saved: SavedCorrection | None) -> go.Figure:
    fig = go.Figure()
    low = 0.0
    if live.metrics is not None:
        scores = live.metrics.classifier_by_condition
        conditions = [str(condition) for condition in scores.index]
        bars = [(BEFORE, scores.reindex(conditions), BEFORE_COLOR)]
        if saved is not None and saved.after:
            bars.append((AFTER, _by_name(saved.after["classifier_by_condition"], "conditions", "values"), AFTER_COLOR))
        if saved is not None and saved.reference:
            reference = _by_name(saved.reference["classifier_by_condition"], "conditions", "values")
            bars.append((REFERENCE, reference, REFERENCE_COLOR))
        lowest: list[float] = []
        for name, values, color in bars:
            shown = values.reindex(conditions).to_numpy(dtype=np.float64)
            lowest += [float(value) for value in shown if np.isfinite(value)]
            fig.add_trace(
                go.Bar(
                    name=name,
                    x=conditions,
                    y=shown,
                    marker_color=color,
                    texttemplate="%{y:.0%}",
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=f"%{{x}}: %{{y:.0%}}<extra>{name}</extra>",
                )
            )
        chance = live.metrics.chance_by_condition.reindex(conditions).dropna()
        if chance.nunique() == 1:
            level = float(chance.iloc[0])
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color=CHANCE_COLOR,
                line_width=1.5,
                annotation_text=f"{CHANCE} {level:.0%}",
                annotation_position="top left",
            )
        elif not chance.empty:
            fig.add_trace(
                go.Scatter(
                    name=CHANCE,
                    x=list(chance.index),
                    y=chance.to_numpy(),
                    mode="markers",
                    marker={"symbol": "line-ew", "size": 60, "line": {"width": 2, "color": CHANCE_COLOR}},
                    hovertemplate="%{x}: %{y:.0%}<extra>chance</extra>",
                )
            )
        lowest += [float(value) for value in chance.to_numpy()]
        if lowest:
            low = max(0.0, min(lowest) - CHANCE_MARGIN)
    _layout(fig, CLASSIFIER_TITLE)
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.05)
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="balanced accuracy", range=[low, 1.08], tickformat=".0%")
    return fig


def _columns(entry: h5py.Dataset | h5py.Group, positions: list[int]) -> npt.NDArray[np.float64]:
    order = np.argsort(positions)
    ascending = [positions[index] for index in order]
    if isinstance(entry, h5py.Dataset):
        values = np.asarray(entry[:, ascending], dtype=np.float64)
    else:
        whole: Any = read_elem(entry)
        if hasattr(whole, "toarray"):
            whole = whole.toarray()
        values = np.asarray(whole, dtype=np.float64)[:, ascending]
    result = np.empty_like(values)
    result[:, order] = values
    return result


def violin_figure(exp: Experiment, saved: SavedCorrection, condition: str, names: Mapping[str, str]) -> go.Figure:
    shares = _by_name(saved.before, "features", "shares")
    features = [str(feature) for feature in shares.nlargest(VIOLIN_FEATURES).index]
    with h5py.File(adata_path(exp), "r") as file:
        var = file["var"]
        var_names = _texts(read_elem(var[var.attrs["_index"]]))
        positions = [var_names.index(feature) for feature in features]
        obs = cast(pd.DataFrame, read_elem(file["obs"]))
        before = _columns(file["X"], positions)
        after = _columns(file["layers"][COMBAT_LAYER], positions)
    obs = obs.reset_index(drop=True)
    included = obs["included"].to_numpy(dtype=bool) if "included" in obs.columns else np.ones(len(obs), dtype=bool)
    batches = batch_labels(obs, saved.batch).astype(str).to_numpy()
    in_condition = condition_labels(obs, saved.keep).eq(condition).fillna(False).to_numpy(dtype=bool)
    chosen = included & in_condition & ~np.isnan(after).any(axis=1)
    rng = np.random.default_rng(0)
    order = sorted(set(batches[chosen]), key=natural_key)
    rows = []
    for batch in order:
        in_batch = np.flatnonzero(chosen & (batches == batch))
        if len(in_batch) > VIOLIN_CELLS:
            in_batch = np.sort(rng.choice(in_batch, size=VIOLIN_CELLS, replace=False))
        rows.append(in_batch)
    drawn = np.concatenate(rows) if rows else np.array([], dtype=int)
    labels = batches[drawn].tolist()

    fig = make_subplots(rows=1, cols=len(features), subplot_titles=[names.get(f, f) for f in features])
    for column in range(len(features)):
        for name, values, color in ((BEFORE, before, BEFORE_COLOR), (AFTER, after, AFTER_COLOR)):
            fig.add_trace(
                go.Violin(
                    name=name,
                    x=labels,
                    y=values[drawn, column],
                    legendgroup=name,
                    offsetgroup=name,
                    showlegend=column == 0,
                    line_color=color,
                    points=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=column + 1,
            )
    _layout(fig, VIOLIN_TITLE)
    fig.update_layout(violinmode="group")
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=order)
    fig.update_yaxes(title_text="z-score", row=1, col=1)
    return fig


def adjustment_figure(saved: SavedCorrection, names: Mapping[str, str]) -> go.Figure:
    adjustment = saved.adjustment
    features = [str(feature) for feature in adjustment.abs().mean(axis=0).nlargest(ADJUSTED_FEATURES).index]
    fig = go.Figure(
        go.Heatmap(
            z=adjustment[features].to_numpy(),
            x=[names.get(feature, feature) for feature in features],
            y=[str(batch) for batch in adjustment.index],
            colorscale=ADJUSTMENT_SCALE,
            zmid=0,
            colorbar={"title": {"text": "mean adjustment (SD)"}},
            hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>",
        )
    )
    _layout(fig, ADJUSTMENT_TITLE)
    fig.update_xaxes(type="category", tickangle=-40)
    fig.update_yaxes(type="category", autorange="reversed")
    return fig
