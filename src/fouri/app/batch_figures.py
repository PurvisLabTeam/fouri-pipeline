from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from anndata.io import read_elem
from plotly.subplots import make_subplots

from fouri.app.batches import LiveNumbers, SavedCorrection, _texts
from fouri.batch import batch_labels, condition_labels
from fouri.config import Experiment
from fouri.io import COMBAT_LAYER, adata_path
from fouri.metadata import natural_key
from fouri.plots import FONT

BEFORE, AFTER, REFERENCE, CHANCE = "before", "after", "ComBat on shuffled batches", "chance"
VIOLIN_FEATURES = 5
VIOLIN_CELLS = 2_000
ADJUSTED_FEATURES = 15
BEFORE_COLOR = "#A8A3BC"
AFTER_COLOR = "#7C3AED"
REFERENCE_COLOR = "#0F9E8E"
ADJUSTMENT_SCALE = "RdBu_r"
CHANCE_COLOR = "#6C6885"
CHANCE_MARGIN = 0.1
TEMPLATE = "simple_white"
SHARES_TITLE = "Variance explained by batch, 15 most affected features"
CLASSIFIER_TITLE = "Batches told apart, per condition"
VIOLIN_TITLE = "Values per batch, five most affected features"
ADJUSTMENT_TITLE = "ComBat's adjustment, 15 most adjusted features"


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
