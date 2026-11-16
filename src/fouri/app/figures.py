from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from fouri.app.cells import WELL, Cells
from fouri.app.names import natural_key
from fouri.plots import FONT, GROUP_COLORS, VALUE_SCALE


class Kind(StrEnum):
    HISTOGRAM = "Histogram"
    SCATTER = "Scatter"
    DENSITY = "Density"
    VIOLIN = "Violin plot"
    WELL_MAP = "Well map"


@dataclass(frozen=True)
class Choices:
    kind: Kind
    x: str
    y: str | None = None
    color: str | None = None
    group: str | None = None
    split: str | None = None
    small_plots: bool = False
    log_x: bool = False
    log_y: bool = False
    hide_outliers: bool = False
    bins_x: int = 100
    bins_y: int = 100
    max_points: int = 20000
    well: str | None = None


TWO_AXES = frozenset({Kind.SCATTER, Kind.DENSITY})
COLORED = frozenset({Kind.HISTOGRAM, Kind.SCATTER})
COLORED_BY_VALUE = frozenset({Kind.SCATTER})
GROUPED = frozenset({Kind.HISTOGRAM, Kind.SCATTER, Kind.DENSITY, Kind.VIOLIN})
SPLIT = frozenset({Kind.VIOLIN})
SMALL_PLOTS = frozenset({Kind.HISTOGRAM, Kind.SCATTER, Kind.DENSITY})
BINNED_X = frozenset({Kind.HISTOGRAM, Kind.DENSITY})
BINNED_Y = frozenset({Kind.DENSITY})
SAMPLED = frozenset({Kind.SCATTER, Kind.WELL_MAP})

KEPT_PERCENTILES = (0.5, 99.5)
SEED = 0
SMALL_PLOTS_PER_ROW = 3
OVERLAID_OPACITY = 0.55
POINT_OPACITY = 0.35
POINT_SIZE = 3
TEMPLATE = "simple_white"
MARGIN = {"l": 60, "r": 20, "t": 30, "b": 50}

ROW, COLUMN = "centroid-0", "centroid-1"
X, Y, VALUE, PLACE_ROW, PLACE_COLUMN = "__x", "__y", "__value", "__row", "__column"
LOG = "log10 "
DENSITY_SCALE_TITLE = "cells per bin"
MAP_AXIS_TITLES = {PLACE_COLUMN: "x (px)", PLACE_ROW: "y (px)"}


def _title(cells: Cells, column: str, log: bool) -> str:
    feature = cells.features.get(column)
    title = feature.title if feature is not None else column
    return LOG + title if log else title


def _numbers(table: pd.DataFrame, column: str) -> npt.NDArray[np.float64]:
    return table[column].to_numpy(dtype=np.float64, na_value=np.nan)


def _order(table: pd.DataFrame, column: str, present: set[str]) -> list[str]:
    values = table[column]
    if isinstance(values.dtype, pd.CategoricalDtype):
        names = [str(value) for value in values.cat.categories]
    else:
        names = sorted({str(value) for value in values.dropna()}, key=natural_key)
    return [name for name in names if name in present]


def _bins(values: pd.Series, count: int) -> dict[str, float]:
    low, high = float(values.min()), float(values.max())
    size = (high - low) / max(count, 1) * (1 + 1e-9)
    if not size > 0:
        size = 1.0
    return {"start": low, "end": high, "size": size}


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}" if count == 1 else f"{count:,} {word}s"


def cells_line(
    drawn: int,
    total: int,
    *,
    covered: int = 0,
    filtered: int = 0,
    not_picked: int = 0,
    sampled: bool = False,
    no_value: int = 0,
    at_or_below: int = 0,
    outliers: int = 0,
    beyond_sds: int = 0,
    sds: float = 3,
) -> str:
    count = f"{drawn:,} of {total:,} cells" if drawn < total else _plural(total, "cell")
    hidden = []
    if no_value:
        hidden.append(f"{no_value:,} with no value")
    if at_or_below:
        hidden.append(f"{at_or_below:,} at or below 0")
    if outliers:
        hidden.append(_plural(outliers, "outlier"))
    if beyond_sds:
        hidden.append(f"{beyond_sds:,} beyond {sds:g} {'SD' if sds == 1 else 'SDs'}")
    reasons = [f"{covered:,} under artifact masks"] if covered else []
    if filtered:
        reasons.append(f"{filtered:,} filtered")
    if not_picked:
        reasons.append(f"{not_picked:,} in samples not picked")
    if sampled:
        reasons.append("random sample")
    if hidden:
        reasons.append(", ".join(hidden) + " hidden")
    return f"{count} ({'; '.join(reasons)})" if reasons else count


def explore_figure(cells: Cells, choices: Choices, kept: pd.Series | None = None) -> tuple[go.Figure, str]:
    kind = choices.kind
    table = cells.table
    if kind is Kind.WELL_MAP:
        table = table[table[WELL].astype(str) == str(choices.well)]
    total = len(table)
    covered = filtered = 0
    if kept is not None:
        inside = kept.loc[table.index].to_numpy(dtype=bool)
        covered = int((cells.covered.loc[table.index].to_numpy(dtype=bool) & ~inside).sum())
        filtered = total - int(inside.sum()) - covered
        table = table[inside]
    count = len(table)

    columns = {X: choices.x}
    logged = {X: choices.log_x}
    if kind in TWO_AXES:
        if choices.y is None:
            raise ValueError(f"{kind} needs a feature for y")
        columns[Y] = choices.y
        logged[Y] = choices.log_y
    color = choices.color if kind in COLORED else None
    color_group = color if color in cells.groups else None
    if kind in COLORED_BY_VALUE and color is not None and color in cells.features:
        columns[VALUE] = color

    values = {name: _numbers(table, column) for name, column in columns.items()}
    keep = np.logical_and.reduce([np.isfinite(numbers) for numbers in values.values()])
    no_value = count - int(keep.sum())

    with_value = int(keep.sum())
    for name, log in logged.items():
        if log:
            keep &= values[name] > 0
    at_or_below = with_value - int(keep.sum())
    for name, log in logged.items():
        if log:
            values[name] = np.log10(values[name], out=np.full(count, np.nan), where=keep)

    outliers = 0
    if choices.hide_outliers and keep.any():
        inside = keep.copy()
        for name in logged:
            low, high = np.percentile(values[name][keep], KEPT_PERCENTILES)
            inside &= (values[name] >= low) & (values[name] <= high)
        outliers = int(keep.sum()) - int(inside.sum())
        keep = inside

    facet = choices.group if choices.small_plots and kind in SMALL_PLOTS else None
    violin_group = choices.group if kind is Kind.VIOLIN else None
    split = choices.split if kind in SPLIT else None
    groups = [name for name in dict.fromkeys([color_group, facet, violin_group, split]) if name is not None]

    frame = pd.DataFrame({name: numbers[keep] for name, numbers in values.items()})
    for name in groups:
        frame[name] = table[name].astype(str).to_numpy()[keep]
    if kind is Kind.WELL_MAP:
        frame[PLACE_ROW] = _numbers(table, ROW)[keep]
        frame[PLACE_COLUMN] = _numbers(table, COLUMN)[keep]

    sampled = kind in SAMPLED and len(frame) > choices.max_points
    if sampled:
        frame = frame.sample(n=choices.max_points, random_state=SEED).sort_index()

    labels = {name: _title(cells, columns[name], logged.get(name, False)) for name in columns}
    orders = {name: _order(table, name, set(frame[name])) for name in groups}
    common: dict[str, Any] = {"labels": labels, "category_orders": orders, "template": TEMPLATE}
    facets: dict[str, Any] = {"facet_col": facet, "facet_col_wrap": SMALL_PLOTS_PER_ROW} if facet else {}

    if kind is Kind.HISTOGRAM:
        fig = px.histogram(
            frame,
            x=X,
            color=color_group,
            barmode="overlay",
            opacity=OVERLAID_OPACITY if color_group else None,
            histnorm="probability density" if color_group else None,
            color_discrete_sequence=GROUP_COLORS,
            **facets,
            **common,
        )
        if len(frame):
            fig.update_traces(xbins=_bins(frame[X], choices.bins_x))
    elif kind is Kind.SCATTER:
        fig = px.scatter(
            frame,
            x=X,
            y=Y,
            color=VALUE if VALUE in columns else color_group,
            opacity=POINT_OPACITY,
            render_mode="webgl",
            color_continuous_scale=VALUE_SCALE,
            color_discrete_sequence=GROUP_COLORS,
            **facets,
            **common,
        )
        fig.update_traces(marker_size=POINT_SIZE)
    elif kind is Kind.DENSITY:
        fig = px.density_heatmap(frame, x=X, y=Y, color_continuous_scale=VALUE_SCALE, **facets, **common)
        if len(frame):
            fig.update_traces(xbins=_bins(frame[X], choices.bins_x), ybins=_bins(frame[Y], choices.bins_y))
        fig.update_layout(coloraxis_colorbar_title_text=DENSITY_SCALE_TITLE)
    elif kind is Kind.VIOLIN:
        fig = px.violin(
            frame,
            x=violin_group,
            y=X,
            color=split or violin_group,
            box=True,
            points=False,
            violinmode="group" if split else "overlay",
            color_discrete_sequence=GROUP_COLORS,
            **common,
        )
    else:
        fig = px.scatter(
            frame,
            x=PLACE_COLUMN,
            y=PLACE_ROW,
            color=X,
            render_mode="webgl",
            color_continuous_scale=VALUE_SCALE,
            labels={**labels, **MAP_AXIS_TITLES},
            template=TEMPLATE,
        )
        fig.update_traces(marker_size=POINT_SIZE)
        fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)

    if facet:
        fig.for_each_annotation(lambda note: note.update(text=note.text.split("=", 1)[-1]))
    fig.update_layout(
        font_family=FONT,
        margin=MARGIN,
        legend_title_text="",
        legend_itemsizing="constant",
    )
    line = cells_line(
        len(frame),
        total,
        covered=covered,
        filtered=filtered,
        sampled=sampled,
        no_value=no_value,
        at_or_below=at_or_below,
        outliers=outliers,
    )
    return fig, line
