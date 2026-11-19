from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go

from fouri.app.figures import cells_line
from fouri.app.gates import RECTANGLE, _numbers, plotted
from fouri.plots import FONT, GROUP_COLORS

SHOWN_POINTS = 30_000
SEED = 0
NO_CLASS = "no class"
NO_CLASS_COLOR = "#A8A3BC"
GUIDE_COLOR = "#6C6885"
NEW_SHAPE_COLOR = "#6D28D9"
POINT_SIZE = 3
POINT_OPACITY = 0.55
VIEW_PERCENTILES = (0.25, 99.75)
VIEW_PAD = 0.05
TEMPLATE = "simple_white"
MARGIN = {"l": 60, "r": 20, "t": 30, "b": 50}
DRAWING_TOOLS = ["drawclosedpath", "drawrect"]


def class_color(index: int) -> str:
    return GROUP_COLORS[index % len(GROUP_COLORS)]


def _view(values: npt.NDArray[np.float64]) -> tuple[float, float] | None:
    if values.size == 0:
        return None
    low, high = (float(value) for value in np.percentile(values, VIEW_PERCENTILES))
    if not high > low:
        low, high = low - 0.5, high + 0.5
    pad = (high - low) * VIEW_PAD
    return low - pad, high + pad


def plot_ranges(
    table: pd.DataFrame, rule: Mapping[str, Any]
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    x = plotted(table, str(rule["x"]), bool(rule.get("x_log10")))
    y = plotted(table, str(rule["y"]), bool(rule.get("y_log10")))
    drawable = np.isfinite(x) & np.isfinite(y)
    return _view(x[drawable]), _view(y[drawable])


def _finite_points(points: Sequence[Sequence[float]]) -> bool:
    return all(math.isfinite(value) for point in points for value in point)


def _add_gates(fig: go.Figure, gates: Sequence[Mapping[str, Any]], names: list[str]) -> None:
    for gate in gates:
        points = gate.get("points") or []
        if not _finite_points(points):
            continue
        name = str(gate.get("class", "")).strip()
        color = class_color(names.index(name)) if name in names else NO_CLASS_COLOR
        style: dict[str, Any] = {
            "line": {"color": color, "width": 2},
            "fillcolor": "rgba(0,0,0,0)",
            "opacity": 1,
            "layer": "above",
            "editable": False,
            "xref": "x",
            "yref": "y",
        }
        if gate.get("shape") == RECTANGLE and len(points) == 2:
            (x0, y0), (x1, y1) = points
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, **style)
        elif len(points) >= 3:
            path = "M" + "L".join(f"{x},{y}" for x, y in points) + "Z"
            fig.add_shape(type="path", path=path, **style)


def _add_guides(fig: go.Figure, guides: tuple[float, float]) -> None:
    gx, gy = guides
    fig.add_trace(
        go.Scatter(
            x=[gx, gx],
            y=[0, 1],
            yaxis="y2",
            mode="lines",
            line={"color": GUIDE_COLOR, "dash": "dash", "width": 2},
            name=f"Otsu on x ({gx:.3g})",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[gy, gy],
            xaxis="x2",
            mode="lines",
            line={"color": GUIDE_COLOR, "dash": "dot", "width": 2},
            name=f"Otsu on y ({gy:.3g})",
            hoverinfo="skip",
        )
    )
    hidden = {"range": [0, 1], "visible": False, "fixedrange": True}
    fig.update_layout(xaxis2={"overlaying": "x", **hidden}, yaxis2={"overlaying": "y", **hidden})


def gate_figure(
    table: pd.DataFrame,
    rule: Mapping[str, Any],
    classes: pd.Categorical,
    titles: tuple[str, str],
    guides: tuple[float, float] | None,
    covered: int,
    filtered: int,
) -> tuple[go.Figure, str]:
    log_x, log_y = bool(rule.get("x_log10")), bool(rule.get("y_log10"))
    raw_x, raw_y = _numbers(table, str(rule["x"])), _numbers(table, str(rule["y"]))
    with_value = np.isfinite(raw_x) & np.isfinite(raw_y)
    drawable = with_value.copy()
    if log_x:
        drawable &= raw_x > 0
    if log_y:
        drawable &= raw_y > 0
    no_value = len(table) - int(with_value.sum())
    at_or_below = int(with_value.sum()) - int(drawable.sum())
    x, y = plotted(table, str(rule["x"]), log_x), plotted(table, str(rule["y"]), log_y)

    rows = np.flatnonzero(drawable)
    sampled = rows.size > SHOWN_POINTS
    if sampled:
        rows = np.sort(np.random.default_rng(SEED).choice(rows, SHOWN_POINTS, replace=False))
    codes = np.asarray(classes.codes)[rows]
    names = [str(name) for name in classes.categories]

    fig = go.Figure()
    traces = [(name, class_color(index), rows[codes == index]) for index, name in enumerate(names)]
    traces.append((NO_CLASS, NO_CLASS_COLOR, rows[codes < 0]))
    for name, color, picked in traces:
        if name == NO_CLASS and picked.size == 0:
            continue
        fig.add_trace(
            go.Scattergl(
                x=x[picked],
                y=y[picked],
                mode="markers",
                name=name,
                marker={"size": POINT_SIZE, "color": color, "opacity": POINT_OPACITY},
            )
        )
    _add_gates(fig, rule.get("gates") or [], names)
    if guides is not None:
        _add_guides(fig, guides)

    x_range, y_range = _view(x[drawable]), _view(y[drawable])
    fig.update_layout(
        template=TEMPLATE,
        font_family=FONT,
        margin=MARGIN,
        legend_itemsizing="constant",
        legend_title_text="",
        uirevision=f"{rule['x']}|{rule['y']}|{log_x}|{log_y}",
        newshape={"line": {"color": NEW_SHAPE_COLOR, "width": 2}, "fillcolor": "rgba(109,40,217,0.06)"},
        modebar_add=DRAWING_TOOLS,
        dragmode=DRAWING_TOOLS[0],
        xaxis={"title_text": titles[0], "range": list(x_range) if x_range else None},
        yaxis={"title_text": titles[1], "range": list(y_range) if y_range else None},
    )
    line = cells_line(
        int(rows.size),
        len(table) + covered + filtered,
        covered=covered,
        filtered=filtered,
        sampled=sampled,
        no_value=no_value,
        at_or_below=at_or_below,
    )
    return fig, line
