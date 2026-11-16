from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go

from fouri.app.figures import cells_line
from fouri.plots import FONT, GROUP_COLORS

SHOWN_POINTS = 30_000
SEED = 0
FILL_SHARE = 0.01
RECTANGLE, FREEHAND = "rectangle", "freehand"
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

SPLITS: dict[str, list[tuple[str, tuple[str, str, str, str]]]] = {
    "4 quadrants": [
        ("bottom left", ("x0", "gx", "y0", "gy")),
        ("bottom right", ("gx", "x1", "y0", "gy")),
        ("top left", ("x0", "gx", "gy", "y1")),
        ("top right", ("gx", "x1", "gy", "y1")),
    ],
    "top, and bottom split left / right": [
        ("top", ("x0", "x1", "gy", "y1")),
        ("bottom left", ("x0", "gx", "y0", "gy")),
        ("bottom right", ("gx", "x1", "y0", "gy")),
    ],
    "bottom, and top split left / right": [
        ("bottom", ("x0", "x1", "y0", "gy")),
        ("top left", ("x0", "gx", "gy", "y1")),
        ("top right", ("gx", "x1", "gy", "y1")),
    ],
    "top / bottom": [
        ("bottom", ("x0", "x1", "y0", "gy")),
        ("top", ("x0", "x1", "gy", "y1")),
    ],
    "left / right": [
        ("left", ("x0", "gx", "y0", "y1")),
        ("right", ("gx", "x1", "y0", "y1")),
    ],
}

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def class_color(index: int) -> str:
    return GROUP_COLORS[index % len(GROUP_COLORS)]


def drawn_points(shape: Mapping[str, Any]) -> list[list[float]]:
    if shape.get("type") == "rect":
        return [[float(shape["x0"]), float(shape["y0"])], [float(shape["x1"]), float(shape["y1"])]]
    numbers = [float(number) for number in _NUMBER.findall(str(shape.get("path", "")))]
    points: list[list[float]] = []
    for index in range(0, len(numbers) - 1, 2):
        point = numbers[index : index + 2]
        if not points or point != points[-1]:
            points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def _filled(points: list[list[float]], logged: tuple[bool, bool], spans: tuple[float, float]) -> list[list[float]]:
    steps = [
        FILL_SHARE * span if log and math.isfinite(span) and span > 0 else math.inf
        for log, span in zip(logged, spans, strict=True)
    ]
    filled: list[list[float]] = []
    for start, end in zip(points, points[1:] + points[:1], strict=True):
        filled.append(start)
        count = 1
        for axis, step in enumerate(steps):
            distance = abs(end[axis] - start[axis])
            if math.isfinite(step) and math.isfinite(distance):
                count = max(count, math.ceil(distance / step))
        for part in range(1, count):
            share = part / count
            filled.append([start[0] + (end[0] - start[0]) * share, start[1] + (end[1] - start[1]) * share])
    return filled


def _power(value: float) -> float:
    with np.errstate(over="ignore"):
        return float(np.power(10.0, value))


def _log(value: float) -> float:
    return math.log10(value) if value > 0 else math.nan


def to_saved(
    points: Sequence[Sequence[float]],
    shape: str,
    log_x: bool,
    log_y: bool,
    spans: tuple[float, float],
) -> list[list[float]]:
    drawn = [[float(point[0]), float(point[1])] for point in points]
    if shape == FREEHAND and (log_x or log_y) and len(drawn) > 1:
        drawn = _filled(drawn, (log_x, log_y), spans)
    return [[_power(x) if log_x else x, _power(y) if log_y else y] for x, y in drawn]


def to_drawn(points: Sequence[Sequence[float]], log_x: bool, log_y: bool) -> list[list[float]]:
    return [
        [_log(float(point[0])) if log_x else float(point[0]), _log(float(point[1])) if log_y else float(point[1])]
        for point in points
    ]


def split_gates(
    kind: str,
    cuts: tuple[float, float],
    x_edges: tuple[float, float],
    y_edges: tuple[float, float],
) -> list[dict[str, Any]]:
    edges = {"x0": x_edges[0], "x1": x_edges[1], "gx": cuts[0], "y0": y_edges[0], "y1": y_edges[1], "gy": cuts[1]}
    return [
        {
            "class": name,
            "shape": RECTANGLE,
            "points": [[float(edges[left]), float(edges[bottom])], [float(edges[right]), float(edges[top])]],
        }
        for name, (left, right, bottom, top) in SPLITS[kind]
    ]


def area(points: Sequence[Sequence[float]], shape: str) -> float:
    if shape == RECTANGLE and len(points) == 2:
        (x0, y0), (x1, y1) = points
        return abs((x1 - x0) * (y1 - y0))
    if len(points) < 3:
        return 0.0
    xs = np.array([point[0] for point in points], dtype=np.float64)
    ys = np.array([point[1] for point in points], dtype=np.float64)
    return float(0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))


def _numbers(table: pd.DataFrame, column: str) -> npt.NDArray[np.float64]:
    if column not in table.columns:
        return np.full(len(table), np.nan)
    values = table[column].to_numpy(dtype=np.float64, na_value=np.nan)
    return np.where(np.isfinite(values), values, np.nan)


def plotted(table: pd.DataFrame, column: str, log: bool) -> npt.NDArray[np.float64]:
    values = _numbers(table, column)
    if not log:
        return values
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(values > 0, np.log10(values), np.nan)


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
