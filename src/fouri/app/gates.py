from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

FILL_SHARE = 0.01
RECTANGLE, FREEHAND = "rectangle", "freehand"

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
