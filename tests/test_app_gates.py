from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from fouri.app.gates import FREEHAND, RECTANGLE, split_gates, to_drawn, to_saved
from fouri.labels import label_cells

CORNERS = [[0.5, 1.0], [2.5, 3.0]]
OUTLINE = [[0.0, 0.0], [2.0, 0.0], [2.0, 10.0]]
SPANS = (2.0, 10.0)
NEAR_THE_SLOPE = [(1.0, 4.5), (1.0, 5.5), (0.5, 2.0), (1.5, 7.0), (1.5, 8.0), (2.5, 1.0)]
INSIDE = [True, False, True, True, False, False]
AXES = [(False, False), (True, False), (False, True), (True, True)]


def _own(value: float, log: bool) -> float:
    return float(10.0**value) if log else value


def _on_outline(point: Sequence[float], outline: Sequence[Sequence[float]]) -> bool:
    x, y = point
    for (x0, y0), (x1, y1) in zip(outline, [*outline[1:], outline[0]], strict=True):
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        between = min(x0, x1) - 1e-9 <= x <= max(x0, x1) + 1e-9 and min(y0, y1) - 1e-9 <= y <= max(y0, y1) + 1e-9
        if abs(cross) < 1e-9 and between:
            return True
    return False


def _gated(points: list[list[float]], shape: str, cells: Sequence[tuple[float, float]]) -> list[bool]:
    table = pd.DataFrame(list(cells), columns=["x", "y"], dtype=float)
    gate = {"class": "in", "shape": shape, "points": points}
    rule = {"column": "gate", "rule": "gates", "x": "x", "y": "y", "gates": [gate]}
    return [name == "in" for name in label_cells(table, rule)]


@pytest.mark.parametrize(("log_x", "log_y"), AXES, ids=["linear", "log-x", "log-y", "log-both"])
def test_a_drawn_gate_round_trips_through_its_saved_points(log_x: bool, log_y: bool) -> None:
    rectangle = to_saved(CORNERS, RECTANGLE, log_x, log_y, SPANS)
    outline = to_saved(OUTLINE, FREEHAND, log_x, log_y, SPANS)
    cells = [(_own(x, log_x), _own(y, log_y)) for x, y in NEAR_THE_SLOPE]

    np.testing.assert_allclose(rectangle, [[_own(x, log_x), _own(y, log_y)] for x, y in CORNERS])
    np.testing.assert_allclose(to_drawn(rectangle, log_x, log_y), CORNERS)
    drawn = to_drawn(outline, log_x, log_y)
    corners = [int(np.argmin([np.hypot(x - cx, y - cy) for x, y in drawn])) for cx, cy in OUTLINE]
    np.testing.assert_allclose([drawn[index] for index in corners], OUTLINE, atol=1e-12)
    assert corners == sorted(corners)
    assert all(_on_outline(point, OUTLINE) for point in drawn)
    if not (log_x or log_y):
        assert outline == OUTLINE
    assert _gated(outline, FREEHAND, cells) == INSIDE


@pytest.mark.parametrize(
    ("kind", "classes"),
    [
        ("4 quadrants", ["bottom left", "bottom right", "top left", "top right", None]),
        ("top, and bottom split left / right", ["bottom left", "bottom right", "top", "top", None]),
        ("bottom, and top split left / right", ["bottom", "bottom", "top left", "top right", None]),
        ("top / bottom", ["bottom", "bottom", "top", "top", None]),
        ("left / right", ["left", "right", "left", "right", None]),
    ],
)
def test_split_gates_make_each_quadrant_with_its_class(kind: str, classes: list[str | None]) -> None:
    gates = split_gates(kind, (2.0, 30.0), (0.0, 5.0), (10.0, 50.0))
    table = pd.DataFrame([(1.0, 20.0), (4.0, 20.0), (1.0, 40.0), (4.0, 40.0), (6.0, 20.0)], columns=["x", "y"])

    labels = label_cells(table, {"column": "split", "rule": "gates", "x": "x", "y": "y", "gates": gates})

    assert [None if pd.isna(name) else str(name) for name in labels] == classes
    assert all(gate["shape"] == RECTANGLE for gate in gates)
    if kind == "4 quadrants":
        assert gates == [
            {"class": "bottom left", "shape": RECTANGLE, "points": [[0.0, 10.0], [2.0, 30.0]]},
            {"class": "bottom right", "shape": RECTANGLE, "points": [[2.0, 10.0], [5.0, 30.0]]},
            {"class": "top left", "shape": RECTANGLE, "points": [[0.0, 30.0], [2.0, 50.0]]},
            {"class": "top right", "shape": RECTANGLE, "points": [[2.0, 30.0], [5.0, 50.0]]},
        ]
