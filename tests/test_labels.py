from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from builders import with_labels

from fouri.config import Experiment
from fouri.labels import apply_labels, label_cells, saved_labels


def _threshold(cutoffs: list[float], classes: list[str], feature: str = "x") -> dict[str, Any]:
    return {"column": "x state", "rule": "threshold", "feature": feature, "cutoffs": cutoffs, "classes": classes}


def _gate(name: str, shape: str, points: list[list[float]]) -> dict[str, Any]:
    return {"class": name, "shape": shape, "points": points}


def _gates(*gates: dict[str, Any], x: str = "x", y: str = "y") -> dict[str, Any]:
    return {"column": "phase", "rule": "gates", "x": x, "y": y, "gates": list(gates)}


def _without(fields: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in fields.items() if name != key}


def _cells(*points: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame(list(points), columns=["x", "y"], dtype=float)


def _names(classes: Iterable[Any]) -> list[str | None]:
    return [None if pd.isna(name) else str(name) for name in classes]


SQUARE = _gate("S", "rectangle", [[0, 0], [1, 1]])
TRIANGLE = _gate("G1", "freehand", [[2, 2], [3, 2], [2.5, 3]])
X_LEVEL = {**_threshold([1], ["low", "high"]), "column": "x level"}


def test_one_cutoff_splits_below_and_at_or_above() -> None:
    table = pd.DataFrame({"x": [1.0, 4.9, 5.0, 5.1, 9.0]})

    classes = label_cells(table, _threshold([5], ["low", "high"]))

    assert _names(classes) == ["low", "low", "high", "high", "high"]


@pytest.mark.parametrize("cutoffs", [[2, 4], [4, 2]], ids=["saved-lower-first", "saved-upper-first"])
def test_two_cutoffs_keep_both_bounds_in_the_middle_class(cutoffs: list[float]) -> None:
    table = pd.DataFrame({"x": [1.0, 1.9, 2.0, 3.0, 4.0, 4.1]})

    classes = label_cells(table, _threshold(cutoffs, ["low", "mid", "high"]))

    assert _names(classes) == ["low", "low", "mid", "mid", "mid", "high"]


@pytest.mark.parametrize(
    ("rule", "named", "expected"),
    [
        (_threshold([5], ["zeta", "alpha"]), ["zeta", "alpha"], ["zeta", "zeta", "zeta", "alpha"]),
        (_threshold([2, 4], ["off", "bright", "dim"]), ["off", "bright", "dim"], ["off", "bright", "dim", "dim"]),
        (
            _gates(
                _gate("late", "rectangle", [[0, 0], [1, 1]]),
                _gate("early", "rectangle", [[2, 2], [3, 3]]),
                _gate("late", "rectangle", [[4, 4], [5, 5]]),
                _gate("mid", "rectangle", [[10, 10], [11, 11]]),
            ),
            ["late", "early", "mid"],
            ["late", "early", "late", None],
        ),
    ],
    ids=["one-cutoff", "two-cutoffs", "gates"],
)
def test_the_classes_are_named_and_ordered_as_saved(
    rule: dict[str, Any], named: list[str], expected: list[str | None]
) -> None:
    table = _cells((0.5, 0.5), (2.5, 2.5), (4.5, 4.5), (9, 9))

    classes = label_cells(table, rule)

    assert classes.categories.tolist() == named
    assert _names(classes) == expected


@pytest.mark.parametrize(
    "corners",
    [[[2, 10], [4, 20]], [[4, 20], [2, 10]], [[2, 20], [4, 10]]],
    ids=["lower-left-first", "upper-right-first", "other-diagonal"],
)
def test_a_rectangle_holds_the_cells_between_its_corners(corners: list[list[float]]) -> None:
    table = _cells((2, 10), (4, 20), (3, 15), (4, 10), (1.9, 15), (4.1, 15), (3, 9.9), (3, 20.1))

    classes = label_cells(table, _gates(_gate("S", "rectangle", corners)))

    assert _names(classes) == ["S", "S", "S", "S", None, None, None, None]


def test_a_freehand_gate_holds_the_cells_inside_its_outline() -> None:
    outline = [[0, 0], [10, 0], [10, 10], [5, 2], [0, 10]]
    table = _cells((5, 1), (1, 8), (9, 8), (5, 5), (2, 8), (11, 5))

    classes = label_cells(table, _gates(_gate("G1", "freehand", outline)))

    assert _names(classes) == ["G1", "G1", "G1", None, None, None]


def test_gates_with_one_name_form_one_class() -> None:
    rule = _gates(SQUARE, TRIANGLE, _gate("S", "freehand", [[4, 4], [5, 4], [4.5, 5]]))
    table = _cells((0.5, 0.5), (2.5, 2.4), (4.5, 4.4))

    classes = label_cells(table, rule)

    assert _names(classes) == ["S", "G1", "S"]
    assert classes.categories.tolist() == ["S", "G1"]


@pytest.mark.parametrize("first", ["G1", "S"])
def test_a_cell_in_two_gates_takes_the_first(first: str) -> None:
    square = _gate("G1", "rectangle", [[0, 0], [2, 2]])
    triangle = _gate("S", "freehand", [[1, 1], [4, 1], [1, 4]])
    gates = [square, triangle] if first == "G1" else [triangle, square]
    table = _cells((0.5, 0.5), (1.5, 1.5), (3, 1.5))

    classes = label_cells(table, _gates(*gates))

    assert _names(classes) == ["G1", first, "S"]


def test_a_cell_in_no_gate_has_no_class() -> None:
    table = _cells((5, 5), (0.5, 0.5), (1.5, 0.5), (2.5, 2.4), (2.5, 3.5))

    classes = label_cells(table, _gates(SQUARE, TRIANGLE))

    assert _names(classes) == [None, "S", None, "G1", None]
    assert classes.categories.tolist() == ["S", "G1"]


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        (_threshold([0], ["low", "high"]), ["high", None, None, None, "high", "high", "high"]),
        (_threshold([-5, 5], ["low", "mid", "high"]), ["mid", None, None, None, "mid", "mid", "mid"]),
        (_gates(_gate("in", "rectangle", [[-10, -10], [10, 10]])), ["in", None, None, None, None, None, None]),
        (
            _gates(_gate("in", "freehand", [[-10, -10], [10, -10], [0, 10]])),
            ["in", None, None, None, None, None, None],
        ),
    ],
    ids=["one-cutoff", "two-cutoffs", "rectangle", "freehand"],
)
def test_a_cell_without_a_value_has_no_class(rule: dict[str, Any], expected: list[str | None]) -> None:
    table = _cells((1, 1), (np.nan, 1), (np.inf, 1), (-np.inf, 1), (1, np.nan), (1, np.inf), (1, -np.inf))

    classes = label_cells(table, rule)

    assert _names(classes) == expected


@pytest.mark.parametrize(
    "rule",
    [
        {**_threshold([5], ["low", "high"]), "rule": "quadrant"},
        _without(_threshold([5], ["low", "high"]), "column"),
        _without(_threshold([5], ["low", "high"]), "rule"),
        _without(_threshold([5], ["low", "high"]), "feature"),
        _without(_threshold([5], ["low", "high"]), "cutoffs"),
        _without(_threshold([5], ["low", "high"]), "classes"),
        _without(_gates(SQUARE), "y"),
        _without(_gates(SQUARE), "gates"),
        _threshold([5], ["low", "high"], feature="z"),
        _gates(SQUARE, y="z"),
        _threshold([], ["all"]),
        _threshold([5], ["low", "mid", "high"]),
        _threshold([2, 4], ["low", "high"]),
        _threshold([1, 2, 3], ["a", "b", "c", "d"]),
        _gates(),
        _gates(_without(SQUARE, "class")),
        _gates(_without(SQUARE, "shape")),
        _gates(_without(SQUARE, "points")),
        _gates(_gate("S", "ellipse", [[0, 0], [1, 1]])),
        _gates(_gate("S", "rectangle", [[0, 0], [1, 1], [2, 2]])),
        _gates(_gate("S", "freehand", [[0, 0], [1, 1]])),
        _gates(_gate("S", "rectangle", [[0, 0, 0], [1, 1, 1]])),
        _gates(_gate("S", "freehand", [[0, 0], [2, 0], [0, float("inf")]])),
    ],
    ids=[
        "unknown-kind",
        "missing-column",
        "missing-rule",
        "threshold-missing-feature",
        "threshold-missing-cutoffs",
        "threshold-missing-classes",
        "gates-missing-y",
        "gates-missing-gates",
        "feature-not-in-table",
        "y-not-in-table",
        "no-cutoff",
        "one-cutoff-three-classes",
        "two-cutoffs-two-classes",
        "three-cutoffs",
        "no-gate",
        "gate-without-class",
        "gate-without-shape",
        "gate-without-points",
        "unknown-shape",
        "rectangle-with-three-corners",
        "freehand-with-two-points",
        "point-of-three-numbers",
        "point-not-finite",
    ],
)
def test_a_rule_that_cannot_be_applied_is_refused(rule: dict[str, Any]) -> None:
    table = _cells((0.5, 0.5), (1.5, 1.5))

    with pytest.raises(ValueError):
        label_cells(table, rule)
    with pytest.raises(ValueError):
        apply_labels(table, [X_LEVEL, rule])


@pytest.mark.parametrize("column", ["well", "x"])
def test_a_column_the_table_has_is_refused(column: str) -> None:
    table = _cells((0.5, 0.5), (1.5, 1.5)).assign(well=["B02", "B03"])

    with pytest.raises(ValueError):
        apply_labels(table, [{**X_LEVEL, "column": column}])


def test_two_saved_labels_with_one_column_are_refused(experiment: Experiment) -> None:
    exp = with_labels(experiment, [_threshold([1], ["low", "high"]), _threshold([2], ["below", "above"])])

    with pytest.raises(ValueError):
        saved_labels(exp)
