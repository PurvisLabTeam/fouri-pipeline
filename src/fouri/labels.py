"""@TODO."""

from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.path import Path as PolygonPath

from fouri.config import Experiment
from fouri.features import threshold_labels


class LabelRule(StrEnum):
    THRESHOLD = "threshold"
    GATES = "gates"

    @property
    def fields(self) -> list[str]:
        match self:
            case LabelRule.THRESHOLD:
                return ["feature", "cutoffs", "classes"]
            case LabelRule.GATES:
                return ["x", "y", "gates"]

    @property
    def columns(self) -> list[str]:
        match self:
            case LabelRule.THRESHOLD:
                return ["feature"]
            case LabelRule.GATES:
                return ["x", "y"]


def saved_labels(exp: Experiment) -> list[dict[str, Any]]:
    """@TODO."""
    labels = exp.config.get("app", {}).get("labels")
    if labels is None:
        return []
    names = [str(label.get("column")) for label in labels]
    duplicate_names = [name for name, count in Counter(names).items() if count > 1]
    if duplicate_names:
        raise ValueError(
            "The following column names are duplicated. Rename them so that each label has a unique name: "
            f"{', '.join(duplicate_names)}"
        )
    return labels


def named_labels(exp: Experiment, names: Sequence[str]) -> list[dict[str, Any]]:
    """@TODO."""
    if len(names) == 0:
        return []
    labels_dict = {str(label.get("column")): label for label in saved_labels(exp)}
    not_found = [col for col in names if col not in labels_dict]
    if not_found:
        raise ValueError(
            f"Could not find the following saved column(s): {', '.join(not_found)}. "
            f"The saved column(s) are: {', '.join(map(str, labels_dict.keys()))}."
        )
    return [labels_dict[name] for name in names]


def label_cells(table: pd.DataFrame, rule: Mapping[str, Any]) -> pd.Categorical:
    """@TODO."""
    rule_kind = _check_rule(table, rule)
    if rule_kind is LabelRule.THRESHOLD:
        return threshold_labels(table[rule["feature"]], cutoffs=rule["cutoffs"], classes=rule["classes"])
    x = table[str(rule["x"])].to_numpy(dtype=float)
    y = table[str(rule["y"])].to_numpy(dtype=float)
    is_finite = np.isfinite(x) & np.isfinite(y)

    result = np.full(len(table), None, dtype=object)

    gates = rule["gates"]
    classes = []
    for gate in gates:
        gate_class = gate["class"]
        if gate_class not in classes:
            classes.append(gate_class)

        shape = gate["shape"]
        points = gate["points"]
        gated = _gate_rect(x, y, points) if shape == "rectangle" else _gate_freehand(x, y, points)
        result_at = gated & is_finite & pd.isna(result)
        result[result_at] = gate_class

    return pd.Categorical(result, categories=classes)


def _gate_rect(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], points: list[list[float]]
) -> npt.NDArray[np.bool]:
    xs = (min(point[0] for point in points), max(point[0] for point in points))
    ys = (min(point[1] for point in points), max(point[1] for point in points))
    return ((x >= xs[0]) & (x <= xs[1])) & ((y >= ys[0]) & (y <= ys[1]))


def _gate_freehand(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], points: list[list[float]]
) -> npt.NDArray[np.bool]:
    path = PolygonPath(points)
    arr = np.column_stack((x, y))
    return path.contains_points(arr)


_CANT_APPLY = "Label '{label}' cannot be applied: "


def _check_rule(table: pd.DataFrame, rule: Mapping[str, Any]) -> LabelRule:
    label = rule.get("column")
    if label is None:
        raise ValueError("A saved label can't be applied because it has no name.")
    for key in ("column", "rule"):
        if key not in rule:
            raise ValueError(f"The label '{label}' can't be applied. It does not have a {key} key.")
    try:
        kind = LabelRule(rule["rule"])
    except ValueError as e:
        raise ValueError(
            f"The label '{label}' can't be applied: {rule['rule']} is not one of {', '.join(r for r in LabelRule)}."
        ) from e

    for key in kind.fields:
        if key not in rule:
            raise ValueError(f"The label '{label}' can't be applied: it has no {key}.")

    for key in kind.columns:
        col = rule[key]
        if col not in table.columns:
            raise ValueError(
                f"The label '{label}' uses '{col}', which is not in the cell table. Check the derived "
                "features on the Experiment page or switch the label off."
            )

    match kind:
        case LabelRule.THRESHOLD:
            _check_threshold(label, rule["cutoffs"], rule["classes"])
        case LabelRule.GATES:
            _check_gates(label, rule["gates"])

    return kind


def _check_threshold(column: str, cutoffs: list[float], classes: list[str]) -> None:
    prefix = _CANT_APPLY.format(label=column)
    n_cutoffs = len(cutoffs)
    n_classes = len(classes)
    if n_cutoffs not in (1, 2):
        raise ValueError(prefix + f"it has {n_cutoffs} cutoffs, but it must have 1 or 2.")
    required = n_cutoffs + 1
    if n_classes != required:
        raise ValueError(prefix + f"it has {n_classes} classes, but {n_cutoffs} cutoffs need {required}.")


def _check_gates(column: str, gates: list[Mapping[str, Any]]) -> None:
    prefix = _CANT_APPLY.format(label=column)
    if len(gates) == 0:
        raise ValueError(prefix + "its list of gates is empty.")
    for gate in gates:
        keys = set(gate.keys())
        required = {"class", "shape", "points"}
        missing = [str(s) for s in (required - keys)]
        if missing:
            raise ValueError(prefix + f"it is missing the following keys: {', '.join(missing)}.")
        shape = gate["shape"]
        points = gate["points"]
        n_points = len(points)
        if shape == "rectangle":
            if n_points != 2:
                raise ValueError(
                    prefix + f"it is a rectangle shape with an invalid number of points (got {n_points}, expected 2)."
                )
        elif shape == "freehand":
            if n_points < 3:
                raise ValueError(
                    prefix + f"it is a freehand shape with an invalid number of points "
                    f"(got {n_points}, expected >= 3)."
                )
        else:
            raise ValueError(prefix + f"it has an invalid shape: '{shape}'. It must be one of rectangle or freehand.")

        for point in points:
            message = prefix + "one of its gates has a point that is not an (x, y) pair."
            try:
                is_finite = np.isfinite(point)
            except TypeError:
                raise ValueError(message) from None
            if len(point) != 2 or not np.all(is_finite):
                raise ValueError(message)


def apply_labels(table: pd.DataFrame, rules: Sequence[Mapping[str, Any]]) -> dict[str, pd.Categorical]:
    """@TODO."""
    for rule in rules:
        _check_rule(table, rule)
        label = rule["column"]
        if label in table.columns:
            raise ValueError(
                f"The label '{label}' would replace a column in the cell table. Rename it on the Labels & gates page."
            )

    results = {}
    for rule in rules:
        label = str(rule["column"])
        results[label] = label_cells(table, rule)

    return results