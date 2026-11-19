from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from fouri.app.cells import Cells, blank_as_missing
from fouri.app.state import save_app_value
from fouri.config import Experiment
from fouri.filters import apply_filters, keep, saved_filters
from fouri.labels import saved_labels

CANT_READ = "Filters not applied: the saved filters can't be read."
CANT_APPLY = 'Filters not applied: the filter "{name}" can\'t be applied. {reason}'


def switched_on_filters(exp: Experiment) -> list[str]:
    return [str(rule.get("name")) for rule in saved_filters(exp) if rule.get("use") is True]


def filter_table(table: pd.DataFrame) -> pd.DataFrame:
    if "sample_ID" not in table.columns:
        return table
    return table.assign(sample_ID=blank_as_missing(table["sample_ID"]))


def kept_cells(cells: Cells, exp: Experiment) -> tuple[pd.Series | None, str | None]:
    covered = cells.covered
    any_covered = bool(covered.any())
    uncovered_only = ~covered if any_covered else None
    try:
        names = set(switched_on_filters(exp))
    except ValueError:
        return uncovered_only, CANT_READ
    if not names:
        return uncovered_only, None
    rules = [rule for rule in saved_filters(exp) if str(rule.get("name")) in names]
    table = filter_table(cells.table[~covered] if any_covered else cells.table)
    try:
        included, _ = apply_filters(table, rules)
    except (ValueError, TypeError):
        for rule in rules:
            try:
                keep(table, rule)
            except (ValueError, TypeError) as error:
                return uncovered_only, CANT_APPLY.format(name=rule.get("name"), reason=error)
        raise
    return included.reindex(cells.table.index, fill_value=False), None


def _save_app_key(exp: Experiment, key: str, rules: Sequence[Mapping[str, Any]]) -> Experiment:
    return save_app_value(exp, key, [dict(rule) for rule in rules])


def save_filters(exp: Experiment, rules: Sequence[Mapping[str, Any]]) -> Experiment:
    return _save_app_key(exp, "filters", rules)


def switched_on_labels(exp: Experiment) -> list[str]:
    return [str(label.get("column")) for label in saved_labels(exp) if label.get("use") is True]


def save_labels(exp: Experiment, labels: Sequence[Mapping[str, Any]]) -> Experiment:
    return _save_app_key(exp, "labels", labels)
