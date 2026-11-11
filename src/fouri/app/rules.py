from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from fouri.app.cells import Cells
from fouri.app.state import open_experiment
from fouri.config import Experiment, save_config
from fouri.filters import apply_filters, keep, saved_filters

CANT_READ = "Showing all cells: the saved filters can't be read."
CANT_APPLY = 'Showing all cells: the filter "{name}" can\'t be applied. {reason}'


def switched_on_filters(exp: Experiment) -> list[str]:
    return [str(rule.get("name")) for rule in saved_filters(exp) if rule.get("use") is True]


def kept_cells(cells: Cells, exp: Experiment) -> tuple[pd.Series | None, str | None]:
    try:
        names = set(switched_on_filters(exp))
    except ValueError:
        return None, CANT_READ
    if not names:
        return None, None
    rules = [rule for rule in saved_filters(exp) if str(rule.get("name")) in names]
    try:
        included, _ = apply_filters(cells.table, rules)
    except (ValueError, TypeError):
        for rule in rules:
            try:
                keep(cells.table, rule)
            except (ValueError, TypeError) as error:
                return None, CANT_APPLY.format(name=rule.get("name"), reason=error)
        raise
    return included, None


def save_filters(exp: Experiment, rules: Sequence[Mapping[str, Any]]) -> Experiment:
    with open(exp.config_path, encoding="utf-8") as file:
        config = json.load(file)
    app = config.get("app")
    config["app"] = {**(app if isinstance(app, dict) else {}), "filters": [dict(rule) for rule in rules]}
    save_config(config, exp.config_path)
    return open_experiment(exp.config_path)
