"""@TODO."""

from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, assert_never, cast

import numpy as np
import pandas as pd

from fouri.config import Experiment


class Rule(StrEnum):
    MIDDLE_PERCENTILES = "middle_percentiles"
    ROBUST_SD = "robust_sd"
    BETWEEN = "between"

    @property
    def fields(self) -> list[str]:
        match self:
            case Rule.MIDDLE_PERCENTILES | Rule.BETWEEN:
                return ["low", "high"]
            case Rule.ROBUST_SD:
                return ["sds"]
            case _ as unhandled:
                assert_never(unhandled)


def saved_filters(exp: Experiment) -> list[dict[str, Any]]:
    """@TODO."""
    filters = exp.config.get("app", {}).get("filters")
    if filters is None:
        return []
    names = [str(f.get("name")) for f in filters]
    duplicate_names = [name for name, count in Counter(names).items() if count > 1]
    if duplicate_names:
        raise ValueError(
            "The following filter names are duplicated. Rename them so each rule has a unique name: "
            f"{', '.join(duplicate_names)}"
        )
    return filters


def named_filters(exp: Experiment, names: Sequence[str]) -> list[dict[str, Any]]:
    """@TODO."""
    filters_dict = {f.get("name"): f for f in saved_filters(exp)}
    not_found = [name for name in names if name not in filters_dict]
    if not_found:
        raise ValueError(
            f"Could not find the following saved filters: {', '.join(not_found)}. "
            f"The saved filters are: {', '.join(map(str, filters_dict.keys()))}"
        )
    return [filters_dict[name] for name in names]


def keep(table: pd.DataFrame, rule: Mapping[str, Any]) -> pd.Series:
    """@TODO."""
    kind = _check_rule(table, rule)
    feature = table[rule["feature"]]
    per_sample = rule.get("per_sample", False)
    if not per_sample or kind is Rule.BETWEEN:
        return _keep_group(feature, rule, kind)
    return feature.groupby(table["sample_ID"], observed=True).transform(_keep_group, rule=rule, kind=kind)


def _keep_group(values: pd.Series, rule: Mapping[str, Any], kind: Rule) -> pd.Series:
    values_finite = values[np.isfinite(values)]
    if values_finite.empty:
        return pd.Series(data=False, index=values.index, dtype=bool)

    match kind:
        case Rule.MIDDLE_PERCENTILES:
            low = np.percentile(values_finite, cast(float, rule["low"]))
            high = np.percentile(values_finite, cast(float, rule["high"]))

        case Rule.BETWEEN:
            low = cast(float, rule["low"])
            high = cast(float, rule["high"])

        case Rule.ROBUST_SD:
            sds = cast(float, rule["sds"])
            median = np.median(values_finite)
            mad = np.median(np.abs(values_finite - median))
            robust_sd = 1.4826 * mad
            bound = sds * robust_sd

            low = median - bound
            high = median + bound

    return (values >= low) & (values <= high)


def _check_rule(table: pd.DataFrame, rule: Mapping[str, Any]) -> Rule:
    name = rule.get("name")
    if name is None:
        raise ValueError("A saved filter can't be applied: it has no name.")
    for key in ("rule", "feature"):
        if key not in rule:
            raise ValueError(f"The filter '{name}' can't be applied. It does not have a {key}.")
    try:
        kind = Rule(rule["rule"])
    except ValueError as e:
        raise ValueError(f"Invalid rule name: {rule['rule']}. Must be one of: {(', ').join(r for r in Rule)}") from e

    for key in kind.fields:
        if key not in rule:
            raise ValueError(f"Invalid rule {kind}: it does not have a '{key}' key, which is required.")

    feature = rule["feature"]
    if feature not in table.columns:
        raise ValueError(f"The filter '{name}' needs the {feature} column, which the cell table does not have.")
    per_sample = rule.get("per_sample", False)
    if per_sample and "sample_ID" not in table.columns:
        raise ValueError(
            f"Filter '{name}' is set per sample but there is no 'sample_ID' column is in the cells table."
        )

    low = rule.get("low", -1)
    high = rule.get("high", -1)
    sds = rule.get("sds", -1)
    match kind:
        case Rule.MIDDLE_PERCENTILES if not ((0 <= low <= 100) and (0 <= high <= 100) and (low < high)):
            raise ValueError(
                f"Invalid low/high values for '{kind}' rule. Low and high must be in [0, 100] and "
                f"low must be less than high. Got low={low}, high={high}."
            )
        case Rule.BETWEEN if low > high:
            raise ValueError(
                f"Invalid low/high values for '{kind}' rule. "
                f"Low must be less than or equal to high. Got low={low}, high={high}."
            )
        case Rule.ROBUST_SD if sds <= 0:
            raise ValueError(f"Invalid sds value for '{kind}' rule. sds must be greater than 0. Got sds={sds}")

    return kind


def apply_filters(table: pd.DataFrame, rules: Sequence[Mapping[str, Any]]) -> tuple[pd.Series, dict[str, int]]:
    """@TODO."""
    for rule in rules:
        _check_rule(table, rule)

    included = pd.Series(data=True, index=table.index, dtype=bool)

    counts: dict[str, int] = {}
    for rule in rules:
        kept = keep(table, rule)
        included &= kept
        count = int((~kept).sum())  # count of cells dropped by this rule
        counts[rule["name"]] = count

    included.name = "included"

    return included, counts
