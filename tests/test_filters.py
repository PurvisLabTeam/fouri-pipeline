from typing import Any

import numpy as np
import pandas as pd
import pytest
from builders import with_filters

from fouri.config import Experiment
from fouri.filters import apply_filters, keep, named_filters, saved_filters


def _rule(kind: str, **fields: Any) -> dict[str, Any]:
    return {"name": f"x {kind}", "use": True, "rule": kind, "feature": "x", **fields}


IN_RANGE = {**_rule("between", low=0, high=10), "name": "x in range"}


@pytest.mark.parametrize(
    ("values", "low", "high", "kept"),
    [(range(11), 10, 90, range(1, 10)), (range(10), 25, 75, range(3, 7))],
    ids=["percentile-on-a-cell", "percentile-between-cells"],
)
def test_middle_percentiles_keeps_the_cells_between_the_two_percentiles(
    values: range, low: float, high: float, kept: range
) -> None:
    table = pd.DataFrame({"x": np.random.default_rng(0).permutation(np.array(values, dtype=float))})

    result = keep(table, _rule("middle_percentiles", low=low, high=high))

    assert table.loc[result, "x"].sort_values().tolist() == list(kept)


def test_robust_sd_keeps_cells_within_that_many_scaled_mads_of_the_median() -> None:
    table = pd.DataFrame({"x": [10.0, 12, 9, 11, 10, 8, 13, 15, 30, -2, 16]})

    result = keep(table, _rule("robust_sd", sds=2))

    assert table.loc[~result, "x"].tolist() == [30, -2]


def test_between_keeps_both_bounds() -> None:
    table = pd.DataFrame({"x": [1.0, 2, 3, 4, 5]})

    result = keep(table, _rule("between", low=2, high=4))

    assert result.tolist() == [False, True, True, True, False]


def test_per_sample_judges_each_sample_on_its_own_cells() -> None:
    table = pd.DataFrame(
        {"x": [*range(0, 11), *range(100, 201, 10)], "sample_ID": pd.Categorical([1] * 11 + [2] * 11)}
    ).astype({"x": float})
    middle = _rule("middle_percentiles", low=10, high=90)
    between = _rule("between", low=5, high=150)

    per_sample = keep(table, {**middle, "per_sample": True})
    over_both = keep(table, middle)

    assert table.loc[per_sample, "x"].tolist() == [*range(1, 10), *range(110, 191, 10)]
    assert per_sample.tolist() != over_both.tolist()
    assert keep(table, {**between, "per_sample": True}).tolist() == keep(table, between).tolist()


@pytest.mark.parametrize(
    ("rule", "kept"),
    [
        (_rule("middle_percentiles", low=0, high=100), [1, 2, 3, 4, 5, 6, 7, 8]),
        (_rule("robust_sd", sds=1), [2, 3, 4, 5, 6, 7]),
        (_rule("between", low=0, high=100), [1, 2, 3, 4, 5, 6, 7, 8]),
    ],
    ids=["middle_percentiles", "robust_sd", "between"],
)
def test_a_cell_without_a_value_is_left_out_and_counted(rule: dict[str, Any], kept: list[float]) -> None:
    table = pd.DataFrame({"x": [1.0, 2, np.nan, 3, 4, np.inf, 5, 6, 7, 8, np.inf, np.inf]})

    included, counts = apply_filters(table, [rule])

    assert table.loc[included, "x"].tolist() == kept
    assert counts == {rule["name"]: len(table) - len(kept)}


def test_each_rule_counts_on_every_cell_and_included_needs_every_rule() -> None:
    table = pd.DataFrame({"x": np.arange(11, dtype=float)})
    up_to_7 = {**_rule("between", low=0, high=7), "name": "up to 7"}
    lower_half = {**_rule("middle_percentiles", low=0, high=50), "name": "lower half"}

    for rules in ([up_to_7, lower_half], [lower_half, up_to_7]):
        included, counts = apply_filters(table, rules)

        assert counts == {"up to 7": 3, "lower half": 5}
        assert included.tolist() == [x <= 5 for x in range(11)]
        assert included.name == "included"


def test_no_rules_include_every_cell() -> None:
    table = pd.DataFrame({"x": [1.0, np.nan, 3.0]}, index=[10, 11, 12])

    included, counts = apply_filters(table, [])

    assert included.tolist() == [True, True, True]
    assert included.index.equals(table.index)
    assert included.name == "included"
    assert counts == {}


@pytest.mark.parametrize(
    "rule",
    [
        _rule("top_cells", low=1, high=99),
        _rule("middle_percentiles", low=1),
        {**_rule("robust_sd", sds=3), "feature": "y"},
        _rule("middle_percentiles", low=-1, high=99),
        _rule("middle_percentiles", low=1, high=101),
        _rule("middle_percentiles", low=50, high=50),
        _rule("robust_sd", sds=0),
        _rule("between", low=5, high=1),
        _rule("between", low=0, high=10, per_sample=True),
    ],
    ids=[
        "unknown-kind",
        "missing-field",
        "feature-not-in-table",
        "percentile-below-0",
        "percentile-above-100",
        "low-not-below-high",
        "sds-not-above-0",
        "between-low-above-high",
        "per-sample-without-sample_ID",
    ],
)
def test_a_rule_that_cannot_be_applied_is_refused(rule: dict[str, Any]) -> None:
    table = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError):
        keep(table, rule)
    with pytest.raises(ValueError):
        apply_filters(table, [IN_RANGE, rule])


def test_two_saved_filters_with_one_name_are_refused(experiment: Experiment) -> None:
    exp = with_filters(experiment, [IN_RANGE, {**IN_RANGE, "high": 5}])

    with pytest.raises(ValueError):
        saved_filters(exp)


def test_an_unknown_name_is_refused(experiment: Experiment) -> None:
    exp = with_filters(experiment, [IN_RANGE])

    with pytest.raises(ValueError):
        named_filters(exp, ["x in range", "no such filter"])
