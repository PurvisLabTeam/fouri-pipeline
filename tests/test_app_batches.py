from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from builders import four_wells, varied_cell_data, with_filters

from fouri import batch, io, run
from fouri.app.batches import live_numbers, matches, saved_correction
from fouri.config import Experiment

BATCH = "plate column"
KEEP = ["treatment"]
TOP_CMYC = {"name": "top cMyc", "use": True, "rule": "between", "feature": "R1_cMyc_nuc_mean", "low": 0, "high": 2250}
SETUP = {"batch": BATCH, "keep": KEEP, "log_intensities": True}


def _corrected(folder: Path, log_intensities: bool = True) -> Experiment:
    folder.mkdir(parents=True, exist_ok=True)
    exp = four_wells(folder)
    varied_cell_data(exp)
    exp = with_filters(exp, [TOP_CMYC])
    run.build_cell_table(exp, filters=[TOP_CMYC["name"]])
    batch.correct_batches(exp, BATCH, keep=KEEP, log_intensities=log_intensities)
    return exp


def _texts(values: Any) -> list[str]:
    return [str(value) for value in np.asarray(values, dtype=object).ravel()]


def test_live_numbers_match_batch_metrics_on_the_same_cells(tmp_path: Path) -> None:
    exp = _corrected(tmp_path)
    cells = ad.read_h5ad(io.adata_path(exp))
    included = cells.obs["included"].to_numpy(dtype=bool)
    obs = cells.obs[included]
    conditions = batch.condition_labels(obs, KEEP)
    batches = batch.batch_labels(obs, BATCH)
    expected = batch.batch_metrics(
        np.asarray(cells.X, dtype=np.float64)[included], list(cells.var_names), conditions, batches
    )

    live = live_numbers(exp, BATCH, KEEP)

    assert 0 < included.sum() < len(included)
    assert live.cells == included.sum()
    assert live.why_not is None
    assert live.groups == batch.batch_groups(conditions, batches)
    assert expected is not None
    assert live.metrics is not None
    pd.testing.assert_series_equal(live.metrics.shares, expected.shares)
    assert live.metrics.batch_median == pytest.approx(expected.batch_median)
    assert live.metrics.features_over == expected.features_over
    assert live.metrics.classifier == pytest.approx(expected.classifier)
    assert live.metrics.chance == pytest.approx(expected.chance)
    pd.testing.assert_series_equal(live.metrics.classifier_by_condition, expected.classifier_by_condition)
    pd.testing.assert_series_equal(live.metrics.chance_by_condition, expected.chance_by_condition)
    assert live.metrics.condition_median == pytest.approx(expected.condition_median)
    assert live.metrics.top == expected.top


def test_saved_correction_reads_back_what_batch_effects_saved(tmp_path: Path) -> None:
    exp = _corrected(tmp_path)
    cells = ad.read_h5ad(io.adata_path(exp))
    record = cells.uns["batch_effects"]

    saved = saved_correction(exp)

    assert saved is not None
    assert saved.batch == BATCH
    assert saved.keep == KEEP
    assert saved.log_transformed == batch.log_columns(cells.var)
    assert saved.log_transformed
    assert saved.conditions == _texts(record["conditions"])
    assert saved.left_out == _texts(record.get("left_out", []))
    assert saved.groups == [_texts(record["groups"]["batches"])]
    adjustment = record["adjustment"]
    pd.testing.assert_frame_equal(
        saved.adjustment,
        pd.DataFrame(
            np.asarray(adjustment["values"], dtype=np.float64),
            index=pd.Index(_texts(adjustment["batches"]), name="batch"),
            columns=_texts(adjustment["features"]),
        ),
    )
    for name in ("before", "after", "reference"):
        entry, stored = getattr(saved, name), record[name]
        for key in ("batch_median", "features_over", "chance", "classifier", "condition_median"):
            assert entry[key] == pytest.approx(stored[key])
        assert entry["features"] == _texts(stored["features"])
        assert entry["top"] == _texts(stored["top"])
        np.testing.assert_allclose(entry["shares"], np.asarray(stored["shares"], dtype=np.float64))
        for key in ("classifier_by_condition", "chance_by_condition"):
            assert entry[key]["conditions"] == _texts(stored[key]["conditions"])
            np.testing.assert_allclose(entry[key]["values"], np.asarray(stored[key]["values"], dtype=np.float64))


def test_matches_compares_the_saved_setup_with_the_pages(tmp_path: Path) -> None:
    logged = saved_correction(_corrected(tmp_path / "logged"))
    unlogged = saved_correction(_corrected(tmp_path / "unlogged", log_intensities=False))

    assert logged is not None
    assert unlogged is not None
    assert matches(logged, SETUP)
    assert not matches(logged, {**SETUP, "batch": "well"})
    assert not matches(logged, {**SETUP, "keep": []})
    assert not matches(logged, {**SETUP, "keep": ["treatment", "treatment_time"]})
    assert not matches(logged, {**SETUP, "log_intensities": False})
    assert matches(unlogged, {**SETUP, "log_intensities": False})
    assert not matches(unlogged, SETUP)
