"""A step run as the app runs it records the wells that failed."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from builders import aligned, converted, measured, measured_values, segmented, with_filters, with_labels

from fouri import io, report, run
from fouri.config import Experiment
from fouri.labels import apply_labels, named_labels


def _nuc_area_between(name: str, low: float, high: float) -> dict[str, Any]:
    return {"name": name, "use": True, "rule": "between", "feature": "nuc_area", "low": low, "high": high}


NOT_THE_BIGGEST = _nuc_area_between("not the biggest", 0, 40)
NOT_THE_SMALLEST = _nuc_area_between("not the smallest", 31, 100)
MISSING_COLUMN = {"name": "EdU spread", "use": True, "rule": "robust_sd", "feature": "R3_EdU_nuc_mean", "sds": 3}

SIZE_LABEL = {
    "column": "size",
    "use": True,
    "rule": "threshold",
    "feature": "nuc_area",
    "cutoffs": [34, 38],
    "classes": ["small", "medium", "large"],
    "set_on_samples": [1],
    "sds": 3,
}
PHASE_LABEL = {
    "column": "phase",
    "use": True,
    "rule": "gates",
    "x": "nuc_area",
    "y": "R2_EdU_nuc_mean",
    "gates": [
        {"class": "early", "shape": "rectangle", "points": [[29.5, 3000], [33.5, 3500]]},
        {"class": "late", "shape": "freehand", "points": [[37.5, 3000], [45, 3000], [37.5, 3600]]},
    ],
}
MISSING_FEATURE_LABEL = {**SIZE_LABEL, "column": "EdU state", "feature": "R3_EdU_nuc_mean"}
WELL_LABEL = {**SIZE_LABEL, "column": "well"}


def _measured_wells(exp: Experiment) -> None:
    for index, well in enumerate(exp.well_list):
        converted(exp, well)
        segmented(exp, well)
        aligned(exp, well)
        measured(exp, well)
        measured_values(exp, well, "nuc_area", [30.0 + 4 * index + cell for cell in range(4)])


def _column(adata: ad.AnnData, name: str) -> np.ndarray:
    return np.asarray(adata[:, name].X).ravel()


def _names(classes: Iterable[Any]) -> list[str | None]:
    return [None if pd.isna(name) else str(name) for name in classes]


def test_a_real_step_records_its_failed_wells(experiment: Experiment) -> None:
    for well in ("B02", "B03"):
        converted(experiment, well)
        segmented(experiment, well)
    aligned(experiment, "B02")
    aligned(experiment, "B03", rounds=[])  # every round but the anchor left unaligned

    outcome = run.run_step("measure", experiment.config_path, wells=["B02", "B03"])

    lines = Path(io.run_record_path(experiment)).read_text(encoding="utf-8").splitlines()
    ended = json.loads(lines[-1])
    assert outcome == "failed"
    assert ended["event"] == "ended"
    assert ended["outcome"] == "failed"
    assert ended["failed"] == ["B03"]


def test_build_cell_table_writes_included_from_the_named_filters(experiment: Experiment) -> None:
    _measured_wells(experiment)
    exp = with_filters(experiment, [NOT_THE_BIGGEST, NOT_THE_SMALLEST])

    run.build_cell_table(exp, filters=["not the biggest"])

    table = pd.read_csv(io.fullest_df_path(exp))
    adata = ad.read_h5ad(io.adata_path(exp))
    expected = (table["nuc_area"] != 41).tolist()
    assert table["included"].tolist() == expected
    assert adata.obs["included"].tolist() == expected


def test_no_filters_include_every_cell(experiment: Experiment) -> None:
    _measured_wells(experiment)

    run.build_cell_table(experiment)

    assert pd.read_csv(io.fullest_df_path(experiment))["included"].tolist() == [True] * 12
    assert ad.read_h5ad(io.adata_path(experiment)).obs["included"].tolist() == [True] * 12


@pytest.mark.parametrize(
    "names",
    [["not the biggest", "not the biggst"], ["not the biggest", "EdU spread"]],
    ids=["unknown-name", "missing-column"],
)
def test_a_filter_that_cannot_be_applied_writes_nothing(experiment: Experiment, names: list[str]) -> None:
    _measured_wells(experiment)
    exp = with_filters(experiment, [NOT_THE_BIGGEST, MISSING_COLUMN])

    with pytest.raises(ValueError):
        run.build_cell_table(exp, filters=names)

    assert not io.fullest_df_path(exp).exists()
    assert not io.adata_path(exp).exists()


def test_z_scores_come_from_the_included_cells(experiment: Experiment) -> None:
    _measured_wells(experiment)
    exp = with_filters(experiment, [NOT_THE_BIGGEST])

    run.build_cell_table(exp, filters=["not the biggest"])

    adata = ad.read_h5ad(io.adata_path(exp))
    included = adata.obs["included"].to_numpy(dtype=bool)
    z = _column(adata, "nuc_area")
    areas = np.arange(30.0, 41.0)
    assert z[included].mean() == pytest.approx(0, abs=1e-12)
    assert z[included].std(ddof=1) == pytest.approx(1)
    assert z[~included] == pytest.approx([(41 - areas.mean()) / areas.std(ddof=1)])


def test_a_column_empty_only_in_excluded_cells_is_kept(experiment: Experiment) -> None:
    _measured_wells(experiment)
    measured_values(experiment, "C02", "R1_cMyc_nuc_mean", [2000.0, 2100.0, 2200.0, np.nan])
    exp = with_filters(experiment, [NOT_THE_BIGGEST])

    run.build_cell_table(exp, filters=["not the biggest"])

    adata = ad.read_h5ad(io.adata_path(exp))
    included = adata.obs["included"].to_numpy(dtype=bool)
    assert "R1_cMyc_nuc_mean" in adata.var_names
    assert not np.isnan(_column(adata, "R1_cMyc_nuc_mean")[included]).any()


def test_a_column_with_one_value_in_every_included_cell_is_dropped(experiment: Experiment, tmp_path: Path) -> None:
    _measured_wells(experiment)
    for well in ("B02", "B03"):
        measured_values(experiment, well, "R1_cMyc_nuc_mean", [0.4762] * 4)
    measured_values(experiment, "C02", "R1_cMyc_nuc_mean", [0.4762, 0.4762, 0.4762, 0.9])
    exp = with_filters(experiment, [NOT_THE_BIGGEST])
    progress = tmp_path / "progress.jsonl"
    report.report_to(progress)

    run.build_cell_table(exp, filters=["not the biggest"])

    lines = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    said = [line["text"] for line in lines if line["event"] == "message"]
    assert "R1_cMyc_nuc_mean" not in ad.read_h5ad(io.adata_path(exp)).var_names
    assert any("R1_cMyc_nuc_mean" in text for text in said)


def test_the_run_record_names_the_filters_applied(experiment: Experiment) -> None:
    _measured_wells(experiment)
    exp = with_filters(experiment, [NOT_THE_BIGGEST, NOT_THE_SMALLEST])
    names = ["not the biggest", "not the smallest"]

    outcome = run.run_step("table", exp.config_path, settings={"filters": names})

    lines = Path(io.run_record_path(exp)).read_text(encoding="utf-8").splitlines()
    started = next(entry for entry in map(json.loads, lines) if entry["event"] == "started")
    assert outcome == "done"
    assert started["settings"] == {"filters": names}


@pytest.mark.parametrize(("named", "other"), [("size", "phase"), ("phase", "size")], ids=["threshold", "gates"])
def test_build_cell_table_adds_the_named_labels(experiment: Experiment, named: str, other: str) -> None:
    _measured_wells(experiment)
    exp = with_labels(with_filters(experiment, [NOT_THE_BIGGEST]), [SIZE_LABEL, PHASE_LABEL])

    run.build_cell_table(exp, filters=["not the biggest"], labels=[named])

    table = pd.read_csv(io.fullest_df_path(exp))
    obs = ad.read_h5ad(io.adata_path(exp)).obs
    expected = apply_labels(table.drop(columns=named), named_labels(exp, [named]))[named]
    assert _names(table[named]) == _names(expected)
    assert _names(obs[named]) == _names(expected)
    assert obs[named].cat.categories.tolist() == expected.categories.tolist()
    assert table.loc[~table["included"], named].notna().all()
    assert other not in table.columns
    assert other not in obs.columns


@pytest.mark.parametrize(
    ("saved", "names"),
    [
        ([SIZE_LABEL, PHASE_LABEL], ["size", "sise"]),
        ([SIZE_LABEL, MISSING_FEATURE_LABEL], ["size", "EdU state"]),
        ([SIZE_LABEL, WELL_LABEL], ["size", "well"]),
        ([SIZE_LABEL, {**PHASE_LABEL, "column": "size"}], ["size"]),
    ],
    ids=["unknown-name", "missing-feature", "column-the-table-has", "two-labels-one-column"],
)
def test_a_label_that_cannot_be_applied_writes_nothing(
    experiment: Experiment, saved: list[dict[str, Any]], names: list[str]
) -> None:
    _measured_wells(experiment)
    exp = with_labels(experiment, saved)

    with pytest.raises(ValueError):
        run.build_cell_table(exp, labels=names)

    assert not io.fullest_df_path(exp).exists()
    assert not io.adata_path(exp).exists()


def test_the_run_record_names_the_labels_applied(experiment: Experiment) -> None:
    _measured_wells(experiment)
    exp = with_labels(experiment, [SIZE_LABEL, PHASE_LABEL])
    names = ["phase", "size"]

    outcome = run.run_step("table", exp.config_path, settings={"labels": names})

    lines = Path(io.run_record_path(exp)).read_text(encoding="utf-8").splitlines()
    started = next(entry for entry in map(json.loads, lines) if entry["event"] == "started")
    assert outcome == "done"
    assert started["settings"] == {"labels": names}


def test_saved_rules_with_one_name_stop_no_run_that_names_none(experiment: Experiment) -> None:
    _measured_wells(experiment)
    exp = with_filters(experiment, [NOT_THE_BIGGEST, {**NOT_THE_SMALLEST, "name": "not the biggest"}])
    exp = with_labels(exp, [SIZE_LABEL, {**PHASE_LABEL, "column": "size"}])

    run.build_cell_table(exp)

    assert io.fullest_df_path(exp).exists()
    assert io.adata_path(exp).exists()
