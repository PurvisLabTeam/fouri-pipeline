"""A step run as the app runs it records the wells that failed."""

import functools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from builders import (
    NUCLEUS_SIDE,
    NUCLEUS_STARTS,
    SWAPPED_INFO,
    aligned,
    converted,
    masked,
    measured,
    measured_values,
    raw_files,
    segmented,
    subsampled,
    varied_cell_data,
    well_outputs,
    with_derived,
    with_embedding_setup,
    with_filters,
    with_info,
    with_labels,
    with_metadata,
)

from fouri import io, preprocess, report, run
from fouri.config import Experiment
from fouri.embedding import subsample
from fouri.embedding_setup import embedding_settings
from fouri.labels import apply_labels, named_labels
from fouri.status import Step


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


MIDDLE_AREAS = {
    "name": "middle areas",
    "use": True,
    "rule": "middle_percentiles",
    "feature": "nuc_area",
    "low": 10,
    "high": 90,
}


def _nucleus(label: int) -> tuple[slice, slice]:
    row = NUCLEUS_STARTS[(label - 1) // 2]
    col = NUCLEUS_STARTS[(label - 1) % 2]
    return slice(row, row + NUCLEUS_SIDE), slice(col, col + NUCLEUS_SIDE)


def _measured_wells(exp: Experiment, covered: tuple[str, int] | None = None) -> None:
    for index, well in enumerate(exp.well_list):
        converted(exp, well)
        segmented(exp, well)
        aligned(exp, well)
        if covered is not None and covered[0] == well:
            masked(exp, well, 2, {"EdU": _nucleus(covered[1])})
        measured(exp, well)
        measured_values(exp, well, "nuc_area", [30.0 + 4 * index + cell for cell in range(4)])


def _column(adata: ad.AnnData, name: str) -> np.ndarray:
    return np.asarray(adata[:, name].X).ravel()


def _names(classes: Iterable[Any]) -> list[str | None]:
    return [None if pd.isna(name) else str(name) for name in classes]


def _cells(frame: pd.DataFrame) -> list[tuple[str, int]]:
    return sorted((str(well), int(label)) for well, label in zip(frame["well"], frame["label"], strict=True))


def _left_out(frame: pd.DataFrame) -> list[tuple[str, int]]:
    return _cells(frame[~frame["included"].astype(bool)])


def _every_cell(exp: Experiment) -> list[tuple[str, int]]:
    return sorted((well, label) for well in exp.well_list for label in range(1, 5))


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


def test_the_run_record_keeps_every_export_setting(experiment: Experiment) -> None:
    _measured_wells(experiment)
    exp = with_filters(experiment, [NOT_THE_BIGGEST])
    run.build_cell_table(exp, filters=["not the biggest"])
    standard = ad.read_h5ad(io.adata_path(exp))
    standard.layers["combat"] = np.asarray(standard.X, dtype=float) + 1
    standard.write_h5ad(io.adata_path(exp))
    key = "X_umap_2d_top1_n_neighbors5"
    subsampled(exp, [0, 4, 8], key)
    settings = {
        "name": "cycling",
        "file_format": "csv",
        "values": "both",
        "corrected": True,
        "features": ["nuc_area"],
        "embeddings": [key],
        "overwrite": True,
    }

    outcome = run.run_step("export", exp.config_path, settings=settings)

    lines = Path(io.run_record_path(exp)).read_text(encoding="utf-8").splitlines()
    started = next(entry for entry in map(json.loads, lines) if entry["event"] == "started")
    assert outcome == "done"
    assert started["settings"] == settings


def test_build_cell_table_removes_the_covered_cells_and_the_cover_columns(experiment: Experiment) -> None:
    _measured_wells(experiment, covered=("B03", 2))

    run.build_cell_table(experiment)

    table = pd.read_csv(io.fullest_df_path(experiment))
    adata = ad.read_h5ad(io.adata_path(experiment))
    expected = [cell for cell in _every_cell(experiment) if cell != ("B03", 2)]
    assert _cells(table) == expected
    assert _cells(adata.obs) == expected
    assert [column for column in table.columns if column.endswith("_covered")] == []
    assert [name for name in [*adata.obs.columns, *adata.var_names] if name.endswith("_covered")] == []


def test_filters_are_judged_on_the_cells_left_after_the_masks(experiment: Experiment) -> None:
    _measured_wells(experiment, covered=("C02", 4))
    exp = with_filters(experiment, [MIDDLE_AREAS])
    run.build_cell_table(exp, filters=["middle areas"])
    left_out_after_the_masks = _left_out(pd.read_csv(io.fullest_df_path(exp)))

    path = io.cell_data_path(exp, "C02")
    cells = pd.read_csv(path)
    cells[cells["label"] != 4].to_csv(path, index=False)
    run.build_cell_table(exp, filters=["middle areas"])

    assert left_out_after_the_masks == _left_out(pd.read_csv(io.fullest_df_path(exp)))


def test_cell_data_without_cover_columns_counts_as_uncovered(experiment: Experiment) -> None:
    _measured_wells(experiment, covered=("B03", 2))
    path = io.cell_data_path(experiment, "B03")
    cells = pd.read_csv(path)
    cells.drop(columns=[column for column in cells.columns if column.endswith("_covered")]).to_csv(path, index=False)

    run.build_cell_table(experiment)

    assert _cells(pd.read_csv(io.fullest_df_path(experiment))) == _every_cell(experiment)
    assert _cells(ad.read_h5ad(io.adata_path(experiment)).obs) == _every_cell(experiment)


def _changed_b02(exp: Experiment, monkeypatch: pytest.MonkeyPatch) -> dict[Path, bytes]:
    raw_files(exp, monkeypatch)
    preprocess.preprocess(exp, wells=["B02"])
    segmented(exp, "B02")
    aligned(exp, "B02")
    with_info(exp, SWAPPED_INFO)
    return well_outputs(exp, "B02")


def _reconvert(exp: Experiment, *flags: str) -> int:
    return run.main(["reconvert", "--config", str(exp.config_path), "--wells", "B02", *flags])


def test_reconvert_without_force_lists_and_refuses(experiment: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    before = _changed_b02(experiment, monkeypatch)

    code = _reconvert(experiment)

    assert code == 2
    assert well_outputs(experiment, "B02") == before


def test_reconvert_dry_run_deletes_nothing(experiment: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    before = _changed_b02(experiment, monkeypatch)

    code = _reconvert(experiment, "--dry-run")

    assert code == 0
    assert well_outputs(experiment, "B02") == before


def test_reconvert_with_force_records_convert_images(experiment: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    _changed_b02(experiment, monkeypatch)

    code = _reconvert(experiment, "--force")

    lines = Path(io.run_record_path(experiment)).read_text(encoding="utf-8").splitlines()
    started = next(entry for entry in map(json.loads, lines) if entry["event"] == "started")
    assert code == 0
    assert started["step"] == "convert"
    assert started["wells"] == ["B02"]
    assert started["settings"]["reconvert_changed"] is True
    assert not (Path(experiment.segmented) / "B02").exists()


@pytest.mark.parametrize("flags", [(), ("--dry-run",), ("--force",)], ids=["no-flag", "dry-run", "force"])
def test_reconvert_with_no_changed_well_runs_nothing(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch, flags: tuple[str, ...]
) -> None:
    raw_files(experiment, monkeypatch)
    preprocess.preprocess(experiment, wells=["B02"])
    record = io.run_record_path(experiment)
    before = record.read_bytes() if record.exists() else None

    code = _reconvert(experiment, *flags)

    assert code == 0
    assert (record.read_bytes() if record.exists() else None) == before


UMAP_RUN = {"method": "UMAP", "dims": 2, "top_n": 2, "settings": {"n_neighbors": 5}}
PICKED = ["nuc_area", "R0_pRB_nuc_mean", "R0_RB_nuc_mean"]
PER_SAMPLE_AREAS = {**MIDDLE_AREAS, "name": "middle areas per sample", "per_sample": True}


def _table_files(exp: Experiment) -> dict[Path, bytes]:
    paths = (io.fullest_df_path(exp), io.adata_path(exp))
    return {path: path.read_bytes() for path in paths if path.exists()}


def _embed(exp: Experiment, **options: Any) -> list[str]:
    return run.make_embeddings(exp, [UMAP_RUN], PICKED, ranking_k=5, **options)


def test_a_failed_anndata_write_leaves_both_files_as_they_were(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch
) -> None:
    varied_cell_data(experiment)
    run.build_cell_table(experiment)
    before = _table_files(experiment)
    measured_values(experiment, "B02", "nuc_area", [41.0, 42.0, 43.0, 44.0])

    def write_fails(self: ad.AnnData, *args: Any, **kwargs: Any) -> None:
        raise OSError("made-up failure while writing the AnnData")

    monkeypatch.setattr(ad.AnnData, "write_h5ad", write_fails)

    with pytest.raises(OSError):
        run.build_cell_table(experiment)

    assert _table_files(experiment) == before
    assert list(Path(experiment.base_dir).glob("*.partial")) == []


def test_a_label_named_twice_is_refused_before_writing(experiment: Experiment) -> None:
    varied_cell_data(experiment)
    exp = with_labels(experiment, [SIZE_LABEL])

    with pytest.raises(ValueError):
        run.build_cell_table(exp, labels=["size", "size"])

    assert _table_files(exp) == {}


def test_a_derived_feature_ending_in_covered_is_refused(experiment: Experiment) -> None:
    varied_cell_data(experiment)
    exp = with_derived(experiment, [{"name": "pRB_covered", "formula": "R0_pRB_nuc_mean / R0_RB_nuc_mean"}])

    with pytest.raises(ValueError):
        run.build_cell_table(exp)

    assert _table_files(exp) == {}


def test_a_derived_feature_named_as_a_measured_column_is_refused(experiment: Experiment) -> None:
    varied_cell_data(experiment)
    exp = with_derived(experiment, [{"name": "nuc_area", "formula": "nuc_area * 2"}])

    with pytest.raises(ValueError):
        run.build_cell_table(exp)

    assert _table_files(exp) == {}


def test_a_per_sample_filter_with_a_well_lacking_sample_id_is_refused(experiment: Experiment) -> None:
    varied_cell_data(experiment)
    exp = with_filters(with_metadata(experiment, "B03", sample_ID=None), [PER_SAMPLE_AREAS])

    with pytest.raises(ValueError):
        run.build_cell_table(exp, filters=[PER_SAMPLE_AREAS["name"]])

    assert _table_files(exp) == {}


def _without_round(exp: Experiment, well: str, round_name: str) -> None:
    path = io.cell_data_path(exp, well)
    cells = pd.read_csv(path)
    cells.drop(columns=[column for column in cells.columns if column.startswith(f"{round_name}_")]).to_csv(
        path, index=False
    )


def test_build_cell_table_builds_from_the_measured_wells_and_names_the_others(
    experiment: Experiment, tmp_path: Path
) -> None:
    varied_cell_data(experiment, ["B02", "B03"])
    _without_round(experiment, "B03", "R2")
    progress = tmp_path / "progress.jsonl"
    report.report_to(progress)

    run.build_cell_table(experiment)

    built = pd.read_csv(io.fullest_df_path(experiment))
    measured_cells = pd.read_csv(io.cell_data_path(experiment, "B02"))
    lines = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    said = [line["text"] for line in lines if line["event"] == "message"]
    assert built["well"].astype(str).tolist() == ["B02"] * len(measured_cells)
    assert any("B03" in text and "C02" in text for text in said)


def test_build_cell_table_refuses_when_no_well_is_measured(experiment: Experiment) -> None:
    varied_cell_data(experiment, ["B02"])
    _without_round(experiment, "B02", "R2")

    with pytest.raises(ValueError):
        run.build_cell_table(experiment)

    assert _table_files(experiment) == {}


def test_embedding_refuses_a_subsample_from_another_cell_table(experiment: Experiment, stand_in_umap: object) -> None:
    varied_cell_data(experiment)
    run.build_cell_table(experiment)
    subsample(experiment)
    run.build_cell_table(experiment)
    before = io.adata_sub_path(experiment).read_bytes()

    with pytest.raises(ValueError):
        _embed(experiment)

    assert io.adata_sub_path(experiment).read_bytes() == before


def test_a_subsample_without_an_id_is_refused(experiment: Experiment, stand_in_umap: object) -> None:
    varied_cell_data(experiment)
    run.build_cell_table(experiment)
    subsampled(experiment, list(range(12)), "X_phate_3d_top2_knn5", cell_table=False)
    before = io.adata_sub_path(experiment).read_bytes()

    with pytest.raises(ValueError):
        _embed(experiment)

    assert io.adata_sub_path(experiment).read_bytes() == before


def test_a_table_without_an_id_is_refused_by_subsample(experiment: Experiment) -> None:
    varied_cell_data(experiment)
    run.build_cell_table(experiment)
    table = ad.read_h5ad(io.adata_path(experiment))
    table.uns.pop("cell_table", None)
    table.write_h5ad(io.adata_path(experiment))

    with pytest.raises(ValueError):
        subsample(experiment)

    assert not io.adata_sub_path(experiment).exists()


def test_build_cell_table_removes_the_subsample_layer(experiment: Experiment) -> None:
    varied_cell_data(experiment)
    run.build_cell_table(experiment)
    subsampled(experiment, list(range(12)), "X_phate_3d_top2_knn5")
    sub = ad.read_h5ad(io.adata_sub_path(experiment))
    sub.layers[io.COMBAT_LAYER] = np.asarray(sub.X, dtype=float)
    sub.write_h5ad(io.adata_sub_path(experiment))

    run.build_cell_table(experiment)

    rebuilt = ad.read_h5ad(io.adata_sub_path(experiment))
    assert io.COMBAT_LAYER not in rebuilt.layers
    assert "X_phate_3d_top2_knn5" in rebuilt.obsm


def test_embedding_overwrite_subsamples_again(experiment: Experiment, stand_in_umap: object) -> None:
    varied_cell_data(experiment)
    run.build_cell_table(experiment)
    subsampled(experiment, [0, 1, 4, 5, 8, 9], "X_phate_3d_top2_knn5")

    keys = _embed(experiment, overwrite=True, cells_per_sample=3)

    sub = ad.read_h5ad(io.adata_sub_path(experiment))
    assert "X_phate_3d_top2_knn5" not in sub.obsm
    assert list(sub.obsm) == keys
    assert sub.uns["subsample"]["cells_per_sample"] == 3
    assert sub.n_obs == 9


def _record_entries(exp: Experiment) -> list[dict[str, Any]]:
    path = Path(io.run_record_path(exp))
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _exit_code(argv: list[str]) -> int | str | None:
    try:
        return run.main(argv)
    except SystemExit as stop:
        return stop.code


@pytest.mark.parametrize(
    ("arguments", "chosen"),
    [
        pytest.param(["--settings", '{"overwrite": true, "wells": ["B03"]}'], "B03", id="settings"),
        pytest.param(["--settings-file", "<settings file>"], "B03", id="settings-file"),
        pytest.param(["wells=[B03]", "--settings", '{"overwrite": true}'], "B03", id="key-value"),
        pytest.param(
            ["--settings", '{"overwrite": true, "wells": ["B03"]}', "--wells", "B02"], "B02", id="wells-given-too"
        ),
    ],
)
def test_a_wells_setting_given_is_used(
    experiment: Experiment, tmp_path: Path, arguments: list[str], chosen: str
) -> None:
    for well in experiment.well_list:
        converted(experiment, well)
    settings_file = tmp_path / "artifacts.yaml"
    settings_file.write_text("overwrite: true\nwells: [B03]\n", encoding="utf-8")
    given = [str(settings_file) if part == "<settings file>" else part for part in arguments]

    code = run.main(["artifacts", "--config", str(experiment.config_path), *given])

    started = next(entry for entry in _record_entries(experiment) if entry["event"] == "started")
    written = {well: [io.mask_path(experiment, well, r).exists() for r in range(3)] for well in experiment.well_list}
    assert code == 0
    assert started["wells"] == [chosen]
    assert written == {well: [well == chosen] * 3 for well in experiment.well_list}


@pytest.mark.parametrize(
    ("step", "settings"),
    [
        pytest.param("artifacts", {"overwrite": "false"}, id="text-for-true-or-false"),
        pytest.param("artifacts", {"overwrite": 1}, id="number-for-true-or-false"),
        pytest.param("measure", {"ring_width": True}, id="true-for-a-whole-number"),
        pytest.param("measure", {"ring_width": "8"}, id="text-for-a-whole-number"),
        pytest.param("align", {"downscale_factor": 2.5}, id="decimal-for-a-whole-number"),
    ],
)
def test_a_setting_of_the_wrong_type_is_refused_before_the_started_line(
    experiment: Experiment, step: str, settings: dict[str, Any]
) -> None:
    outcome = run.run_step(step, experiment.config_path, settings=settings)

    assert outcome == "failed"
    assert [entry["event"] for entry in _record_entries(experiment)] == []


@pytest.mark.parametrize(
    ("step", "settings"),
    [
        pytest.param(Step.SEGMENT, {"flow_threshold": 1}, id="whole-number-for-a-number"),
        pytest.param(Step.TABLE, {"filters": ["not the biggest"]}, id="list-for-a-tuple"),
        pytest.param(Step.SEGMENT, {"model": "20261117T204206"}, id="text-for-none"),
        pytest.param(Step.EXPORT, {"name": "cycling", "features": ["nuc_area"]}, id="list-for-none"),
        pytest.param(Step.EXPORT, {"name": "cycling", "file_format": "csv"}, id="text-for-a-text-choice"),
    ],
)
def test_an_int_is_taken_for_a_float_setting(step: Step, settings: dict[str, Any]) -> None:
    run.check_settings(step, settings)


EMBEDDING_SETUP = {
    "picked": ["nuc_area", "R0_pRB_nuc_mean"],
    "samples": [1, 3],
    "ranking_k": 10,
    "top_n": [2],
    "methods": {
        "PHATE": {"dims": [3], "settings": {"knn": [5], "t": ["auto"], "gamma": [1], "decay": [40], "n_pca": [100]}},
        "UMAP": {"dims": [2], "settings": {"n_neighbors": [5, 10], "min_dist": [0.1], "metric": ["euclidean"]}},
    },
    "skipped": ["X_umap_2d_top2_metriceuclidean_min_dist0.1_n_neighbors10"],
    "seed": 3,
    "values": "combat",
}
TICKED_RUNS = [
    {
        "method": "PHATE",
        "dims": 3,
        "top_n": 2,
        "settings": {"knn": 5, "t": "auto", "gamma": 1, "decay": 40, "n_pca": 100},
    },
    {"method": "UMAP", "dims": 2, "top_n": 2, "settings": {"n_neighbors": 5, "min_dist": 0.1, "metric": "euclidean"}},
]
EVERY_RUN_UNTICKED = {
    **EMBEDDING_SETUP,
    "skipped": [
        "X_phate_3d_top2_decay40_gamma1_knn5_n_pca100_tauto",
        "X_umap_2d_top2_metriceuclidean_min_dist0.1_n_neighbors5",
        "X_umap_2d_top2_metriceuclidean_min_dist0.1_n_neighbors10",
    ],
}


def _sorted_runs(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(runs, key=lambda entry: (entry["method"], json.dumps(entry, sort_keys=True)))


def _recorded_embeddings(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    @functools.wraps(run.make_embeddings)
    def recorder(exp: Experiment, **settings: Any) -> list[str]:
        calls.append(settings)
        return []

    monkeypatch.setattr(run, "make_embeddings", recorder)
    return calls


def test_given_settings_replace_the_saved_setups(experiment: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    exp = with_embedding_setup(experiment, EMBEDDING_SETUP)
    calls = _recorded_embeddings(monkeypatch)

    outcome = run.run_step("embedding", exp.config_path, settings={"seed": 7})

    started = next(entry for entry in _record_entries(exp) if entry["event"] == "started")
    recorded = dict(started["settings"])
    assert outcome == "done"
    assert _sorted_runs(recorded.pop("runs")) == TICKED_RUNS
    assert recorded == {
        "picked": ["nuc_area", "R0_pRB_nuc_mean"],
        "samples": [1, 3],
        "ranking_k": 10,
        "seed": 7,
        "values": "combat",
    }
    assert [call["seed"] for call in calls] == [7]
    assert _sorted_runs(calls[0]["runs"]) == TICKED_RUNS


@pytest.mark.parametrize(
    "setup", [pytest.param(None, id="no-setup"), pytest.param(EVERY_RUN_UNTICKED, id="every-run-unticked")]
)
def test_the_saved_setups_refusals(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch, setup: dict[str, Any] | None
) -> None:
    exp = experiment if setup is None else with_embedding_setup(experiment, setup)
    calls = _recorded_embeddings(monkeypatch)

    outcome = run.run_step("embedding", exp.config_path)

    assert outcome == "failed"
    assert [entry["event"] for entry in _record_entries(exp)] == []
    assert calls == []


def test_embedding_settings_keep_the_ticked_runs_and_the_setup(experiment: Experiment) -> None:
    exp = with_embedding_setup(experiment, EMBEDDING_SETUP)

    settings = embedding_settings(exp)

    assert settings is not None
    kept = dict(settings)
    assert _sorted_runs(kept.pop("runs")) == TICKED_RUNS
    assert kept == {
        "picked": ["nuc_area", "R0_pRB_nuc_mean"],
        "samples": [1, 3],
        "ranking_k": 10,
        "seed": 3,
        "values": "combat",
    }


@pytest.mark.parametrize(
    ("arguments", "code", "recorded"),
    [
        pytest.param(["artifacts", "overwrite"], 2, None, id="override-without-equals"),
        pytest.param(["artifacts", "=true"], 2, None, id="override-without-name"),
        pytest.param(["artifacts", "overwrite=[true"], 2, None, id="override-not-yaml"),
        pytest.param(
            ["artifacts", "--settings", "{}", "--settings-file", "<yaml>"], 2, None, id="settings-and-settings-file"
        ),
        pytest.param(["artifacts", "--settings-file", "<missing>"], 2, None, id="settings-file-missing"),
        pytest.param(["artifacts", "--settings-file", "<broken>"], 2, None, id="settings-file-not-yaml"),
        pytest.param(["artifacts", "--settings", "[true]"], 2, None, id="settings-not-a-mapping"),
        pytest.param(["artifacts", "--settings", "{oops"], 2, None, id="settings-not-json"),
        pytest.param(["artifacts", "--settings-file", "<listed>"], 2, None, id="settings-file-not-a-mapping"),
        pytest.param(["table", "--wells", "B02"], 1, None, id="wells-for-a-step-without-wells"),
        pytest.param(["artifacts", "--settings", '{"bogus": true}'], 1, None, id="unknown-setting"),
        pytest.param(["export"], 1, None, id="missing-required-setting"),
        pytest.param(
            ["artifacts", "--wells", "B02", "overwrite=true"], 0, {"overwrite": True}, id="wells-holding-equals"
        ),
        pytest.param(None, 0, {"overwrite": True}, id="command-with-settings-file"),
    ],
)
def test_command_line_refusals(
    experiment: Experiment,
    tmp_path: Path,
    arguments: list[str] | None,
    code: int,
    recorded: dict[str, Any] | None,
) -> None:
    converted(experiment, "B02")
    files = {
        "<yaml>": tmp_path / "artifacts.yaml",
        "<missing>": tmp_path / "absent.yaml",
        "<broken>": tmp_path / "broken.yaml",
        "<listed>": tmp_path / "listed.yaml",
    }
    files["<yaml>"].write_text("overwrite: true\n", encoding="utf-8")
    files["<broken>"].write_text("overwrite: [true\n", encoding="utf-8")
    files["<listed>"].write_text("- overwrite\n- true\n", encoding="utf-8")
    if arguments is None:
        argv = run.command("artifacts", experiment.config_path, wells=["B02"], settings_file=files["<yaml>"])[3:]
        assert "--settings" not in argv
        assert argv[argv.index("--settings-file") + 1] == str(files["<yaml>"])
    else:
        given = [str(files[part]) if part in files else part for part in arguments[1:]]
        argv = [arguments[0], "--config", str(experiment.config_path), *given]

    exit_code = _exit_code(argv)

    entries = _record_entries(experiment)
    assert exit_code == code
    if recorded is None:
        assert entries == []
    else:
        assert [entry["event"] for entry in entries] == ["started", "ended"]
        assert entries[0]["settings"] == recorded
        assert entries[0]["wells"] == ["B02"]


@pytest.mark.parametrize("name", ["treatment", "sample_ID", "well", "label", "included"])
def test_a_derived_feature_named_as_a_reserved_name_is_refused(experiment: Experiment, name: str) -> None:
    varied_cell_data(experiment)
    exp = with_derived(experiment, [{"name": name, "formula": "nuc_area * 2"}])

    with pytest.raises(ValueError):
        run.build_cell_table(exp)

    assert _table_files(exp) == {}


HELD_BY_UNMEASURED_WELLS = [
    pytest.param("dose", {"C02": {"dose": "10nM"}}, id="dose"),
    pytest.param("sample_ID", {"B02": {"sample_ID": None}, "B03": {"sample_ID": None}}, id="sample-id"),
]


@pytest.mark.parametrize(("column", "changes"), HELD_BY_UNMEASURED_WELLS)
def test_a_metadata_column_held_only_by_unmeasured_wells_is_kept_empty(
    experiment: Experiment, column: str, changes: dict[str, dict[str, Any]]
) -> None:
    varied_cell_data(experiment, ["B02", "B03"])
    exp = experiment
    for well, values in changes.items():
        exp = with_metadata(exp, well, **values)

    run.build_cell_table(exp)

    table = pd.read_csv(io.fullest_df_path(exp))
    obs = ad.read_h5ad(io.adata_path(exp)).obs
    assert _cells(table) == sorted((well, label) for well in ("B02", "B03") for label in range(1, 5))
    assert column in table.columns
    assert table[column].isna().all()
    assert obs[column].isna().all()


@pytest.mark.parametrize("held_open", ["csv", "h5ad"])
def test_a_failed_replacement_of_either_file_leaves_both_as_they_were(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch, held_open: str
) -> None:
    varied_cell_data(experiment)
    run.build_cell_table(experiment)
    before = _table_files(experiment)
    listing = sorted(Path(experiment.base_dir).iterdir())
    measured_values(experiment, "B02", "nuc_area", [41.0, 42.0, 43.0, 44.0])
    locked = io.fullest_df_path(experiment) if held_open == "csv" else io.adata_path(experiment)
    replace = Path.replace

    def refused_while_open(self: Path, target: str | Path) -> Path:
        if Path(self) == locked or Path(target) == locked:
            raise PermissionError(13, "The process cannot access the file", str(locked))
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", refused_while_open)

    with pytest.raises(OSError):
        run.build_cell_table(experiment)

    assert _table_files(experiment) == before
    assert sorted(Path(experiment.base_dir).iterdir()) == listing


def _stand_in(monkeypatch: pytest.MonkeyPatch, module: str, name: str) -> None:
    def reached(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(f"{module}.{name} was called before the refusal")

    monkeypatch.setattr(f"{module}.{name}", reached)
    monkeypatch.setattr(run, name, reached, raising=False)


def test_a_label_named_twice_is_refused_before_anndata_is_reached(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch
) -> None:
    varied_cell_data(experiment)
    exp = with_labels(experiment, [SIZE_LABEL])
    _stand_in(monkeypatch, "fouri.adata", "build_adata")

    with pytest.raises(ValueError):
        run.build_cell_table(exp, labels=["size", "size"])


def test_a_covered_derived_name_is_refused_before_any_cell_is_removed(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch
) -> None:
    varied_cell_data(experiment)
    exp = with_derived(experiment, [{"name": "pRB_covered", "formula": "R0_pRB_nuc_mean / R0_RB_nuc_mean"}])
    _stand_in(monkeypatch, "fouri.artifacts", "remove_covered")

    with pytest.raises(ValueError):
        run.build_cell_table(exp)


def test_no_measured_well_is_refused_before_any_cell_data_is_read(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch
) -> None:
    varied_cell_data(experiment, ["B02"])
    _without_round(experiment, "B02", "R2")
    _stand_in(monkeypatch, "fouri.features", "build_fullest_df")

    with pytest.raises(ValueError):
        run.build_cell_table(experiment)


EMBEDDING_REFUSALS = [
    pytest.param({"values": "corrected"}, id="unknown-values"),
    pytest.param({"picked": [*PICKED, "R9_none_nuc_mean"]}, id="feature-not-in-x"),
    pytest.param({"runs": [{**UMAP_RUN, "method": "tSNE"}]}, id="unknown-method"),
    pytest.param({"runs": [{**UMAP_RUN, "top_n": 4}]}, id="top-n-too-large"),
    pytest.param({"values": "combat"}, id="combat-without-a-layer"),
    pytest.param({"ranking_k": 3}, id="ranking-k-of-3"),
    pytest.param({"ranking_k": 12}, id="ranking-k-equal-to-the-cells-ranked"),
]


@pytest.mark.parametrize("existing", [True, False], ids=["over-a-subsample", "no-subsample"])
@pytest.mark.parametrize("change", EMBEDDING_REFUSALS)
def test_every_embedding_refusal_leaves_the_subsample_unchanged(
    experiment: Experiment, stand_in_umap: object, change: dict[str, Any], existing: bool
) -> None:
    varied_cell_data(experiment)
    run.build_cell_table(experiment)
    if existing:
        subsampled(experiment, list(range(12)), "X_phate_3d_top2_knn5")
    before = io.adata_sub_path(experiment).read_bytes() if existing else None
    options: dict[str, Any] = {"runs": [UMAP_RUN], "picked": PICKED, "ranking_k": 5, **change}
    runs, picked = options.pop("runs"), options.pop("picked")

    with pytest.raises(ValueError):
        run.make_embeddings(experiment, runs, picked, overwrite=True, **options)

    if existing:
        assert io.adata_sub_path(experiment).read_bytes() == before
    else:
        assert not io.adata_sub_path(experiment).exists()
