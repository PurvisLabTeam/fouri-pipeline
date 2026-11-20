import contextlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from builders import four_wells, measured_values, varied_cell_data, with_filters, with_metadata

from fouri import batch, io, run
from fouri.config import Experiment
from fouri.embedding import subsample

SMALL_AREAS = {"name": "small areas", "use": True, "rule": "between", "feature": "nuc_area", "low": 0, "high": 40}


def _table(exp: Experiment, filters: Sequence[Mapping[str, Any]] = ()) -> Experiment:
    varied_cell_data(exp)
    exp = with_filters(exp, filters) if filters else exp
    run.build_cell_table(exp, filters=[str(rule["name"]) for rule in filters])
    return exp


def _correct(exp: Experiment, by: str = "plate column", keep: Sequence[str] = ("treatment",)) -> ad.AnnData:
    batch.correct_batches(exp, by, keep=list(keep))
    return ad.read_h5ad(io.adata_path(exp))


def _keys(cells: ad.AnnData) -> list[tuple[str, int]]:
    return [(str(well), int(label)) for well, label in zip(cells.obs["well"], cells.obs["label"], strict=True)]


def _layer(cells: ad.AnnData) -> np.ndarray:
    return np.asarray(cells.layers[io.COMBAT_LAYER], dtype=float)


def _wells(cells: ad.AnnData) -> np.ndarray:
    return cells.obs["well"].astype(str).to_numpy()


def test_cells_with_a_blank_condition_are_left_out_and_named(tmp_path: Path) -> None:
    exp = _table(with_metadata(four_wells(tmp_path), "C03", treatment=None))

    cells = _correct(exp)

    assert np.isnan(_layer(cells)[_wells(cells) == "C03"]).all()
    assert np.isfinite(_layer(cells)[np.isin(_wells(cells), ["B02", "B03"])]).all()
    assert dict(cells.uns["batch_effects"]["blank_wells"]) == {"C03": 4}


def test_cells_with_a_blank_batch_are_left_out_and_named(tmp_path: Path) -> None:
    exp = _table(with_metadata(four_wells(tmp_path), "C03", batch=None))

    cells = _correct(exp, by="batch")

    assert np.isnan(_layer(cells)[_wells(cells) == "C03"]).all()
    assert np.isfinite(_layer(cells)[np.isin(_wells(cells), ["B02", "B03"])]).all()
    assert dict(cells.uns["batch_effects"]["blank_wells"]) == {"C03": 4}


def test_a_stale_subsample_gets_no_layer_and_the_table_is_corrected(experiment: Experiment) -> None:
    exp = _table(experiment)
    subsample(exp)
    exp = with_filters(exp, [SMALL_AREAS])
    run.build_cell_table(exp, filters=[SMALL_AREAS["name"]])
    stale = io.adata_sub_path(exp).read_bytes()

    cells = _correct(exp)

    assert io.COMBAT_LAYER in cells.layers
    assert "batch_effects" in cells.uns
    assert io.adata_sub_path(exp).read_bytes() == stale


def test_every_corrected_row_is_wholly_empty_or_wholly_filled(experiment: Experiment) -> None:
    exp = _table(experiment, [SMALL_AREAS])

    layer = _layer(_correct(exp))

    empty = np.isnan(layer)
    assert (empty.all(axis=1) | ~empty.any(axis=1)).all()


def test_left_out_conditions_and_excluded_cells_have_no_corrected_value(experiment: Experiment) -> None:
    exp = _table(experiment, [SMALL_AREAS])

    cells = _correct(exp)

    layer = _layer(cells)
    included = cells.obs["included"].to_numpy(dtype=bool)
    control = _wells(cells) == "B02"
    assert np.isnan(layer[control]).all()
    assert np.isnan(layer[~included]).all()
    assert np.isfinite(layer[included & ~control]).all()
    assert (~included).sum() == 2


def test_corrected_rows_match_their_cells(experiment: Experiment) -> None:
    exp = _table(experiment)

    cells = _correct(exp)

    measured = cells.var_names[cells.var["kind"] != "derived"]
    columns = cells.var_names.get_indexer(measured)
    original = np.asarray(cells.X, dtype=float)[:, columns]
    corrected = _layer(cells)[:, columns]
    for column_batch in (["B03"], ["C02"]):
        rows = np.isin(_wells(cells), column_batch)
        np.testing.assert_array_equal(
            np.argsort(original[rows], axis=0, kind="stable"), np.argsort(corrected[rows], axis=0, kind="stable")
        )


def test_the_subsample_layer_rows_match_their_cells(experiment: Experiment) -> None:
    exp = _table(experiment)
    subsample(exp, cells_per_sample=3)

    cells = _correct(exp)

    sub = ad.read_h5ad(io.adata_sub_path(exp))
    rows = {key: row for row, key in enumerate(_keys(cells))}
    expected = _layer(cells)[[rows[key] for key in _keys(sub)]]
    np.testing.assert_array_equal(np.asarray(sub.layers[io.COMBAT_LAYER], dtype=float), expected)


def test_corrected_values_match_combat_on_the_same_values(experiment: Experiment) -> None:
    exp = _table(experiment)

    cells = _correct(exp)

    obs = cells.obs.copy()
    features = list(cells.var_names)
    full = pd.read_csv(io.fullest_df_path(exp)).set_index(["well", "label"])
    original = full.loc[_keys(cells)].reset_index(drop=True)
    values = original[features].to_numpy(dtype=float)
    logged = cells.var_names.isin(batch.log_columns(cells.var))
    values[:, logged] = np.log1p(values[:, logged])
    batches = batch.batch_labels(obs, "plate column").reset_index(drop=True)
    conditions = batch.condition_labels(obs, ["treatment"]).reset_index(drop=True)
    corrected = batch.combat(values, batches, conditions)
    corrected[:, logged] = np.expm1(corrected[:, logged])
    own_units = original.copy()
    own_units[features] = corrected
    for spec in exp.derived_features:
        if spec["name"] in features:
            own_units[spec["name"]] = own_units.eval(spec["formula"])
    expected = (own_units[features] - original[features].mean()) / original[features].std()
    assert logged.any()
    assert "pRB_RB_ratio" in features
    np.testing.assert_allclose(_layer(cells), expected.to_numpy(dtype=float), equal_nan=True, rtol=1e-9, atol=1e-9)


def test_conditions_are_passed_as_covariates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scanpy

    exp = _table(four_wells(tmp_path))
    covariates = []
    real_combat = scanpy.pp.combat

    def record_covariates(adata: ad.AnnData, *args: Any, **kwargs: Any) -> Any:
        covariates.append(kwargs.get("covariates"))
        return real_combat(adata, *args, **kwargs)

    monkeypatch.setattr(scanpy.pp, "combat", record_covariates)

    _correct(exp)

    assert covariates
    assert all(list(each or []) == ["condition_palbo"] for each in covariates)


def test_linked_batches_form_one_group() -> None:
    conditions = pd.Series(["a", "a", "b", "b", "c", "c", "d"])
    batches = pd.Series(["1", "2", "2", "3", "4", "5", "6"])

    groups = batch.batch_groups(conditions, batches)

    assert groups == [["1", "2", "3"], ["4", "5"]]


def test_a_refused_correction_writes_no_layer(experiment: Experiment) -> None:
    exp = _table(experiment)
    before = io.adata_path(exp).read_bytes()

    with pytest.raises(ValueError):
        batch.correct_batches(exp, "well", keep=["treatment", "treatment time"])

    assert io.adata_path(exp).read_bytes() == before


def test_the_adjustment_and_left_out_records(experiment: Experiment) -> None:
    exp = _table(experiment)

    cells = _correct(exp)

    record = cells.uns["batch_effects"]
    used = ~np.isnan(_layer(cells)).all(axis=1)
    difference = pd.DataFrame(_layer(cells)[used] - np.asarray(cells.X, dtype=float)[used], columns=cells.var_names)
    expected = difference.groupby(batch.batch_labels(cells.obs, "plate column").to_numpy()[used]).mean()
    assert list(record["left_out"]) == ["control"]
    assert list(record["conditions"]) == ["palbo"]
    assert list(record["adjustment"]["batches"]) == ["2", "3"]
    assert list(record["adjustment"]["features"]) == list(cells.var_names)
    np.testing.assert_allclose(np.asarray(record["adjustment"]["values"]), expected.loc[["2", "3"]].to_numpy())


def test_the_main_file_is_written_before_the_subsample(
    experiment: Experiment, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp = _table(experiment)
    subsample(exp, cells_per_sample=3)
    written: list[Path] = []
    real_written_whole = io.written_whole

    @contextlib.contextmanager
    def record_writes(path: str | Path) -> Iterator[Path]:
        written.append(Path(path))
        with real_written_whole(path) as partial:
            yield partial

    monkeypatch.setattr(io, "written_whole", record_writes)

    _correct(exp)

    assert written == [io.adata_path(exp), io.adata_sub_path(exp)]


def test_a_condition_absent_from_one_batch_is_uneven() -> None:
    conditions = pd.Series(["a", "a", "b", "b", "a", "a", "b", "b", "b", "b"])
    batches = pd.Series(["1", "1", "1", "1", "2", "2", "2", "2", "3", "3"])

    uneven = batch.uneven_conditions(batch.condition_shares(conditions, batches))

    assert uneven == [("a", "3", "1")]


def test_a_cell_missing_from_the_csv_is_refused(experiment: Experiment) -> None:
    exp = _table(experiment)
    path = io.fullest_df_path(exp)
    table = pd.read_csv(path)
    table.drop(index=table.index[-1]).to_csv(path, index=False)
    before = io.adata_path(exp).read_bytes()

    with pytest.raises(ValueError):
        batch.correct_batches(exp, "plate column", keep=["treatment"])

    assert io.adata_path(exp).read_bytes() == before


VARIED_MEASURES = ["R0_pRB_ring_mean", "R0_pRB_total_nuc_protein", "R0_pRB_nuc_cyto_ratio"]


def test_only_intensity_measures_are_log_transformed(experiment: Experiment) -> None:
    varied_cell_data(experiment)
    for index, well in enumerate(experiment.well_list):
        for offset, column in enumerate(VARIED_MEASURES, start=1):
            measured_values(experiment, well, column, [100.0 * offset + 7 * (4 * index + cell) for cell in range(4)])
    run.build_cell_table(experiment)

    cells = _correct(experiment)

    logged = {str(name) for name in cells.uns["batch_effects"]["log_transformed"]}
    intensities = {
        str(name) for name in cells.var_names if str(name).endswith(("_nuc_mean", "_ring_mean", "_total_nuc_protein"))
    }
    assert {*VARIED_MEASURES, "nuc_area", "pRB_RB_ratio"} <= set(cells.var_names)
    assert {"R0_pRB_ring_mean", "R0_pRB_total_nuc_protein"} <= logged
    assert logged == intensities
