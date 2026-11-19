from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from builders import aligned, converted, measured, measured_values, segmented, subsampled, with_filters, with_labels

from fouri import io, run
from fouri.config import Experiment
from fouri.export import export_cells

NOT_THE_BIGGEST = {
    "name": "not the biggest",
    "use": True,
    "rule": "between",
    "feature": "nuc_area",
    "low": 0,
    "high": 40,
}
SIZE_LABEL = {
    "column": "size",
    "use": True,
    "rule": "threshold",
    "feature": "nuc_area",
    "cutoffs": [34, 38],
    "classes": ["small", "medium", "large"],
}

NAME = "cycling"
VARIED = ["nuc_area", "R0_pRB_nuc_mean", "R1_cMyc_nuc_mean"]
NO_Z_SCORES = "R2_EdU_nuc_mean"
KEY = "X_phate_3d_top2_knn5"
COORDINATES = [f"{KEY}_1", f"{KEY}_2", f"{KEY}_3"]
ROWS = [9, 2, 6, 0, 4, 10]
FORMATS = pytest.mark.parametrize("file_format", ["h5ad", "csv"])


def _cell_table(exp: Experiment) -> Experiment:
    for index, well in enumerate(exp.well_list):
        converted(exp, well)
        segmented(exp, well)
        aligned(exp, well)
        measured(exp, well)
        cells = [4 * index + cell for cell in range(4)]
        measured_values(exp, well, "nuc_area", [30.0 + k for k in cells])
        measured_values(exp, well, "R0_pRB_nuc_mean", [1000.0 + 50 * k + 7 * (k % 3) for k in cells])
        measured_values(exp, well, "R1_cMyc_nuc_mean", [2000.0 + 30 * (5 * k % 12) for k in cells])
    exp = with_labels(with_filters(exp, [NOT_THE_BIGGEST]), [SIZE_LABEL])
    run.build_cell_table(exp, filters=["not the biggest"], labels=["size"])
    return exp


def _with_combat(exp: Experiment, empty: Sequence[int] = ()) -> None:
    adata = ad.read_h5ad(io.adata_path(exp))
    combat = np.asarray(adata.X, dtype=float) + 1
    combat[list(empty)] = np.nan
    adata.layers["combat"] = combat
    adata.write_h5ad(io.adata_path(exp))


def _cells(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [frame["well"].astype(str).to_numpy(), frame["label"].astype(int).to_numpy()], names=["well", "label"]
    )


def _keys(frame: pd.DataFrame) -> list[tuple[str, int]]:
    return sorted(_cells(frame))


def _text(values: Iterable[Any]) -> list[str | None]:
    return [None if pd.isna(value) else str(value) for value in values]


def _table(exp: Experiment) -> pd.DataFrame:
    table = pd.read_csv(io.fullest_df_path(exp))
    return table.set_axis(_cells(table))


def _standard(exp: Experiment) -> ad.AnnData:
    return ad.read_h5ad(io.adata_path(exp))


def _obs_columns(exp: Experiment) -> list[str]:
    return [column for column in _standard(exp).obs.columns if column != "included"]


def _matrix(adata: ad.AnnData, layer: str | None = None) -> pd.DataFrame:
    values = adata.X if layer is None else adata.layers[layer]
    return pd.DataFrame(np.asarray(values, dtype=float), columns=list(adata.var_names), index=_cells(adata.obs))


def _cells_exported(path: Path, file_format: str) -> pd.DataFrame:
    return ad.read_h5ad(path).obs if file_format == "h5ad" else pd.read_csv(path)


def _measurements(exp: Experiment, path: Path, file_format: str) -> pd.DataFrame:
    if file_format == "h5ad":
        return _matrix(ad.read_h5ad(path))
    table = pd.read_csv(path)
    obs_columns = set(_standard(exp).obs.columns)
    return table[[column for column in table.columns if column not in obs_columns]].set_axis(_cells(table))


def _z_scores(path: Path, file_format: str) -> pd.DataFrame:
    if file_format == "h5ad":
        return _matrix(ad.read_h5ad(path))
    table = pd.read_csv(path)
    columns = [column for column in table.columns if column.endswith("_zscore")]
    renamed = [column.removesuffix("_zscore") for column in columns]
    return table[columns].set_axis(renamed, axis=1).set_axis(_cells(table))


def _feature_columns(exp: Experiment, path: Path, file_format: str) -> list[str]:
    if file_format == "h5ad":
        return list(ad.read_h5ad(path).var_names)
    obs_columns = set(_standard(exp).obs.columns)
    return [column for column in pd.read_csv(path).columns if column not in obs_columns]


def _same_values(exported: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert not exported.index.duplicated().any()
    np.testing.assert_allclose(
        exported.to_numpy(dtype=float),
        expected.loc[exported.index, list(exported.columns)].to_numpy(dtype=float),
        rtol=1e-6,
    )


def _files_in_exports(exp: Experiment) -> list[Path]:
    folder = Path(exp.base_dir) / "exports"
    return sorted(path for path in folder.rglob("*") if path.is_file()) if folder.exists() else []


@FORMATS
def test_only_the_included_cells_are_exported(experiment: Experiment, file_format: str) -> None:
    exp = _cell_table(experiment)

    path = export_cells(exp, NAME, file_format=file_format)

    table = pd.read_csv(io.fullest_df_path(exp))
    assert path == Path(exp.base_dir) / "exports" / f"{NAME}.{file_format}"
    assert _keys(_cells_exported(path, file_format)) == _keys(table[table["included"]])


@FORMATS
def test_each_measurement_is_on_its_own_cell(experiment: Experiment, file_format: str) -> None:
    exp = _cell_table(experiment)

    path = export_cells(exp, NAME, file_format=file_format, values="measured")

    exported = _measurements(exp, path, file_format)
    assert set(VARIED) <= set(exported.columns)
    _same_values(exported, _table(exp))


@FORMATS
def test_each_z_score_is_on_its_own_cell(experiment: Experiment, file_format: str) -> None:
    exp = _cell_table(experiment)

    path = export_cells(exp, NAME, file_format=file_format)

    exported = _z_scores(path, file_format)
    assert set(VARIED) <= set(exported.columns)
    _same_values(exported, _matrix(_standard(exp)))


def test_both_puts_z_scores_in_x_and_measurements_in_the_measured_layer(experiment: Experiment) -> None:
    exp = _cell_table(experiment)

    exported = ad.read_h5ad(export_cells(exp, NAME, values="both", features=VARIED))

    assert list(exported.var_names) == VARIED
    _same_values(_matrix(exported), _matrix(_standard(exp)))
    _same_values(_matrix(exported, "measured"), _table(exp))


def test_a_csv_names_z_scores_with_zscore(experiment: Experiment) -> None:
    exp = _cell_table(experiment)
    zscore_names = [f"{feature}_zscore" for feature in VARIED]

    zscores = pd.read_csv(export_cells(exp, "zscores", file_format="csv", features=VARIED))
    both = pd.read_csv(export_cells(exp, "both", file_format="csv", values="both", features=VARIED))

    obs_columns = _obs_columns(exp)
    assert list(zscores.columns) == [*obs_columns, *zscore_names]
    assert list(both.columns) == [*obs_columns, *VARIED, *zscore_names]
    _same_values(both[VARIED].set_axis(_cells(both)), _table(exp))
    _same_values(both[zscore_names].set_axis(VARIED, axis=1).set_axis(_cells(both)), _matrix(_standard(exp)))


@FORMATS
def test_metadata_well_and_labels_come_with_every_cell(experiment: Experiment, file_format: str) -> None:
    exp = _cell_table(experiment)

    path = export_cells(exp, NAME, file_format=file_format)

    standard = _standard(exp)
    obs_columns = _obs_columns(exp)
    exported = _cells_exported(path, file_format)
    expected = standard.obs.set_axis(_cells(standard.obs)).loc[_cells(exported)]
    assert list(exported.columns[: len(obs_columns)]) == obs_columns
    for column in obs_columns:
        assert _text(exported[column]) == _text(expected[column]), column
    if file_format == "h5ad":
        names = pd.Series(standard.obs_names, index=_cells(standard.obs))
        assert list(exported.columns) == obs_columns
        assert list(exported.index) == names.loc[_cells(exported)].tolist()


@FORMATS
def test_named_features_only_are_exported_in_table_order(experiment: Experiment, file_format: str) -> None:
    exp = _cell_table(experiment)

    path = export_cells(
        exp, NAME, file_format=file_format, features=["R1_cMyc_nuc_mean", "nuc_area", "R0_pRB_nuc_mean"]
    )

    expected = VARIED if file_format == "h5ad" else [f"{feature}_zscore" for feature in VARIED]
    assert _feature_columns(exp, path, file_format) == expected


@FORMATS
def test_a_feature_without_z_scores_is_left_out_of_z_scores_only(experiment: Experiment, file_format: str) -> None:
    exp = _cell_table(experiment)
    standard = _standard(exp)
    table = pd.read_csv(io.fullest_df_path(exp))
    every_feature = [column for column in table.select_dtypes("number").columns if column not in standard.obs.columns]
    with_z_scores = [feature for feature in every_feature if feature in standard.var_names]

    measured_path = export_cells(exp, "measured", file_format=file_format, values="measured")
    zscores_path = export_cells(exp, "zscores", file_format=file_format)
    both_path = export_cells(exp, "both", file_format=file_format, values="both")

    suffix = "" if file_format == "h5ad" else "_zscore"
    zscore_names = [f"{feature}{suffix}" for feature in with_z_scores]
    assert NO_Z_SCORES in every_feature
    assert NO_Z_SCORES not in standard.var_names
    assert _feature_columns(exp, measured_path, file_format) == every_feature
    assert _feature_columns(exp, zscores_path, file_format) == zscore_names
    if file_format == "h5ad":
        both = ad.read_h5ad(both_path)
        assert list(both.var_names) == every_feature
        assert _matrix(both)[NO_Z_SCORES].isna().all()
        _same_values(_matrix(both)[with_z_scores], _matrix(standard))
        _same_values(_matrix(both, "measured"), _table(exp))
    else:
        both = pd.read_csv(both_path)
        assert _feature_columns(exp, both_path, file_format) == [*every_feature, *zscore_names]
        _same_values(both[every_feature].set_axis(_cells(both)), _table(exp))


@FORMATS
def test_corrected_takes_the_combat_layer(experiment: Experiment, file_format: str) -> None:
    exp = _cell_table(experiment)
    _with_combat(exp)

    path = export_cells(exp, NAME, file_format=file_format, corrected=True, features=VARIED)

    exported = _z_scores(path, file_format)
    assert list(exported.columns) == VARIED
    _same_values(exported, _matrix(_standard(exp), "combat"))


@FORMATS
def test_a_corrected_export_leaves_out_the_cells_without_corrected_values(
    experiment: Experiment, file_format: str
) -> None:
    exp = _cell_table(experiment)
    _with_combat(exp, empty=[1, 5])

    path = export_cells(exp, NAME, file_format=file_format, corrected=True)

    table = pd.read_csv(io.fullest_df_path(exp))
    kept = table[table["included"]].drop(index=[1, 5])
    assert _keys(_cells_exported(path, file_format)) == _keys(kept)


@FORMATS
def test_embeddings_export_only_the_subsampled_cells_with_their_coordinates(
    experiment: Experiment, file_format: str
) -> None:
    exp = _cell_table(experiment)
    subsampled(exp, ROWS, KEY)

    path = export_cells(exp, NAME, file_format=file_format, embeddings=[KEY])

    subsample = ad.read_h5ad(io.adata_sub_path(exp))
    expected = pd.DataFrame(subsample.obsm[KEY], index=_cells(subsample.obs), columns=COORDINATES)
    if file_format == "h5ad":
        exported = ad.read_h5ad(path)
        cells = exported.obs
        coordinates = pd.DataFrame(np.asarray(exported.obsm[KEY]), index=_cells(cells), columns=COORDINATES)
        assert KEY in exported.uns["embeddings"]
    else:
        cells = pd.read_csv(path)
        coordinates = cells[COORDINATES].set_axis(_cells(cells))
    assert _keys(cells) == _keys(subsample.obs)
    _same_values(coordinates, expected)


def test_an_existing_export_is_refused_without_overwrite_and_replaced_with_it(experiment: Experiment) -> None:
    exp = _cell_table(experiment)
    first = export_cells(exp, NAME, features=["nuc_area"])
    export_cells(exp, NAME, file_format="csv", features=["nuc_area"])

    with pytest.raises(FileExistsError):
        export_cells(exp, NAME, features=VARIED)
    kept = list(ad.read_h5ad(first).var_names)
    replaced = export_cells(exp, NAME, features=VARIED, overwrite=True)

    assert kept == ["nuc_area"]
    assert replaced == first
    assert list(ad.read_h5ad(replaced).var_names) == VARIED


@pytest.mark.parametrize(
    ("name", "settings", "arrange", "error"),
    [
        ("", {}, "", ValueError),
        ("   ", {}, "", ValueError),
        ("a/b", {}, "", ValueError),
        ("a\\b", {}, "", ValueError),
        ("a:b", {}, "", ValueError),
        ("a*b", {}, "", ValueError),
        ("a?b", {}, "", ValueError),
        ('a"b', {}, "", ValueError),
        ("a<b", {}, "", ValueError),
        ("a>b", {}, "", ValueError),
        ("a|b", {}, "", ValueError),
        ("cycling.", {}, "", ValueError),
        ("cycling ", {}, "", ValueError),
        (NAME, {"values": "measured", "corrected": True}, "combat", ValueError),
        (NAME, {"corrected": True}, "", ValueError),
        (NAME, {"embeddings": [KEY]}, "", FileNotFoundError),
        (NAME, {"embeddings": ["X_umap_2d_top2_n_neighbors5"]}, "subsample", ValueError),
        (NAME, {"features": ["nuc_area", "R9_cMyc_nuc_mean"]}, "", ValueError),
        (NAME, {"features": ["nuc_area", "sample_ID"]}, "", ValueError),
    ],
    ids=[
        "blank-name",
        "name-of-spaces",
        "slash",
        "backslash",
        "colon",
        "star",
        "question-mark",
        "double-quote",
        "less-than",
        "greater-than",
        "vertical-bar",
        "ends-in-a-dot",
        "ends-in-a-space",
        "corrected-measured-only",
        "corrected-without-a-combat-layer",
        "embeddings-without-a-subsample",
        "unknown-embedding",
        "unknown-feature",
        "a-name-obs-holds",
    ],
)
def test_each_refusal_writes_nothing(
    experiment: Experiment, name: str, settings: dict[str, Any], arrange: str, error: type[Exception]
) -> None:
    exp = _cell_table(experiment)
    if arrange == "combat":
        _with_combat(exp)
    if arrange == "subsample":
        subsampled(exp, ROWS, KEY)

    with pytest.raises(error):
        export_cells(exp, name, **settings)

    assert _files_in_exports(exp) == []


def test_export_refuses_a_subsample_from_another_cell_table(experiment: Experiment) -> None:
    exp = _cell_table(experiment)
    subsampled(exp, ROWS, KEY)
    measured_values(exp, "B02", "R0_pRB_nuc_mean", [1500.0, 1600.0, 1700.0, 1800.0])
    run.build_cell_table(exp, filters=["not the biggest"], labels=["size"])

    with pytest.raises(ValueError):
        export_cells(exp, NAME, embeddings=[KEY])

    assert not io.export_path(exp, NAME, "h5ad").exists()


def test_a_cell_missing_from_the_anndata_is_refused(experiment: Experiment) -> None:
    exp = _cell_table(experiment)
    table = ad.read_h5ad(io.adata_path(exp))
    table[1:].copy().write_h5ad(io.adata_path(exp))

    with pytest.raises(ValueError):
        export_cells(exp, NAME, values="measured")

    assert not io.export_path(exp, NAME, "h5ad").exists()


def test_a_cell_missing_from_the_table_is_refused(experiment: Experiment) -> None:
    exp = _cell_table(experiment)
    table = pd.read_csv(io.fullest_df_path(exp))
    table.iloc[1:].to_csv(io.fullest_df_path(exp), index=False)

    with pytest.raises(ValueError):
        export_cells(exp, NAME, values="measured")

    assert not io.export_path(exp, NAME, "h5ad").exists()


def test_an_x_feature_missing_from_the_table_is_refused(experiment: Experiment) -> None:
    exp = _cell_table(experiment)
    assert "nuc_area" in ad.read_h5ad(io.adata_path(exp)).var_names
    table = pd.read_csv(io.fullest_df_path(exp))
    table.drop(columns="nuc_area").to_csv(io.fullest_df_path(exp), index=False)

    with pytest.raises(ValueError):
        export_cells(exp, NAME)

    assert not io.export_path(exp, NAME, "h5ad").exists()
