"""Embedding saves the runs that finished when another run fails."""

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from fouri import embedding, io
from fouri.config import Experiment
from fouri.embedding import Method, Run, embed_grid, subsample
from fouri.report import StepFailed

FEATURES = ["R0_pRB_nuc_mean", "R0_RB_nuc_mean", "R1_cMyc_nuc_mean", "nuc_area"]


def _small_subsample(exp: Experiment, cells: int = 60) -> None:
    """Write a subsample file of random values: three samples, four features."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"sample_ID": np.repeat([1, 2, 3], cells // 3)}, index=[str(i) for i in range(cells)])
    adata = ad.AnnData(X=rng.normal(size=(cells, len(FEATURES))), obs=obs, var=pd.DataFrame(index=FEATURES))
    adata.write_h5ad(io.adata_sub_path(exp))


def _small_standard_adata(exp: Experiment, excluded: list[int]) -> None:
    rng = np.random.default_rng(0)
    cells = 60
    obs = pd.DataFrame(
        {"sample_ID": pd.Categorical(np.repeat([1, 2, 3], 20)), "included": ~np.isin(np.arange(cells), excluded)},
        index=[f"C_{i}" for i in range(cells)],
    )
    adata = ad.AnnData(X=rng.normal(size=(cells, len(FEATURES))), obs=obs, var=pd.DataFrame(index=FEATURES))
    adata.uns["cell_table"] = {"id": "0123456789abcdef0123456789abcdef"}
    adata.write_h5ad(io.adata_path(exp))


@pytest.mark.slow  # runs UMAP
def test_the_runs_that_finished_are_saved(experiment: Experiment) -> None:
    _small_subsample(experiment)
    runs = [
        Run(Method.UMAP, 2, 2, {"n_neighbors": 5}),
        Run(Method.UMAP, 2, 2, {"n_neighbors": 1}),  # UMAP refuses fewer than 2 neighbors
    ]

    with pytest.raises(StepFailed) as raised:
        embed_grid(experiment, runs, FEATURES[:3], ranking_k=10, n_workers=2)

    saved = ad.read_h5ad(io.adata_sub_path(experiment))
    assert "X_umap_2d_top2_n_neighbors5" in saved.obsm
    assert "X_umap_2d_top2_n_neighbors5" in saved.uns["embeddings"]
    assert "X_umap_2d_top2_n_neighbors1" not in saved.obsm
    assert raised.value.failed == ["X_umap_2d_top2_n_neighbors1"]


def test_the_subsample_takes_only_included_cells(experiment: Experiment) -> None:
    excluded = [0, 1, 2, 20, 21, 22, 23, 24, *range(40, 48)]
    _small_standard_adata(experiment, excluded)

    subsample(experiment)

    saved = ad.read_h5ad(io.adata_sub_path(experiment))
    assert not set(saved.obs_names) & {f"C_{i}" for i in excluded}
    assert saved.n_obs == 3 * 12


def test_a_sample_with_no_included_cell_is_refused(experiment: Experiment) -> None:
    _small_standard_adata(experiment, list(range(40, 60)))

    with pytest.raises(ValueError):
        subsample(experiment)

    assert not io.adata_sub_path(experiment).exists()


UMAP_RUN = Run(Method.UMAP, 2, 2, {"n_neighbors": 5})


def _with_corrected(exp: Experiment, without: int) -> None:
    sub = ad.read_h5ad(io.adata_sub_path(exp))
    layer = np.asarray(sub.X, dtype=float).copy()
    layer[sub.obs["sample_ID"].to_numpy() == without] = np.nan
    sub.layers[io.COMBAT_LAYER] = layer
    sub.write_h5ad(io.adata_sub_path(exp))


def test_the_record_lists_only_samples_with_an_embedded_cell(experiment: Experiment, stand_in_umap: object) -> None:
    _small_subsample(experiment)
    _with_corrected(experiment, without=3)

    keys = embed_grid(experiment, [UMAP_RUN], FEATURES[:3], ranking_k=10, values="combat")

    record = ad.read_h5ad(io.adata_sub_path(experiment)).uns["embeddings"][keys[0]]
    assert [int(sample) for sample in record["samples"]] == [1, 2]


def test_coordinates_land_on_their_own_cells(experiment: Experiment, stand_in_umap: object) -> None:
    _small_subsample(experiment)

    keys = embed_grid(experiment, [UMAP_RUN], FEATURES[:3], samples=[1, 2], ranking_k=10)

    sub = ad.read_h5ad(io.adata_sub_path(experiment))
    used = [str(name) for name in sub.uns["embeddings"][keys[0]]["features"]]
    chosen = sub.obs["sample_ID"].isin([1, 2]).to_numpy()
    coordinates = np.asarray(sub.obsm[keys[0]])
    values = np.asarray(sub.X, dtype=float)[:, sub.var_names.get_indexer(used)]
    np.testing.assert_array_equal(coordinates[chosen], values[chosen])
    assert np.isnan(coordinates[~chosen]).all()


def test_samples_not_chosen_are_not_ranked(experiment: Experiment, monkeypatch: pytest.MonkeyPatch) -> None:
    _small_subsample(experiment)
    adata = ad.read_h5ad(io.adata_sub_path(experiment))
    ranked_rows = []
    real_feature_select = embedding.feature_select

    def record_rows(matrix: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        ranked_rows.append(len(matrix))
        return real_feature_select(matrix, *args, **kwargs)

    monkeypatch.setattr(embedding, "feature_select", record_rows)

    embedding.rank_features(adata, FEATURES[:3], samples=[1], k=10)

    assert ranked_rows == [int((adata.obs["sample_ID"] == 1).sum())]


def test_a_top_n_beyond_the_features_used_is_refused(experiment: Experiment, stand_in_umap: object) -> None:
    _small_subsample(experiment)
    before = io.adata_sub_path(experiment).read_bytes()

    with pytest.raises(ValueError):
        embed_grid(experiment, [Run(Method.UMAP, 2, 4, {"n_neighbors": 5})], FEATURES[:3], ranking_k=10)

    assert io.adata_sub_path(experiment).read_bytes() == before


@pytest.mark.slow
def test_an_embedding_with_other_inputs_takes_a_new_key(experiment: Experiment) -> None:
    _small_subsample(experiment)

    first = embed_grid(experiment, [UMAP_RUN], FEATURES[:3], ranking_k=10, n_workers=1)
    second = embed_grid(experiment, [UMAP_RUN], FEATURES[1:], ranking_k=10, n_workers=1)

    sub = ad.read_h5ad(io.adata_sub_path(experiment))
    assert first == ["X_umap_2d_top2_n_neighbors5"]
    assert second == ["X_umap_2d_top2_n_neighbors5_2"]
    assert set(sub.obsm) == {"X_umap_2d_top2_n_neighbors5", "X_umap_2d_top2_n_neighbors5_2"}
