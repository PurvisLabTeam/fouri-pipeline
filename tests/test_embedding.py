"""Embedding saves the runs that finished when another run fails."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from fouri import io
from fouri.config import Experiment
from fouri.embedding import Method, Run, embed_grid
from fouri.report import StepFailed

pytestmark = pytest.mark.slow  # runs UMAP

FEATURES = ["R0_pRB_nuc_mean", "R0_RB_nuc_mean", "R1_cMyc_nuc_mean", "nuc_area"]


def _small_subsample(exp: Experiment, cells: int = 60) -> None:
    """Write a subsample file of random values: three samples, four features."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"sample_ID": np.repeat([1, 2, 3], cells // 3)}, index=[str(i) for i in range(cells)])
    adata = ad.AnnData(X=rng.normal(size=(cells, len(FEATURES))), obs=obs, var=pd.DataFrame(index=FEATURES))
    adata.write_h5ad(io.adata_sub_path(exp))


def test_the_runs_that_finished_are_saved(experiment: Experiment) -> None:
    _small_subsample(experiment)
    runs = [
        Run(Method.UMAP, 2, 2, {"n_neighbors": 5}),
        Run(Method.UMAP, 2, 2, {"n_neighbors": 1}),  # UMAP refuses fewer than 2 neighbours
    ]

    with pytest.raises(StepFailed) as raised:
        embed_grid(experiment, runs, FEATURES[:3], ranking_k=10, n_workers=2)

    saved = ad.read_h5ad(io.adata_sub_path(experiment))
    assert "X_umap_2d_top2_n_neighbors5" in saved.obsm
    assert "X_umap_2d_top2_n_neighbors5" in saved.uns["embeddings"]
    assert "X_umap_2d_top2_n_neighbors1" not in saved.obsm
    assert raised.value.failed == ["X_umap_2d_top2_n_neighbors1"]
