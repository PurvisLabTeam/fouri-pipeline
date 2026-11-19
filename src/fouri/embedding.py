"""@TODO."""

import functools
import itertools
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
from delve import feature_select
from sketchKH import sketch

from fouri.config import Experiment
from fouri.io import COMBAT_LAYER, adata_path, adata_sub_path, written_whole
from fouri.parallel import Dropped, Failed, check_results, parallel_map
from fouri.report import say

EMBED_MEM_PER_WORKER = 3e9


class Method(StrEnum):
    PHATE = "PHATE"
    UMAP = "UMAP"


def subsample(exp: Experiment, cells_per_sample: int = 2000, seed: int = 42) -> ad.AnnData:
    rpath = adata_path(exp)
    adata = ad.read_h5ad(rpath)
    adata_id = adata.uns.get("cell_table", {}).get("id")
    if adata_id is None:
        raise ValueError("The adata cell table was built by an older version. Build the cell table again.")

    selected = adata if "included" not in adata.obs.columns else adata[adata.obs["included"]]
    adata_samples = set(adata.obs["sample_ID"])
    selected_samples = set(selected.obs["sample_ID"])
    if adata_samples != selected_samples:
        missing_samples = sorted(str(s) for s in (adata_samples - selected_samples))
        raise ValueError(
            "The folowing samples have no cells left after filters so they cannot be subsampled. "
            "Either change the filters on the Filters page, or leave the sample(s) out of the experiment: "
            f"{', '.join(missing_samples)}"
        )
    cells_per_sample_actual = selected.obs["sample_ID"].value_counts()
    min_cells = cells_per_sample_actual.min()
    min_cells_sample = typing.cast(int, cells_per_sample_actual.idxmin())
    if min_cells < cells_per_sample:
        say(f"Subsampling to {min_cells} cells because sample {min_cells_sample} only has that many cells.")
        cells_per_sample = min_cells
    _, subsampled = sketch(
        selected, sample_set_key="sample_ID", num_subsamples=cells_per_sample, frequency_seed=seed, n_jobs=-1
    )
    subsampled.uns["subsample"] = {
        "cell_table": adata.uns["cell_table"]["id"],
        "cells_per_sample": cells_per_sample,
        "seed": seed,
    }
    with written_whole(adata_sub_path(exp)) as partial:
        subsampled.write_h5ad(partial)
    say(f"Wrote subsampled dataframe with {len(cells_per_sample_actual)} samples and {subsampled.n_obs:,} cells.")
    return subsampled


def rank_features(
    adata: ad.AnnData,
    features: Sequence[str],
    samples: Sequence[int] | None = None,
    k: int = 100,
    values: str = "original",
) -> pd.Series:
    """@TODO."""
    _check_features(adata, features)
    if values == "combat":
        _check_corrected(adata, samples)
    matrix = _values(adata, values)[_sample_mask(adata, samples)]
    if values == "combat":
        matrix = matrix[_has_values(matrix)]
    selected_features = feature_select(
        matrix, feature_names=adata.var_names.to_numpy(), dyn_feats=features, k=k, n_jobs=-1
    )
    return pd.Series(selected_features["DELVE"], name="score")


def embed(
    path: str | Path,
    method: Method,
    features: Sequence[str],
    *,
    samples: Sequence[int] | None = None,
    dims: int = 3,
    settings: Mapping[str, typing.Any] | None = None,
    seed: int = 42,
    values: str = "original",
) -> npt.NDArray[np.float64]:
    if dims not in (2, 3):
        raise ValueError(f"Invalid number of dimensions: {dims}. Pick either 2 or 3.")

    path = Path(path)
    adata = ad.read_h5ad(path)
    _check_features(adata, features)
    if values == "combat":
        _check_corrected(adata, samples)
    mask = _sample_mask(adata, samples)
    columns = adata.var_names.get_indexer(pd.Index(features))
    selected = _values(adata, values)[np.ix_(mask, columns)]
    rows = np.flatnonzero(mask)
    if values == "combat":
        kept = _has_values(selected)
        selected, rows = selected[kept], rows[kept]

    settings = settings if settings is not None else {}
    match method:
        case Method.PHATE:
            import phate

            model = phate.PHATE(
                n_components=dims,
                random_state=seed,
                verbose=0,
                n_jobs=1,
                **settings,
            )
        case Method.UMAP:
            import umap

            model = umap.UMAP(n_components=dims, random_state=seed, n_jobs=1, **settings)
        case _:
            raise ValueError(f"Unkown embedding method: {method}")

    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        results = model.fit_transform(selected)

    final = np.full((adata.n_obs, dims), np.nan)
    final[rows] = results
    return final


def _check_features(adata: ad.AnnData, features: Sequence[str]) -> None:
    not_in_file = [name for name in features if name not in adata.var_names]
    if not_in_file:
        raise ValueError(
            "The following features were selected for ranking but do not exist in the anndata file:"
            + "".join(f"\n{name}" for name in not_in_file)
            + "\n"
        )


def _sample_mask(adata: ad.AnnData, samples: Sequence[int] | None) -> npt.NDArray[np.bool]:
    if samples is None:
        return np.ones(adata.n_obs, dtype=bool)
    cells = adata.obs["sample_ID"].isin(samples).to_numpy()
    if not np.any(cells):
        raise ValueError(f"Selecting on the following samples gave 0 cells: {', '.join(map(str, samples))}")
    return cells


def _values(adata: ad.AnnData, values: str) -> npt.NDArray[np.float64]:
    if values == "original":
        return np.asarray(adata.X, dtype=np.float64)
    elif values == "combat":
        if COMBAT_LAYER not in adata.layers:
            raise ValueError(
                "The subsample has no ComBat-corrected values. Run Batch effects or embed the original values."
            )
        return np.asarray(adata.layers[COMBAT_LAYER], dtype=np.float64)
    else:
        raise ValueError(f"Unrecognized value type: {values}. Must be one of 'original' or 'combat'.")


def _check_corrected(adata: ad.AnnData, samples: Sequence[int] | None) -> None:
    has_values = _has_values(_values(adata, "combat"))
    chosen = _sample_mask(adata, samples)
    if has_values[chosen].any():
        return
    sample_ids = adata.obs["sample_ID"].to_numpy()
    with_values = sorted({int(sample) for sample in sample_ids[has_values]})
    if not with_values:
        raise ValueError("No cell of the subsample has ComBat-corrected values. Embed the original values.")
    empty = sorted({int(sample) for sample in sample_ids[chosen]})
    raise ValueError(
        f"No cell of sample{'s' if len(empty) > 1 else ''} {', '.join(map(str, empty))} has ComBat-corrected "
        f"values. Samples with corrected values: {', '.join(map(str, with_values))}. Choose among them, or embed "
        "the original values."
    )


def _has_values(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.bool]:
    return ~np.isnan(matrix).any(axis=1)


@dataclass(frozen=True)
class Run:
    method: Method
    dims: int
    top_n: int
    settings: Mapping[str, typing.Any]


def make_runs(
    method: Method,
    dims: Sequence[int],
    top_n: Sequence[int],
    settings: Mapping[str, Sequence[typing.Any]],
) -> list[Run]:
    """@TODO."""
    names = list(settings)
    combinations = [dict(zip(names, values, strict=True)) for values in itertools.product(*settings.values())]
    return [
        Run(method, n_dims, n_features, combination)
        for n_dims, n_features, combination in itertools.product(dims, top_n, combinations)
    ]


def embed_grid(
    exp: Experiment,
    runs: Sequence[Run],
    picked: Sequence[str],
    *,
    samples: Sequence[int] | None = None,
    ranking_k: int = 100,
    seed: int = 42,
    n_workers: int | None = None,
    values: str = "original",
) -> list[str]:
    """@TODO."""
    path = adata_sub_path(exp)
    adata = ad.read_h5ad(path)
    ranking = rank_features(adata, picked, samples=samples, k=ranking_k, values=values)
    if values == "combat":
        left_out = int((~_has_values(_values(adata, values)[_sample_mask(adata, samples)])).sum())
        if left_out:
            say(f"Left out, with no corrected values: {left_out:,} cells.")
    picked_set = set(picked)
    order = [feature for feature in ranking.index if feature in picked_set]

    out_of_range = [run for run in runs if not 1 <= run.top_n <= len(order)]
    if out_of_range:
        raise ValueError(
            f"A run can use 1 to {len(order)} features, but these ask for something else:"
            + "".join(f"\n{run_key(run)}" for run in out_of_range)
            + "\n"
        )

    coordinates = parallel_map(
        function=functools.partial(_embed_job, path=path, samples=samples, seed=seed, values=values),
        items=[(run, order[: run.top_n]) for run in runs],
        desc="Embedding",
        unit="run",
        n_workers=n_workers,
        mem_per_worker=EMBED_MEM_PER_WORKER,
        names=[run_key(run) for run in runs],
    )

    sample_ids = adata.obs["sample_ID"].to_numpy()
    embeddings = adata.uns.setdefault("embeddings", {})
    keys = []
    for run, coords in zip(runs, coordinates, strict=True):
        if isinstance(coords, (Failed, Dropped)):
            continue
        features = order[: run.top_n]
        embedded = ~np.isnan(np.asarray(coords, dtype=np.float64)).any(axis=1)
        record = {
            "method": str(run.method),
            "dims": run.dims,
            "top_n": run.top_n,
            "settings": dict(run.settings),
            "picked": list(picked),
            "samples": sorted({int(sample) for sample in sample_ids[embedded]}),
            "ranking_k": ranking_k,
            "seed": seed,
            "values": values,
            "features": features,
            "scores": ranking[features].to_numpy(),
        }
        key = _free_key(adata, run_key(run), record)
        adata.obsm[key] = coords
        embeddings[key] = record
        keys.append(key)

    if keys:  # a file where every run failed is left as it was
        with written_whole(path) as partial:
            adata.write_h5ad(partial)
        say(
            f"Saved {len(keys)} embedding(s) to {path.name}, under these names in .obsm (the cell coordinates) "
            'and .uns["embeddings"] (the settings that made each):' + "".join(f"\n{key}" for key in keys) + "\n"
        )
    check_results(coordinates, unit="run")
    return keys


def run_key(run: Run) -> str:
    # e.g. X_phate_3d_top10_gamma1_knn80_tauto; settings sorted by name, so a run keeps its key
    parts = [f"X_{run.method.lower()}", f"{run.dims}d", f"top{run.top_n}"]
    parts += [f"{name}{value}" for name, value in sorted(run.settings.items())]
    return "_".join(parts)


def _embed_job(
    job: tuple[Run, list[str]], path: Path, samples: Sequence[int] | None, seed: int, values: str
) -> npt.NDArray[np.float64]:
    run, features = job
    return embed(
        path, run.method, features, samples=samples, dims=run.dims, settings=run.settings, seed=seed, values=values
    )


def _free_key(adata: ad.AnnData, key: str, record: Mapping[str, typing.Any]) -> str:
    embeddings = adata.uns.get("embeddings", {})
    candidate, number = key, 1
    while candidate in adata.obsm and not (candidate in embeddings and _same_inputs(embeddings[candidate], record)):
        number += 1
        candidate = f"{key}_{number}"
    return candidate


def _same_inputs(stored: Mapping[str, typing.Any], record: Mapping[str, typing.Any]) -> bool:
    return (
        [str(name) for name in stored["picked"]] == list(record["picked"])
        and [int(sample) for sample in stored["samples"]] == list(record["samples"])
        and int(stored["ranking_k"]) == record["ranking_k"]
        and int(stored["seed"]) == record["seed"]
        and str(stored.get("values", "original")) == record["values"]
    )
