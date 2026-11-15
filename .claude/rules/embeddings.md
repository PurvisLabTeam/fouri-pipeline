---
paths:
  - "src/fouri/{embedding,plots,run}.py"
  - "src/fouri/app/embeddings.py"
  - "src/fouri/app/pages/{embedding,run}.py"
  - "tests/test_embedding.py"
---

# Embeddings (08)

Loaded with the root `CLAUDE.md` when a file matching `paths` is read.

## Embedding set-up in experiment.json

`app.embedding` is the experiment's one embedding set-up, saved by the Embedding page's set-up tab: `picked`
(column names), `samples` (sample IDs), `ranking_k`, `top_n` (a list), `methods` (always both `PHATE` and
`UMAP`, each with `dims`, empty when the method makes no run, and `settings`, a list of values per setting as
`fouri.embedding` passes them: `knn`, `t`, `gamma`, `decay`, `n_pca`; `n_neighbors`, `min_dist`, `metric`),
`skipped` (the `run_key` of each run unticked), `seed` and `values` (`original` until batch correction).
`fouri.app.embeddings.saved_setup` reads it, None when missing, and refuses one with a key missing or of the
wrong type; `embedding_settings` turns it into what the Run page's Embedding card sends `make_embeddings`: every
run of the set-up not skipped, `picked`, `samples`, `ranking_k` and `seed`.

## fouri.embedding

`fouri/embedding.py`, one call per step, all on `standard_adata_sub.h5ad` (`io.adata_sub_path`):

- `subsample(exp, cells_per_sample=2000, seed=42)`: sketchKH kernel herding grouped by `sample_ID`, on the
  included cells of `standard_adata.h5ad` (every cell when it has no `included` column); a sample the filters
  left with no included cell is refused, by name, before anything is sketched or written; lowered to the
  smallest sample's count of included cells (sketch would do it silently), recorded in `.uns["subsample"]`. Every feature stays in `.X`, so an embedding can be colored by any measurement.
- `rank_features(adata, features, samples=None, k=100)`: the picked features decide which cells are
  neighbors (k of them), and delve's `feature_select` scores every feature by how smoothly it changes
  between neighbors, lower is better. For the picked features this is exactly old 08's Laplacian ranking.
  delve's own `delve_fs` is not used: its test only finds changing features when most features don't change,
  and on a 4i panel nearly every column follows the cell cycle (nothing found in 21 of 22 settings tried).
- `embed(path, method, features, *, samples, dims, settings, seed=42)`: one PHATE or UMAP run (`Method`),
  2D or 3D, returning a row per cell of the file (NaN for samples left out). Seeded, and fitted on one maths
  thread (`threadpoolctl`): PHATE's coordinates otherwise shift with the thread count, mainly on its third
  axis, and a seeded UMAP runs on one core anyway. `phate` and `umap` are imported inside it.
- `make_runs(method, dims, top_n, settings)` builds a `Run` (method, dims, top_n, settings) per combination;
  `embed_grid(exp, runs, picked, *, samples, ranking_k, seed, n_workers)` ranks once, runs every `Run` side by
  side with `parallel_map` on its top-ranked picked features, and saves each under
  `X_{method}_{dims}d_top{n}_{setting}{value}…` (settings sorted), the name `run_key(run)` gives, which the
  app's set-up tab also uses, so the set-up, the Run page and `embed_grid` agree. The same run with the same picked
  features, samples, ranking k and seed replaces its embedding (the result is identical); other inputs get
  `_2`, `_3`. `.uns["embeddings"][key]` records method, dims, top_n, settings, picked, the samples actually
  used, ranking_k, seed, and the features used with their scores.
- `plots.plot_embedding(adata, key, color_by)`: plotly, 2D (WebGL) or 3D; a measurement or a number column
  on a Viridis scale from the 1st to the 99th percentile, a text or category column one trace and color per
  group (`GROUP_COLORS`, the app's).
