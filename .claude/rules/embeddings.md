---
paths:
  - "src/fouri/{embedding,embedding_setup,plots,run}.py"
  - "src/fouri/app/{embeddings,embedding_figures}.py"
  - "src/fouri/app/pages/{embedding,run}.py"
  - "tests/test_embedding.py"
---

# Embeddings

Loaded with the root `CLAUDE.md` when a file matching `paths` is read.

## Embedding set-up in experiment.json

`app.embedding` is the experiment's one embedding set-up, saved by the Embedding page's set-up tab: `picked`
(column names), `samples` (sample IDs), `ranking_k`, `top_n` (a list), `methods` (always both `PHATE` and
`UMAP`, each with `dims`, empty when the method makes no run, and `settings`, a list of values per setting as
`fouri.embedding` passes them: `knn`, `t`, `gamma`, `decay`, `n_pca`; `n_neighbors`, `min_dist`, `metric`),
`skipped` (the `run_key` of each run unticked), `seed` and `values` (`original` or `combat`). The set-up tab
offers Values, Original or ComBat-corrected, only when `standard_adata_sub.h5ad` holds `layers["combat"]`, ranks
the features with the choice and saves the choice; without the layer the tab saves `original`, and states so under
Save when the saved set-up said `combat`.
`fouri.embedding_setup` holds `saved_setup`, which reads the set-up, None when missing, and refuses one with a key
missing or of the wrong type, and `embedding_settings`, which turns the set-up into what the Run page's Embedding
card sends `make_embeddings`: every run of the set-up not skipped, `picked`, `samples`, `ranking_k`, `seed` and
`values`. The app imports both from there. The Embedding step given no `runs` key, as typed after
`python -m fouri.run`, uses the same `embedding_settings`: `fouri.run.run_step` resolves the saved set-up before
checking the settings and writing the started line, each setting given replacing the set-up's of the same key, so
the run record holds the settings the step runs with, as from the Run page. No set-up saved, or a set-up with every
run unticked, is refused there; runs given, an empty list included, are passed as given.

The view tab shows a Corrected values switch with Color by marker when the subsample holds `layers["combat"]`;
choosing an embedding sets the switch to whether the record's `values` is `combat`, and with the switch on the
markers are colored by their ComBat-corrected z-score (`plot_embedding`'s `layer`). The record lists Values,
Original or ComBat-corrected (Original for a record without `values`). The view tab notes, above an embedding, that
the subsample holding it was cut from an earlier cell table (`fouri.adata.is_current_subsample`): `Made from an
earlier cell table. Run Embedding with Overwrite to update it.`

## fouri.embedding

`fouri/embedding.py`, one call per step, all on `standard_adata_sub.h5ad` (`io.adata_sub_path`):

- `subsample(exp, cells_per_sample=2000, seed=42)`: sketchKH kernel herding grouped by `sample_ID`, on the
  included cells of `standard_adata.h5ad` (every cell when it has no `included` column); a sample the filters
  left with no included cell is refused, by name, before anything is sketched or written; lowered to the
  smallest sample's count of included cells (sketch would do it silently), recorded in `.uns["subsample"]`, with the
  cell table's id in `.uns["subsample"]["cell_table"]`; a table without an id is refused. `subsample` is
  `cut_subsample` (in memory) then `write_subsample`. `run.make_embeddings` reusing a subsample and Export refuse one
  cut from another cell table or holding no id (`fouri.adata.check_subsample`), the message naming Embedding with
  Overwrite; Batch effects corrects the cell table past such a subsample and writes no layer into it
  (`.claude/rules/batch-effects.md`). Every feature stays in `.X`, so an embedding can be colored by any measurement.
- `run.make_embeddings` refuses every setting before any file is written: it builds the runs (an unknown method
  refused), cuts the new subsample in memory (with Overwrite or no file) or reads the existing one, checks it with
  `check_ranking` and `check_runs`, and writes a newly cut subsample only after both pass, so a refusal leaves the old
  subsample and its embeddings as they were. `check_ranking(adata, features, samples, k, values)` refuses unknown
  `values`, a feature not in `.X`, ComBat values without the layer or without corrected cells in the chosen samples,
  chosen samples with no cell, and a `ranking_k` below 4 (`MIN_RANKING_K`, delve's minimum) or not below the number of
  cells ranked (the chosen samples' cells, those with corrected values for `combat`): `ranking_k must be 4 or more and
  below the number of cells ranked (60); got 3.` `check_runs(runs, picked)` refuses a `top_n` outside 1 to the number
  of distinct features in `picked`. `rank_features` calls `check_ranking` and `embed_grid` calls `check_runs`, so the
  set-up tab's ranking gives the same refusal.
  The cells selected keep their rows of `layers["combat"]` when Batch effects has run.
- `values`, taken by `rank_features`, `embed`, `embed_grid` and, after `seed`, `run.make_embeddings`: `original`
  reads `.X`, `combat` reads `layers["combat"]`, written by Batch effects (`.claude/rules/batch-effects.md`); any
  other value is refused. With `combat`, a subsample without the layer is refused, and so are chosen samples with
  no corrected cell, before ranking or embedding, the message naming the samples that have corrected values; a
  cell with no corrected value is excluded from the ranking and the embedding, with empty coordinates, and
  `embed_grid` states how many.
- `rank_features(adata, features, samples=None, k=100, values="original")`: the picked features decide which
  cells are neighbors (k of them), and delve's `feature_select` scores every feature by how smoothly it changes
  between neighbors, lower is better. For the picked features this is exactly the pre-refactor notebooks'
  Laplacian ranking. delve's own `delve_fs` is not used: its test only finds changing features when most features
  don't change, and on a 4i panel nearly every column follows the cell cycle (nothing found in 21 of 22 settings
  tried).
- `embed(path, method, features, *, samples, dims, settings, seed=42, values="original")`: one PHATE or UMAP
  run (`Method`), 2D or 3D, returning a row per cell of the file (NaN for samples left out). Seeded, and fitted on one maths
  thread (`threadpoolctl`): PHATE's coordinates otherwise shift with the thread count, mainly on its third
  axis, and a seeded UMAP runs on one core anyway. `phate` and `umap` are imported inside it.
- `make_runs(method, dims, top_n, settings)` builds a `Run` (method, dims, top_n, settings) per combination;
  `embed_grid(exp, runs, picked, *, samples, ranking_k, seed, n_workers, values)` ranks once, runs every `Run` side by
  side with `parallel_map` on its top-ranked picked features, and saves each under
  `X_{method}_{dims}d_top{n}_{setting}{value}…` (settings sorted), the name `run_key(run)` gives, which the app's
  set-up tab also uses, so the set-up, the Run page and `embed_grid` agree. The same run with the same picked features,
  samples, ranking k, seed and values replaces its embedding (the result is identical; a record without `values` counts
  as `original`); other inputs get `_2`, `_3`. `.uns["embeddings"][key]` records method, dims, top_n, settings, picked,
  the samples with at least one embedded cell, ranking_k, seed, values, and the features used with their scores.
- `plots.plot_embedding(adata, key, color_by, *, title=None, layer=None)`: plotly, 2D (WebGL) or 3D; a
  measurement or a number column on a Viridis scale from the 1st to the 99th percentile, a text or category column
  one trace and color per group (`GROUP_COLORS`, the app's). With `layer`, `color_by` must be a measurement and is
  read from that layer (a column describing the cells is refused): the cells with a value on the scale, the
  colorbar titled `corrected z-score` and the percentiles over the cells with a value; the cells with no value, when
  any, a trace of their own drawn first in `NO_VALUE_COLOR` (light gray), named `No corrected value`, with no
  colorbar.
