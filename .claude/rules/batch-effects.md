---
paths:
  - "src/fouri/batch.py"
---

# Batch effects

Loaded with the root `CLAUDE.md` when a file matching `paths` is read.

`fouri.batch.correct_batches(exp, batch, keep=None, log_intensities=True)` reads `standard_adata.h5ad` and works
on the included cells (every cell when the table has no `included` column).

- Batches and conditions: `batch_labels` gives each cell's batch, `well`, `plate row` or `plate column` (from
  `well`), or any `.obs` column. `condition_labels` joins the `keep` columns' values with ` · `; `keep=None` means
  the condition columns with a value in at least one well, and an empty `keep` puts every cell in one condition,
  `all cells`. A batch or keep column `.obs` lacks is refused.
- Used conditions: a condition with cells in two or more batches is used (`used_conditions`); the others are
  excluded from the metrics and the correction and named with their cell counts. Batches linked through used
  conditions form groups (`batch_groups`); with more than one group, each group is corrected on its own and the
  groups are named.
- Metrics (`batch_metrics`, a `BatchMetrics`), on the used cells, within each condition: per feature the share of
  variance explained by batch (between-batch over total sum of squares), the median share over features and the
  count of features over `SHARE_OVER` (10 %); a random forest (100 trees, `random_state` 0) predicting batch,
  balanced accuracy over 3 stratified folds on at most 5,000 cells (`default_rng(0)`), averaged over the
  conditions weighted by cell count, beside chance (1 over the condition's number of batches); a condition with a
  batch under 3 sampled cells is not scored, and `classifier` is None when no condition is; the median share
  explained by condition, None with one condition; `top`, the 15 features with the largest batch share.
- Before, after, reference: the metrics are computed on `.X`, on the corrected z-scores, and on a reference,
  ComBat run with the batches shuffled within each condition (`default_rng(0)`) and scored against the shuffled
  batches, all in `.X`'s units with `log_intensities` on or off. ComBat rescales each batch's variance, which
  alone makes batches separable on heavy-tailed features; the reference score measures that effect alone.
- ComBat: `combat` calls `scanpy.pp.combat` once per group, the conditions passed as drop-first numeric dummy
  covariates (scanpy's categorical covariates give a singular design matrix), no covariate with one condition.
  The adjustment is the mean of the corrected z-scores minus `.X` per batch and feature, in SD units.
- Values corrected: ComBat and the reference run on the included cells' measurements in their own units, read
  from `fullest_df.csv` and matched to the cells of `standard_adata.h5ad` by well and label, for `.X`'s features.
  With `log_intensities`, `numpy.log1p` is applied before ComBat and `numpy.expm1` after to `log_columns(var)`:
  the features of `.var` kind `channel` with a measure in `LOG_MEASURES` (`nuc_mean`, `ring_mean`,
  `total_nuc_protein`), chosen by `.var` and not by name. Every feature is then z-scored with the mean and
  sample SD of the included cells' original values, those `.X` was made with.
- Derived features: `.X`'s derived features (kind `derived`) are recomputed from the formulas in
  `.uns["derived_features"]`, in order, with `DataFrame.eval` as in `features.add_derived_features`, on the
  corrected measurements in their own units and every other column of `fullest_df.csv` at the original values.
  A recomputed value that is infinite or empty keeps ComBat's value, and the step states the count per feature.
  A formula raising `UndefinedVariableError` or `KeyError` is passed over without a message when the formula's
  name is not a `.X` feature; when the name is a `.X` derived feature, every cell with a corrected value keeps
  ComBat's value and is counted. Any other error stops the step before any write. The reference is recomputed the
  same way, without a message.
- Warnings, the correction continuing: a condition whose share of one batch's cells is under half its share of
  another's (`uneven_conditions`); a before classifier score at most `NEAR_CHANCE` (0.05) above chance.
- Refusals, before any write: no two batches share a condition (`why_not_combat`); a subsample holding a cell
  (well, label) the cell table lacks, as made from an earlier cell table; a cell (well, label) or `.X` feature of
  `standard_adata.h5ad` that `fullest_df.csv` lacks, as from two Build cell table runs.
- Output: `layers["combat"]` (float64, empty for excluded cells and excluded conditions) and
  `.uns["batch_effects"]` in `standard_adata.h5ad`; the same layer, matched by well and label, in
  `standard_adata_sub.h5ad` when that file exists; both written through `io.written_whole`, uncompressed. In
  `.uns["batch_effects"]`, values keyed by a condition, batch or feature are saved as lists beside an array, since
  anndata refuses a key holding `/`; a None number is left out. `log_transformed` holds the names of the features
  `numpy.log1p` was applied to, in `log_columns` order, empty with `log_intensities` off or when `.var` has no such
  feature; the done message states "on log intensities" when `log_transformed` is not empty. Build cell table
  writes `standard_adata.h5ad` again without either and removes the layer from `standard_adata_sub.h5ad`; Batch
  effects then runs again.
