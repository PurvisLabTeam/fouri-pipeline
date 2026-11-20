---
paths:
  - "src/fouri/batch.py"
  - "src/fouri/app/batches.py"
  - "src/fouri/app/batch_figures.py"
  - "src/fouri/app/pages/batch.py"
---

# Batch effects

Loaded with the root `CLAUDE.md` when a file matching `paths` is read.

`fouri.batch.correct_batches(exp, batch, keep=None, log_intensities=True)` reads `standard_adata.h5ad` and works
on the included cells (every cell when the table has no `included` column).

- Batches and conditions: `batch_labels` gives each cell's batch, `well`, `plate row` or `plate column` (from
  `well`), or any `.obs` column. `condition_labels` joins the `keep` columns' values with ` · `; `keep=None` means
  `default_keep(exp)`, the condition columns with a value in at least one well, and an empty `keep` puts every cell in one condition,
  `all cells`. A batch or keep column `.obs` lacks is refused.
- Blank values: a cell whose well has no value in the batch column or in a kept condition column is left out of the
  metrics and the correction, with no corrected value; the step's messages and the page name those wells with their
  count of cells, and `.uns["batch_effects"]["blank_wells"]` records each well with its count.
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
  another's (`uneven_conditions`), a used condition absent from a batch counting as a share of 0 (`condition_shares`
  still gives it empty, and the warning, in the step and on the page, reads it as 0%); a before classifier score at
  most `NEAR_CHANCE` (0.05) above chance.
- Refusals, before any write: no two batches share a condition (`why_not_combat`); a `standard_adata.h5ad` without a
  cell table id, when a subsample exists (`fouri.adata.check_subsample`'s first check); a cell (well, label) or `.X`
  feature of `standard_adata.h5ad` that `fullest_df.csv` lacks, as from two Build cell table runs.
- A subsample cut from another cell table, or holding no id, is left as it was: the cell table is corrected, no layer
  is written into the subsample, and after the write the step says `standard_adata_sub.h5ad was made from an earlier
  cell table, so its corrected values were not written. Run Embedding with Overwrite to make a new subsample.` Export
  still refuses such a subsample.
- Output: `layers["combat"]` (float64, empty for excluded cells and excluded conditions) and
  `.uns["batch_effects"]` in `standard_adata.h5ad`; the same layer, matched by well and label, in
  `standard_adata_sub.h5ad` when that file exists and was cut from the same cell table; both written through
  `io.written_whole`, uncompressed. In
  `.uns["batch_effects"]`, values keyed by a condition, batch or feature are saved as lists beside an array, since
  anndata refuses a key holding `/`; a None number is left out. `log_transformed` holds the names of the features
  `numpy.log1p` was applied to, in `log_columns` order, empty with `log_intensities` off or when `.var` has no such
  feature; the done message states "on log intensities" when `log_transformed` is not empty. Build cell table
  writes `standard_adata.h5ad` again without either and removes the layer from `standard_adata_sub.h5ad`; Batch
  effects then runs again.

## Set-up, step and page

- `app.batch_effects` in experiment.json: `batch` (text: `well`, `plate row`, `plate column` or an `.obs` metadata
  column), `keep` (a list of `.obs` metadata column names) and `log_intensities` (true or false). The Batch effects
  page saves the set-up through `fouri.app.batches.save_batch_setup`, every other key unchanged;
  `saved_batch_setup` reads the set-up, None when missing, and raises `ValueError` for a value that is not a
  mapping, lacks a key or holds a key of the wrong type.
- The step: `Step.BATCH`, `batch` in `python -m fouri.run`, Batch effects on the Run page (card 7, between Build
  cell table and Embedding). The card has no settings rows: the card states the saved set-up in one line
  (`batch_setup_line`), cannot be ticked without a readable set-up, and sends `batch`, `keep` and `log_intensities`,
  read from experiment.json when Run is clicked, as the step's settings. `fouri.status.experiment_status` gives
  Batch effects as complete, with the note `by {batch}` from `.uns["batch_effects"]["batch"]`, when
  `standard_adata.h5ad` holds `layers["combat"]`, and as not started otherwise.
- Choices: Batch offers `well`, `plate row`, `plate column`, then the metadata columns that are not condition
  columns, under their `.obs` names, except `sample_ID` (`batch_choices`); Biology to keep offers the condition
  columns among the metadata columns, under their `.obs` names, except the chosen batch (`keep_choices`). With no saved set-up the page starts at `well`, `default_keep(exp)` and
  Log-transform intensities on; a saved batch no longer offered starts at `well`, and a saved keep column no longer
  offered is dropped, the page's set-up then differing from the saved one.
- Live numbers (`live_numbers`, run off the page's thread; a result for an earlier choice is discarded): the cells
  of `fouri.app.cells.load_cells` kept by `fouri.app.rules.kept_cells`, and the columns of `Cells.features`
  z-scored with the mean and sample SD of those cells, a column with an empty or infinite value in a kept cell or a
  single value among them excluded, as `build_adata` makes `.X`; then `batch_labels`, `condition_labels`,
  `conditions_in_batches`, `why_not_combat`, `condition_shares`, `uneven_conditions`, `batch_groups` and
  `batch_metrics`. With the filters unchanged since Build cell table, and `well_list` in plate order, the live
  numbers equal the step's saved `before` numbers. With `why_not_combat` set the page shows only the refusal.
- Saved result (`saved_correction`: `.uns["batch_effects"]` read with h5py and `read_elem` without `.X`, every
  text array as a list of `str`): the After card, the after bars, the after and shuffled-reference classifier
  bars, the violins and the adjustment heatmap are shown only when the saved `batch` and `keep` equal the page's
  set-up and `log_transformed` is not empty exactly when Log-transform intensities is on (`matches`); otherwise
  the After card names the saved correction's set-up, or, with no record, states that the step has not run.
- Figures (plotly, features named by their `Cells.features` titles): variance explained by batch for the live
  `top` 15, before and after; balanced accuracy per condition, before, after and ComBat on shuffled batches, with
  each condition's chance; violins of the five features with the largest saved `before` share, per batch, before
  (`.X`) and after (`layers["combat"]`), for one saved condition, at most 2,000 cells per batch (`default_rng(0)`);
  a heatmap of the mean adjustment per batch for the 15 features with the largest mean absolute adjustment over
  batches, on a diverging scale centered at 0.
