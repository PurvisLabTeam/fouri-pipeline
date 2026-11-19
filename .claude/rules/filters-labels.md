---
paths:
  - "src/fouri/{filters,labels,features,plots,run}.py"
  - "src/fouri/app/{rules,gates,gate_figures}.py"
  - "src/fouri/app/pages/{filters,labels}.py"
  - "tests/test_{filters,labels,run}.py"
---

# Filters and labels

Loaded with the root `CLAUDE.md` when a file matching `paths` is read.

## Saved rules in experiment.json

`app.filters` is a list of filter rules, each with `name`, `use` (the page's switch), `rule` and `feature` (a
column name such as `R0_DNA_total_nuc_protein`): `middle_percentiles` takes `low`/`high` as percentiles from 0
to 100, `robust_sd` takes `sds`, `between` takes `low`/`high` in the feature's own values, and `per_sample`
(optional, false when left out) judges each `sample_ID` on its own cells. `fouri.filters.saved_filters` reads
them, an empty list when either key is missing, and refuses two rules with one name. `app.labels` is a list of
label rules, each with `column` (the label's name, and the column it becomes), `use` and `rule`: a `threshold`
takes `feature`, one or two `cutoffs` and `classes`, one name more than the cut-offs (`set_on_samples` and `sds`
are the page's record of how the cut-offs were chosen; Build cell table ignores them); a `gates` rule takes `x`,
`y` and `gates`, each gate a `class`, a `shape` and `points`, [x, y] pairs in the features' own values: two
opposite corners for a `rectangle`, the outline (three or more) for a `freehand` (`x_log10` and `y_log10` are
the page's record of whether each axis was drawn on log10, a value on such an axis saved as 10 to the value
drawn; Build cell table ignores them). `fouri.labels.saved_labels` reads them the same way and refuses two
labels with one column.

## Build cell table

- Build cell table (`run.build_cell_table(exp, filters=(), labels=())`) applies the saved filters named in its
  `filters` setting after the derived features and the removal of the cells an artifact mask covers, so a
  filter may use a derived feature, and adds the result as the column `included`.
  `fouri.filters.keep` works out one rule on every cell left in the table by itself: the percentiles (numpy's
  linear ones) and `robust_sd`'s median and median absolute deviation (times 1.4826) come from the cells with a
  value, per `sample_ID` when `per_sample` is set (it changes nothing for `between`), both bounds are kept, and
  an empty or infinite value is never kept. `apply_filters` checks every rule before working any out, then
  returns `included` (true where every rule keeps the cell) and each rule's count of the cells it leaves out on
  its own, whatever the others do. Build cell table refuses, before writing anything, a name no saved filter
  has, a feature the table lacks, or a rule that can't be applied (unknown kind, missing field, bad bounds,
  `per_sample` without a `sample_ID` column or while a measured well has no sample ID, the message naming neither
  the wells nor the filter); it says each rule's count and the total kept. With no names every cell is included and
  nothing is said about filters. `fouri.app.rules.kept_cells` returns the per-sample refusal as for any rule it
  cannot apply; the Filters page shows the refusal on that rule, and Explore states that the filter can't be
  applied.
- Then Build cell table applies the saved labels named in its `labels` setting (`fouri.labels`) to every cell,
  included or not, each as a column named as the label's `column`: a category with the classes in saved order,
  every one listed even with no cell in it, and no value for a cell in no class. `label_cells` splits a
  threshold as `features.threshold_labels` does: with one cut-off, below it the first class and at or above it
  the second; with two, below the lower the first, from the lower to the upper both kept the second, above the
  upper the third. Gates are tried in saved order and a cell takes the first that holds it: a rectangle holds
  the cells between its corners on both axes, both bounds kept, a freehand gate those inside its outline as
  matplotlib's `Path` finds them; gates with one class name form one class. An empty or infinite value gives
  no class. `apply_labels` checks every rule before working any out; Build cell table refuses, before writing
  anything, a name no saved label has, a feature the table lacks, a label named as a column the table already
  has, or a rule that can't be applied (unknown kind, missing field, no cut-off or more than two, classes not
  one more than the cut-offs, no gate, an unknown shape, a rectangle without two corners, a freehand gate with
  fewer than three points, a point that isn't two finite numbers); it says each label's cells per class and in
  no class. Given no names, `named_filters` and `named_labels` return no rules without reading the saved ones,
  so a mistake in saved rules stops only a run that names some. The run record's settings name the filters and
  labels applied.

## Threshold helpers for the app

- For the app: `features.suggest_cutoffs` (Otsu, two-cut Otsu, the valley of the smoothed curve, on cells
  within `sds` SD of the mean, 3 by default; `middle_values`, `curve_range` and `plot_threshold_histogram`
  take the same `sds`), `features.threshold_labels(values, cutoffs, classes)` (the classes are the label's
  names; `fouri.labels` uses it) and `plots.plot_threshold_histogram` (plotly; the cut-off lines are its only
  shapes, so a page can make every shape draggable, and the Otsu–valley band is a trace).
