---
paths:
  - "src/fouri/{adata,run}.py"
  - "tests/test_run.py"
---

# The AnnData built by Build cell table

Loaded with the root `CLAUDE.md` when a file matching `paths` is read.

- `fouri.adata.build_adata(exp, fullest_df, labels=())` builds the AnnData: `.obs` = label, well,
  `included`, the label columns Build cell table hands it (kept as categories), metadata (every metadata column of
  the experiment, empty for a cell whose well has no value in it; no derived feature takes one of these names,
  `features.reserved_names`), centroid, bbox and orientation; `.X` = every other number column z-scored with the mean and sample SD of the included cells,
  applied to every cell. A column is dropped from `.X` and named when an included cell has no usable value
  (missing or infinite) or when its included cells all hold one value, judged by the values themselves, not by
  the SD coming out 0, which values not stored exactly in binary miss (the pre-refactor notebooks dropped such
  columns silently, judged over every cell: an earlier experiment lost all 22 `_nuc_cyto_ratio` features to a
  few cells with a ring mean of 0). An excluded cell may hold an empty value in `.X`; a table with no
  `included` column counts every cell as included. `.var` = kind (channel, shape, derived), round, signal, measure and
  acquisition from the round tables of the wells the table holds (a value differing between wells left empty and
  named); `.uns` = the experiment's details (the Experiment page's details card), condition columns, derived formulas,
  plate (every well of `well_list`), microscope and round tables (both from the wells the table holds), and
  `cell_table`, the build's id (`{"id": <uuid4 hex>}`). It writes `fullest_df.csv` (no row-number column) and
  `standard_adata.h5ad` each under its `.partial` name, then replaces both together (`_replace_together`: the old CSV
  set aside, the new CSV and then the new h5ad renamed in through `Path.replace`, the set-aside copy removed); a
  failed write or rename of either leaves both as they were, with no file left beside them, and raises an `OSError` naming
  the file; it then removes `layers["combat"]` from `standard_adata_sub.h5ad` when that file
  holds one, through `io.written_whole`, every other part of the file unchanged.
