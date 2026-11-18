---
paths:
  - "src/fouri/{adata,run}.py"
  - "tests/test_run.py"
---

# The AnnData built by Build cell table

Loaded with the root `CLAUDE.md` when a file matching `paths` is read.

- `fouri.adata.build_adata(exp, fullest_df, labels=())` builds the AnnData: `.obs` = label, well,
  `included`, the label columns Build cell table hands it (kept as categories), metadata, centroid, bbox and
  orientation; `.X` = every other number column z-scored with the mean and sample SD of the included cells,
  applied to every cell. A column is dropped from `.X` and named when an included cell has no usable value
  (missing or infinite) or when its included cells all hold one value, judged by the values themselves, not by
  the SD coming out 0, which values not stored exactly in binary miss (the pre-refactor notebooks dropped such
  columns silently, judged over every cell: an earlier experiment lost all 22 `_nuc_cyto_ratio` features to a
  few cells with a ring mean of 0). An excluded cell may hold an empty value in `.X`; a table with no
  `included` column counts every cell as included. `.var` = kind (channel, shape,
  derived), round, signal, measure and acquisition from the round tables (a value differing between wells
  left empty and named); `.uns` = setup-form details, condition columns, derived formulas, plate, microscope,
  round tables. It saves `fullest_df.csv` (no row-number column) and `standard_adata.h5ad`, then removes
  `layers["combat"]` from `standard_adata_sub.h5ad` when that file holds one, through `io.written_whole`, every
  other part of the file unchanged.
