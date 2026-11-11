# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Jupyter notebooks for the Purvis lab 4i image-analysis pipeline: stitched ND2 images from the Photon
microscope → per-cell feature tables → AnnData → PHATE plots. Lab members run the notebooks in order
(`00`–`08`, after `setup_experiment.ipynb`), mostly on Windows with experiments on a `Z:` network drive.
Code shared between notebooks lives in the `fouri` package, `src/fouri/`.

## Commands

```bash
uv sync                        # create/update .venv (Python 3.12, locked by uv.lock)
uv run nbstripout --install    # once per clone: git strips notebook outputs on commit
git config filter.nbstripout.extrakeys metadata.language_info.version  # once per clone: and the Python version a notebook records
uv run jupyter lab             # or VS Code with the .venv kernel
uv run ruff check              # lint src/fouri and the notebooks
uv run ruff format             # format them
uv run mypy                    # type-check src/fouri
uv run pytest -m 'not slow and not gpu'  # the quick tests; -m slow for the ones importing torch or UMAP

# run one notebook top to bottom without touching the file in the repo
uv run jupyter nbconvert --to notebook --execute --output-dir /tmp/nbrun 06_calculate_cell_properties.ipynb

# run one step as its own program, as the app does (steps: convert segment align measure table embedding)
uv run python -m fouri.run measure --config ~/data/4i_local_test/experiment.json --wells B02 --settings '{"overwrite": true}'
```

Tests are pytest under `tests/`, on a made-up three-well experiment (`tests/builders.py` writes what each
step would have, `tests/conftest.py` the fixture); they cover only mistakes that would lose or miscount data
without anyone noticing. `slow` marks a test that imports torch, Cellpose, PHATE or UMAP, `gpu` one that
needs a CUDA GPU. ruff and mypy are configured in `pyproject.toml`: ruff uses foursight's rule
families, with docstrings optional but checked when present, and lets notebooks import after their
`CONFIG` cell; mypy checks `src/fouri` (not strict) and skips pystackreg, cellpose, ipywidgets and napari,
which ship no type information. `fouri` is installed editable, so changes under `src/fouri/` apply after a
kernel restart.

Not every notebook runs headless: `01` needs a CUDA GPU (it raises if Cellpose can't use one, unless its
`allow_cpu` setting is `True`: about 20 minutes per image on the processor); `03` is hand work in napari and
`02` ends with an optional check, both in the alignment viewer (`fouri.viewers.AlignmentViewer`);
`05` is not used for now (see the file table); `07` has two alternative threshold
cells (one cutoff or two) meant to be run one or the other, so a top-to-bottom run keeps the two-cutoff labels.

## experiment.json drives every notebook

Each notebook's first code cell is `CONFIG = r'...experiment.json'` followed by
`exp = load_config(CONFIG)`; every folder and the well list come from `exp` (`fouri.config.Experiment`).
`setup_experiment.ipynb` calls `fouri.setup_form` (an ipywidgets plate form), and the app's Experiment page
saves its paths and anchor round; both write the file through `save_config`, which keeps the previous version
as `experiment.json.bak`.

- `base_dir` is `mac_base_dir` on non-Windows if set, else `base_dir`, else the folder holding the config.
- Folder names are `DEFAULT_PATHS` in `config.py`; each can be overridden in the config's `paths`
  (relative to `base_dir` or absolute). A new folder needs an entry there and a property on `Experiment`.
- The app's Experiment page (`config_to_save` in `fouri/app/pages/experiment.py`) writes the results folder
  to the key it is read from on that PC (`mac_base_dir` on non-Windows if set, else `base_dir`), left out
  only when the same folder would be read without it (the folder holding the config, with no other key to
  take over: on non-Windows `mac_base_dir` stays while `base_dir` is set), and the ND2 folder and info_exp.csv as
  `paths.data`/`paths.info_file`, each left out when its box is empty or it is base_dir's own `data` or
  `info_exp.csv`, relative to `base_dir` when inside it, in full otherwise. A path typed without a drive or
  root is read as relative to the results folder.
- The format stays readable as the older `project_config` files (`experiment_key`, `base_dir`,
  `mac_base_dir`, `well_list`, `well_metadata`); `Experiment.__getitem__`/`.get` exist for code written
  against that dict. New fields must be optional.
- Optional `anchor_round`: the round every other round is aligned onto, counted from 0 like `round_index`;
  default 0, the first round in `info_exp.csv`. The app's Experiment page sets it and always writes it, 0
  included. It is not in the setup form, which keeps any key it does not manage when it saves.
- Optional `app`: what the app's pages save, through `save_config`. `app.filters` is a list of filter rules,
  each with `name`, `use` (the page's switch), `rule` and `feature` (a column name such as
  `R0_DNA_total_nuc_protein`): `middle_percentiles` takes `low`/`high` as percentiles from 0 to 100,
  `robust_sd` takes `sds`, `between` takes `low`/`high` in the feature's own values, and `per_sample`
  (optional, false when left out) judges each `sample_ID` on its own cells. `fouri.filters.saved_filters`
  reads them, an empty list when either key is missing, and refuses two rules with one name.
- Wells are always zero-padded (`B02`), normalized by `metadata.normalize_well`, sorted by `plate_sort_key`.
- Committed notebooks keep the placeholder `CONFIG = r'Z:\PATH\TO\EXPERIMENT\experiment.json'`; do not
  commit a local path.

## Files passed between notebooks

Notebooks talk only through files under `base_dir`, one set per well. File names are hard-coded on both
the writing and the reading side, so change them together.

| Step | Writes | Read by |
|------|--------|---------|
| 00 | `output_df/round_table_{well}.csv`: one row per round × channel listed in `info_exp.csv`, with its ND2 file, acquisition settings, `is_align_channel` and `raw_file_name`. `output_images/{well}/Round_{round_name}_well{well}_{signal}.tif` (the `raw_file_name`): every channel of every round, cropped to the smallest image, with an OME header (pixel size, wavelengths, exposure); 0.3–0.9 GB per well | table: 01, 02, 03, 06; images: 01 (the `is_align_channel` ones), 06 (all) |
| 01 | `segmented/{well}/labels_{well}_round_{NNN}.tif` (Cellpose-SAM) | 02, 03, 06 (anchor round only) |
| 02 | `output_df/tmat_{well}.npy`: rigid transforms (rounds × 3 × 3) indexed by `round_index`, identity for the anchor; found on the outlines at half size by default (`downscale_factor` 2: within 0.27 px of full size on the local wells, 4× faster) | 03, 06 |
| 03 | replaces one entry of `tmat_{well}.npy` per saved round with a hand alignment: matching cells clicked in the alignment viewer (`fouri.viewers.AlignmentViewer`); creates the file if 02 never ran | 06 |
| 04 | removed: 06 moves the anchor's outlines into each round instead of measuring aligned copies of the images | — |
| 05 | not used for now: it draws masks on 04's aligned images and its masks never removed cells (07 dropped `nuc_mask`/`ring_mask`). Masks return with the artifact U-Net, and will remove the cells they cover | — |
| 06 | `cell_data/cell_data_{well}_df.csv` (no row-number column; older ones have `Unnamed: 0`) | 07 |
| 07 | `fullest_df.csv`, `standard_adata.h5ad` in `base_dir`: every cell, with `included` (true or false: whether the filters named in Build cell table's `filters` setting kept it); excluded cells stay in both | 08 |
| 08 | `standard_adata_sub.h5ad` in `base_dir`: the subsample with every measurement, each embedding in `.obsm` and the settings that made it in `.uns["embeddings"]`. Rerunning the subsample replaces the file and its embeddings | — |

`NNN` is the 0-based `round_index`, zero-padded to 3 digits. A step run through `python -m fouri.run` (the app)
also adds a "started" and an "ended" line to `run_record.jsonl` in `base_dir` (`io.run_record_path`). The app's
Run page starts its steps that way one after another (`fouri.app.runner.Run`), each step's progress file, stop file
and printed output in `fouri-app/<run id>/` under the system's temporary folder; the run id, also in the run record,
is the run's start time in ISO's compact form, `20261109T173005`, with `-2`, `-3` and so on added when a run's folder
of that id already exists.

File names live in `fouri/io.py` (`round_table_path`, `tmat_path`, `cell_data_path`, `label_paths`,
`label_path`, `fullest_df_path`, `adata_path`, `adata_sub_path`, `run_record_path`), with the functions that read and write those
files.

The round table is only read through `fouri.io.read_round_table`. Reading falls back to the `df_{well}.pkl`
that 00 wrote before the switch, renaming `nameRound` → `round_name`, `alignRound` → `round_index` and
`alignIm` → `is_align_channel`, so 02 and 03 still run on older experiments. 01 and 06 open 00's TIFFs, so
they need `raw_file_name`, which only the new table has: `io.require_raw_files` stops with "rerun 00" for an
older one.

The transforms file is only read and written through `fouri.io.load_transforms` and `save_transforms`
(the name lives in `tmat_path`). Reading falls back to the `tmat_{well}.pkl` that 02 and 03 wrote before
the switch to `.npy`, so older experiments still align without rerunning 02. A round nobody has aligned is
stored as a 3 × 3 matrix of NaN (the identity at the anchor): `align.save_manual_transform` creates the file
that way when a round is aligned by hand before 02 has run, and `measure_well` refuses a well with any NaN
round, naming the rounds.

00, 01, 02 and 06 only create outputs that are missing, unless their `overwrite` argument (a setting in each
notebook, default `False`) is `True`. Done means done for the rounds `info_exp.csv` lists now, so rounds added
at the end of `info_exp.csv` are processed without redoing the earlier ones:

- 00 skips a well whose `round_table_{well}.csv` (not the old `.pkl`) lists the same rounds. If the saved table
  lists only the first of them, `convert_well` writes just the new rounds' TIFFs and saves the full table. If a
  new round's images are smaller, the crop shrinks: every round is written again and the well's labels in
  `segmented` are cut to the new size. The crop only trims the bottom and right edges, so outlines and saved
  transforms keep their positions (within 0.04 px on B02). 00 refuses a well whose rounds were inserted,
  removed, renamed or reordered, since later steps store their results by round number.
- 01 skips a round whose labels exist.
- 02 skips a round whose matrix isn't NaN, so rounds fixed in 03 are kept; a saved file with fewer rounds keeps
  them and aligns only the new ones. It refuses a well whose label images don't match its rounds, that has no
  label image for `anchor_round`, or whose saved file has more rounds than the well or a non-identity matrix at
  `anchor_round` (the anchor changed); the checks run in that order and a well gets the first one's message.
- A refused well is a failed well: it is left out of the work and passed to `check_results` with the others, so
  the other wells still run and the step ends failed naming it.
- 06 skips a well whose CSV has a `{prefix}_nuc_mean` column for every channel in its round table; a well
  missing some measures all its rounds again.

Nothing tracks what an output was made from, so after a fix in 03 or a new 01 run, 06 needs `overwrite=True`.
The checks run in the main process before work is handed out; `report.say_skipped` prints what was skipped.
Every output a step writes goes through `io.written_whole`: it is written to its own name plus `.partial`
(`cell_data_B02_df.csv.partial`), renamed over the output when the write ends cleanly, and removed when it doesn't
(an error or Ctrl+C), so a stopped run never leaves a half-written file that would count as done and an earlier
file stays as it was. Readers match exact names or the `.tif` suffix, so a `.partial` left by a killed process is
never read. `save_transforms` hands `numpy.save` an open file, since it adds `.npy` to any other name.
`fouri.status` (`step_status` per well, `experiment_status` for 07 and 08) applies the same
rules, read from disk, for the app's Status page; change it with a step's skip rule. Every step takes
`wells=` (checked and put in plate order by `Experiment.pick_wells`); `None` means every well.

## Inputs, naming and couplings

- `info_exp.csv`: one row per round; the first column is the round name (`R0`, `R1`, …), read as
  `round_name` whatever its header (`myRound` in existing files); every other column is an ND2 optical
  configuration name and its cell is the signal (antibody) in that round, empty if unused. Every channel
  in an ND2 file needs a column. 00 finds the round folder in `data/` by the substring
  `Round_{round_name}_` and the file by `Well{well}_` and the `.nd2` suffix.
- 00's `align_signal` setting (default `"DNA"`) marks each round's alignment channel as
  `is_align_channel`; `round_table` refuses a well unless every round has exactly one. 01 segments those
  channels, listed in round order by `fouri.io.alignment_paths`. `round_table` also refuses a round that
  names the same signal twice, since both channels would be written to the same TIFF.
- Feature columns from 06 are `{prefix}_{suffix}` with the prefix from `metadata.feature_prefix(round_name,
  signal)`, e.g. `R0_pRB_nuc_mean`, `_ring_mean`, `_total_nuc_protein`, `_nuc_cyto_ratio`; plus the anchor's
  shape columns from skimage's `regionprops` (`axis_major_length`, … with `area` renamed `nuc_area` and
  `ring_area`). `metadata.channel_names` builds its suggestions with the same `feature_prefix`.
- Cells are outlined once, on the anchor round's segmentation (`exp.anchor_round`, so 02, 03 and 06
  cannot disagree). `fouri.measure.measure_well` moves those outlines and their rings into each other
  round with `align.move_to_round` (inverse of the round's transform, nearest pixel, cv2; `move_to_anchor`
  goes the other way), blanks pixels where every channel of the round is 0 (no image data in the montage),
  and measures 00's TIFFs. A cell is dropped if its nucleus or ring, in any round, falls outside the image
  or on such a blank area, or within 1 px of one: each round's has-data area is moved into the anchor's
  frame, the areas are combined, and the result is eroded once (3 × 3). `measure_well` returns the count
  with its path and `measure_wells` prints it per well. Cells with no ring are dropped by the inner joins.
- Cytoplasmic rings come from `fouri.rings.make_rings(labels, width, gap)` (06's `ring_width`/`ring_gap`
  settings, 6 and 1): a pixel joins a ring only if exactly one nucleus lies within `width`, so pixels two
  cells could claim are dropped. cv2 dilate/erode, identical to the original skimage version.

## Metadata and notebook 07

- `sample_ID` is filled by `metadata.assign_sample_ids` unless every well already has one: wells with the
  same values in `condition_columns` share an ID, numbered in plate order.
- `features.build_fullest_df` joins each well's CSV (via `io.cell_data_path`, dropping an old `Unnamed: 0`
  column if present) with its `well_metadata`; `treatment_time` is renamed
  `treatment time` in `fullest_df` and `adata.obs` (`_OBS_NAMES`), the name 07 has always used.
  `sample_ID` stays a category of numbers (08 selects samples with `isin([1, 2, …])`), other metadata text.
- `derived_features` are `pandas.eval` formulas applied in order, so one may use an earlier one; a formula
  naming a missing column is skipped with a printed message rather than raising.
- Build cell table (`run.build_cell_table(exp, filters=())`) applies the saved filters named in its `filters`
  setting after the derived features, so a filter may use one, and adds the result as the column `included`.
  `fouri.filters.keep` works out one rule on every cell of the table by itself: the percentiles (numpy's
  linear ones) and `robust_sd`'s median and median absolute deviation (times 1.4826) come from the cells with a
  value, per `sample_ID` when `per_sample` is set (it changes nothing for `between`), both bounds are kept, and
  an empty or infinite value is never kept. `apply_filters` checks every rule before working any out, then
  returns `included` (true where every rule keeps the cell) and each rule's count of the cells it leaves out on
  its own, whatever the others do. Build cell table refuses, before writing anything, a name no saved filter
  has, a feature the table lacks, or a rule that can't be applied (unknown kind, missing field, bad bounds,
  `per_sample` without `sample_ID`); it says each rule's count and the total kept. With no names every cell is
  included and nothing is said about filters. The run record's settings name the filters applied.
- 07's pRB thresholds are its own cells, unchanged until the app. `fouri.adata.build_adata(exp, fullest_df)`
  then builds the AnnData: `.obs` = label, well, `included`, metadata, centroid, bbox, orientation and
  `pRB_label` if a threshold ran; `.X` = every other number column z-scored with the mean and sample SD of the
  included cells, applied to every cell. A column is dropped from `.X` and named when an included cell has no
  usable value (missing or infinite) or when its included cells all hold one value, judged by the values
  themselves, not by the SD coming out 0, which values not stored exactly in binary miss (old 07 dropped such
  columns silently, judged over every cell: an earlier experiment lost all 22 `_nuc_cyto_ratio` features to a
  few cells with a ring mean of 0). An excluded cell may hold an empty value in `.X`; a table with no
  `included` column counts every cell as included. `.var` = kind (channel, shape,
  derived), round, signal, measure and acquisition from the round tables (a value differing between wells
  left empty and named); `.uns` = setup-form details, condition columns, derived formulas, plate, microscope,
  round tables. It saves `fullest_df.csv` (no row-number column) and `standard_adata.h5ad`.
- For the app: `features.suggest_cutoffs` (Otsu, two-cut Otsu, the valley of the smoothed curve, on cells
  within 3 SD of the mean), `features.threshold_labels` and `plots.plot_threshold_histogram` (plotly; the
  cut-off lines are its only shapes, so a page can make every shape draggable, and the Otsu–valley band is a
  trace). 07 doesn't use them yet.

## Embeddings (08)

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
  `X_{method}_{dims}d_top{n}_{setting}{value}…` (settings sorted). The same run with the same picked
  features, samples, ranking k and seed replaces its embedding (the result is identical); other inputs get
  `_2`, `_3`. `.uns["embeddings"][key]` records method, dims, top_n, settings, picked, the samples actually
  used, ranking_k, seed, and the features used with their scores.
- `plots.plot_embedding(adata, key, color_by)`: plotly, 2D (WebGL) or 3D; a measurement or a number column
  on a Viridis scale from the 1st to the 99th percentile, a text or category column one trace and color per
  group (`GROUP_COLORS`, the app's).

## Dependencies

- `torch`/`torchvision` come from the PyTorch CUDA 13.0 index on Windows only (`[tool.uv.sources]`);
  other platforms take PyPI builds.
- `delve-fs` is pinned to a tag on the omar-merhebi fork until the upstream change is merged.
- OpenCV is `opencv-python-headless`, the build Cellpose also uses; the plain `opencv-python` bundles its
  own Qt, which can clash with napari's.

## Writing for this repo

- The readers are lab members, not programmers. Messages printed by `fouri` say in plain words what to
  do next. During the refactor (the `fouri-refactor` branch) the step-by-step browser guide
  (`docs/pipeline_guide.html`, still on main) is removed and `README.md` is not updated per change: both
  are rewritten once at the end, before the merge into main.
- Logic used by more than one notebook goes in `fouri`, not copied between notebooks. Notebooks are
  moving toward settings, one `fouri` call per step, and displays (napari, plots); 00 to 03 and 06 have
  moved, into `fouri/preprocess.py`, `segment.py`, `align.py` and `measure.py`, 07's AnnData into
  `adata.py`, and 08 into `embedding.py`; figures go in `plots.py` (plotly). Torch, Cellpose, PHATE and UMAP
  are imported inside the functions that use them, so importing `fouri` stays fast; napari is imported at the
  top of `fouri/viewers/` only, which nothing else in `fouri` imports.
- Hand-work viewers live in `fouri/viewers/`. `base.BaseViewer` (abstract) opens napari with a docked panel:
  well and round dropdowns, a ▲▼◀▶ pad, the viewer's own tools, Undo and Save, and a shortcut list; it binds
  the keys in `KeyboardShortcuts` (Up/Down round, Left/Right well, Shift for first/last, Ctrl+Z, Shift+S;
  napari 0.9 already uses Ctrl+S and most bare letters) and shows prompts in the canvas text overlay.
  Subclasses fill hooks (`_build_layers`, `_tool_widgets`, `_round_choices`, `_load_well`, `_load_round`,
  `save`, `undo`, `_extra_keys`). `alignment.AlignmentViewer` is 02's optional check and 03: it starts each
  round from its saved transform or from none, clicks snap to cell centers, Fit (Shift+F) needs 3 pairs with
  two clicks a third of the image apart. A failed well or round switch leaves the old one on screen and puts
  the reason in the prompt; a round the viewer doesn't offer (the anchor, or one the well lacks) opens its first
  round instead, with the reason in the prompt. The panel is docked as its Qt widget (`panel.native`): napari
  resets a docked magicgui panel's dropdowns whenever a layer is added, removed or renamed, which would empty
  the round list. 03 no longer uses Affinder, whose 0.5.0 calls `viewer._canvas_size`, removed in napari 0.9.
- `python -m fouri.viewers alignment --config <experiment.json> --well B02 [--round R2]` opens one viewer as
  its own program (the names are `EDITORS` in `viewers/__main__.py`) and waits until its window closes, as the
  app's Review page will start it: exit code 0 when the window is closed, 2 for an unknown editor or a well not
  in the experiment, 1 when the viewer can't be opened (no experiment.json, a well whose files won't load).
- Work run side by side: `fouri.parallel.parallel_map(function, items, …)` runs a top-level function (or a
  `functools.partial` of one) over items in spawned processes with a progress bar. It returns, in item order,
  each item's result, a `Failed` (its name and the error) when its job raised, or a `Dropped` when a stop left
  it unstarted: the pool gets only as many items as it has workers, and the next as each one finishes, so on
  `report.Stopped` from the bar nothing more is handed out and running ones finish; a worker killed outright
  fails every unfinished item, including those not yet handed out. Each caller saves what finished, then calls
  `check_results`, which says each failure and raises `report.Stopped` if anything was dropped, else
  `report.StepFailed` naming the failed items (wells, or embedding keys), which `fouri.run` records. Its
  message gives each item's reasons in brackets, the first sentence of each distinct error, because the Run
  page shows only that line in red.
  00, 02, 06 and `embed_grid` use it; `segment_wells`, which has no pool, carries on past a failed well the
  same way. `fouri.worker_count()` picks how many at once on the current PC (memory at
  8 GB per process, and cores, never more than the items) and prints why; an empty list returns `[]`
  without starting any. Worker processes start with `spawn` on every platform,
  so a run on Linux behaves like the Windows lab PCs; a worker is handed file paths, not arrays, and
  prints nothing (on Windows its prints never reach the notebook). Progress bars use `tqdm.auto`, and
  nothing prints while one is running.
- Messages and bars go through `fouri.report` (`say`, `say_skipped`, `progress`), never `print` or tqdm directly:
  after `report.report_to(path)` (the app, when it starts a step) they are written to that file as JSON lines
  (message; bar open, move, close with its parent bar; `report.finished` at the end) instead of printed.
- `.gitignore` ignores every dotfile except `.gitignore`, `.gitattributes` and `.python-version`, so a new
  dot-folder (e.g. `.claude/`) is not committed unless that rule changes.

## Checking a change

On the maintainer's workstation, `~/data/4i_refactor_test/experiment.json` is a one-well (B02) experiment whose
`paths` point at the raw data and pre-refactor segmentation in `~/data/4i_baseline/`.
`~/data/4i_baseline/old/` holds the pre-refactor outputs to compare against, e.g.
`old/cell_data/cell_data_B02_df.csv` against `~/data/4i_refactor_test/cell_data/cell_data_B02_df.csv`.

`~/data/4i_local_test/experiment.json` is the 5-well experiment (B02, B09, C02, E02, E09) run through the
new 00–06; its outputs are safe to overwrite. Its segmentation is Cellpose-SAM, while `old/` is Cellpose
2.0.5's `cyto` model run on the processor, so cell numbers don't match the old tables. To compare 06 with
`old/cell_data/cell_data_B02_df.csv`, use a scratch experiment whose `segmented` is `old/segmented`, whose
`output_df` holds the new `round_table_B02.csv` plus `old/output_df/tmat_B02.pkl`, and whose `output_images`
is the local test's. Expected (checked 2026-11-06): 5,324 of the old 5,388 cells (the edge rule drops 83, 19
of which the old table already lacked); the same columns except `nuc_mask`/`ring_mask` (gone) and
`major_axis_length`/`minor_axis_length` (now `axis_major_length`/`axis_minor_length`); shape columns and
every R0 value identical; rounds 1–5 differ by a median 0.3 % (nuclei) and 0.4 % (rings), worst about 16 %
and 26 %.
