# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Jupyter notebooks for the Purvis lab 4i image-analysis pipeline: stitched ND2 images from the Photon
microscope → per-cell feature tables → AnnData → PHATE plots. Lab members run the notebooks in order
(`00`–`08`), mostly on Windows with experiments on a `Z:` network drive.
Code shared between notebooks lives in the `fouri` package, `src/fouri/`.

Further guidance is read on demand: a subdirectory's `CLAUDE.md` when a file under that directory is read, a
file in `.claude/rules/` when a file matching its `paths` list is read.

- `src/fouri/app/CLAUDE.md`: lab install, update, `fouri-shortcuts`, the icons, the Experiment page
- `src/fouri/viewers/CLAUDE.md`: the napari editors
- `.claude/rules/skip-rules.md`: each step's skip and refuse rules, `fouri.status`
- `.claude/rules/filters-labels.md`: the saved filter and label rules, Build cell table's filters and labels,
  the threshold helpers
- `.claude/rules/embeddings.md`: subsampling, feature ranking, PHATE and UMAP runs, embedding keys, the
  `app.embedding` format
- `.claude/rules/adata.md`: what `build_adata` writes to `.obs`, `.X`, `.var` and `.uns`
- `.claude/rules/batch-effects.md`: Batch effects (`fouri.batch`): the batch metrics, the ComBat correction, the
  log transform, the recomputed derived features, the warnings and refusals, `layers["combat"]` and
  `.uns["batch_effects"]`

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

# run one step as its own program, as the app does (steps: convert segment align artifacts measure table batch embedding export finetune)
uv run python -m fouri.run measure --config ~/data/4i_local_test/experiment.json --wells B02 --settings '{"overwrite": true}'
# the same with the settings in a YAML or JSON file, and one setting replaced on the command line
uv run python -m fouri.run measure --config experiment.json --wells B02 --settings-file measure.yaml ring_width=8
```

`--settings` takes a step's settings as a JSON object and `--settings-file` as a YAML or JSON file; giving both is
refused. Each `key=value` argument in any position, a `--wells` value holding `=` included, replaces one setting,
with the value read as YAML (`8`, `true`, `[B02, B09]`), and the run record holds the merged settings. The Embedding
step given no `runs` makes the runs of the experiment's saved set-up (`app.embedding`), each setting given replacing
the set-up's.

Tests are pytest under `tests/`, on a made-up three-well experiment (`tests/builders.py` writes what each
step would have, `tests/conftest.py` the fixture); they cover only mistakes that would lose or miscount data
without anyone noticing. `slow` marks a test that imports torch, Cellpose, PHATE or UMAP, `gpu` one that
needs a CUDA GPU. ruff and mypy are configured in `pyproject.toml`: ruff uses foursight's rule
families, with docstrings optional but checked when present, and lets notebooks import after their
`CONFIG` cell; mypy checks `src/fouri` (not strict) and skips pystackreg, cellpose, ipywidgets and napari,
which ship no type information. `fouri` is installed editable, so changes under `src/fouri/` apply after a
kernel restart.

Not every notebook runs headless: `01` needs a CUDA GPU (it raises if Cellpose can't use one, unless its
`allow_cpu` setting is `True`); `03` is hand work in napari and
`02` ends with an optional check, both in the alignment viewer (`fouri.viewers.AlignmentViewer`);
`05` is not used for now (see the file table); `07` has two alternative threshold
cells (one cutoff or two) meant to be run one or the other, so a top-to-bottom run keeps the two-cutoff labels.

## experiment.json drives every notebook

Each notebook's first code cell is `CONFIG = r'...experiment.json'` followed by
`exp = load_config(CONFIG)`; every folder and the well list come from `exp` (`fouri.config.Experiment`).
The app's Experiment page starts an experiment and saves, each with its own Save, its paths and anchor round,
the details, the wells with their metadata and condition columns, and the derived features; each Save writes
the file through `save_config`, which keeps the previous version as `experiment.json.bak`, and writes only its
own keys, every other key as it was.
The page's rules are in `src/fouri/app/CLAUDE.md`.

- `base_dir` is `mac_base_dir` on non-Windows if set, else `base_dir`, else the folder holding the config.
- Folder names are `DEFAULT_PATHS` in `config.py`; each can be overridden in the config's `paths`
  (relative to `base_dir` or absolute). A new folder needs an entry there and a property on `Experiment`.
- The format stays readable as the older `project_config` files (`experiment_key`, `base_dir`,
  `mac_base_dir`, `well_list`, `well_metadata`); `Experiment.__getitem__`/`.get` exist for code written
  against that dict. New fields must be optional.
- Optional `anchor_round`: the round every other round is aligned onto, counted from 0 like `round_index`;
  default 0, the first round in `info_exp.csv`. The app's Experiment page sets it and always writes it, 0
  included.
- Optional `app`: what the app's pages save, through `save_config`. `app.filters` and `app.labels` hold the
  filter and label rules, in the format given in `.claude/rules/filters-labels.md`; `app.embedding` holds the
  Embedding page's set-up, in the format given in `.claude/rules/embeddings.md`; `app.batch_effects` holds the
  Batch effects page's set-up, in the format given in `.claude/rules/batch-effects.md`.
- Wells are always zero-padded (`B02`), normalized by `metadata.normalize_well`, sorted by `plate_sort_key`.
- Committed notebooks keep the placeholder `CONFIG = r'Z:\PATH\TO\EXPERIMENT\experiment.json'`; do not
  commit a local path.

## Files passed between notebooks

Notebooks talk only through files under `base_dir`, one set per well. File names are hard-coded on both
the writing and the reading side, so change them together.

| Step | Writes | Read by |
|------|--------|---------|
| 00 | `output_df/round_table_{well}.csv`: one row per round × channel listed in `info_exp.csv`, with its ND2 file, acquisition settings, `is_align_channel` and `raw_file_name`. `output_images/{well}/Round_{round_name}_well{well}_{signal}.tif` (the `raw_file_name`): every channel of every round, cropped to the smallest image, with an OME header (pixel size, wavelengths, exposure); 0.3–0.9 GB per well | table: 01, 02, 03, 06, Find artifacts; images: 01 (the `is_align_channel` ones), 06 (all), Find artifacts (the image size from each header) |
| 01 | `segmented/{well}/labels_{well}_round_{NNN}.tif` (Cellpose-SAM, or the kept fine-tuned model named in Segment nuclei's `model` setting). The segmentation editor saves over these, and the first time it saves a round keeps the file as Cellpose left it in `segmented/{well}/cellpose_original/` under the same name; a round with a file there is hand-edited | 02, 03, 06 (anchor round only) |
| 02 | `output_df/tmat_{well}.npy`: rigid transforms (rounds × 3 × 3) indexed by `round_index`, identity for the anchor; found on the outlines at half size by default (`downscale_factor` 2: within 0.27 px of full size on the local wells, 4× faster) | 03, 06 |
| 03 | replaces one entry of `tmat_{well}.npy` per saved round with a hand alignment: matching cells clicked in the alignment viewer (`fouri.viewers.AlignmentViewer`); creates the file if 02 never ran | 06 |
| 04 | removed: 06 moves the anchor's outlines into each round instead of measuring aligned copies of the images | — |
| 05 | not used for now: it draws masks on 04's aligned images and its masks never removed cells (07 dropped `nuc_mask`/`ring_mask`). Masks return with the artifact U-Net, and will remove the cells they cover | — |
| Find artifacts | `masks/{well}/masks_{well}_round_{NNN}.npz` (`fouri.artifacts.find_artifacts`; the step `artifacts` in `fouri.run`, Find artifacts on the Run page): `final_{signal}` (true or false per pixel) and `outlines_{signal}` (numbered objects, 0 for none) for every channel in the round's round table, in the shape of the round's images; `masks_{well}_round_{NNN}.json` beside the `.npz`: `edited` and, per signal, `spread` and `pad`. Until the artifact U-Net is retrained, the step is a stub: every mask empty, `spread` and `pad` 0, `edited` false; `fouri.artifacts.round_masks` is the one function the U-Net replaces. The mask editor (`fouri.viewers.artifacts.ArtifactViewer`, Review in napari) writes the same two files for each saved round, with `edited` true and, per signal, each artifact's own `spread` and `pad` under `artifacts` (`{"1": {"spread": 3, "pad": 5}}`); Find artifacts never replaces such a round | Measure cells (`final_{signal}`); the mask editor (both files) |
| 06 | `cell_data/cell_data_{well}_df.csv` (no row-number column; older ones have `Unnamed: 0`) | 07 |
| 07 | `fullest_df.csv`, `standard_adata.h5ad` in `base_dir`: every cell no artifact mask covers, with `included` (true or false: whether the filters named in Build cell table's `filters` setting kept it) and a column per label named in its `labels` setting; excluded cells stay in both. Batch effects (`fouri.batch.correct_batches`; the step `batch` in `fouri.run`, Batch effects on the Run page) adds `layers["combat"]` (the ComBat-corrected z-scores, empty where not corrected) and `.uns["batch_effects"]` to `standard_adata.h5ad`, and the same layer to `standard_adata_sub.h5ad` when that file exists; Build cell table writes `standard_adata.h5ad` again without either and removes the layer from `standard_adata_sub.h5ad` | 08, Batch effects |
| 08 | `standard_adata_sub.h5ad` in `base_dir`: the subsample with every measurement, each embedding in `.obsm` and the settings that made it in `.uns["embeddings"]`. Rerunning the subsample replaces the file and its embeddings | — |
| Export | `exports/<name>.h5ad` or `exports/<name>.csv` in `base_dir` (`fouri.export.export_cells`, not listed on the Run page: the app's Export page, `fouri/app/pages/export.py`, starts the step and lists the exports made, with Load settings, which refills its form from the run record, and Delete): the included cells, or with embeddings the subsampled cells still included, each with its metadata, label columns and the features chosen, as measurements, z-scores (as built or from `layers["combat"]`) or both; an existing file is replaced only with `overwrite`, which the page sends after asking | nothing in the pipeline |

`NNN` is the 0-based `round_index`, zero-padded to 3 digits. A step run through `python -m fouri.run` (the app)
also adds a "started" and an "ended" line to `run_record.jsonl` in `base_dir` (`io.run_record_path`). The app's
Run page starts its steps that way one after another (`fouri.app.runner.Run`), each step's settings file
(`<step>.settings.json`, passed to the step through `--settings-file`), progress file, stop file and printed output in
`fouri-app/<run id>/` under the system's temporary folder; the run id, also in the run record, is the run's start
time in ISO's compact form, `20261109T173005`, with `-2`, `-3` and so on added when a run's folder of that id already
exists.

File names live in `fouri/io.py` (`round_table_path`, `tmat_path`, `cell_data_path`, `label_paths`,
`label_path`, `original_label_path`, `mask_path`, `mask_paths`, `mask_record_path`, `fullest_df_path`, `adata_path`, `adata_sub_path`, `run_record_path`, `export_path`), with the functions that read and write those
files.

The round table is only read through `fouri.io.read_round_table`. Reading falls back to the `df_{well}.pkl`
that 00 wrote before the switch, renaming `nameRound` → `round_name`, `alignRound` → `round_index` and
`alignIm` → `is_align_channel`, so 02 and 03 still run on older experiments. 01, 06 and Find artifacts open 00's
TIFFs, so they need `raw_file_name`, which only the new table has: `io.require_raw_files` stops with "rerun 00" for an
older one.

The transforms file is only read and written through `fouri.io.load_transforms` and `save_transforms`
(the name lives in `tmat_path`). Reading falls back to the `tmat_{well}.pkl` that 02 and 03 wrote before
the switch to `.npy`, so older experiments still align without rerunning 02. A round nobody has aligned is
stored as a 3 × 3 matrix of NaN (the identity at the anchor): `align.save_manual_transform` creates the file
that way when a round is aligned by hand before 02 has run, and `measure_well` refuses a well with any NaN
round, naming the rounds.

00, 01, 02, Find artifacts and 06 only create outputs that are missing, unless their `overwrite` argument (a
setting in each notebook, default `False`) is `True`. Done means done for the rounds `info_exp.csv` lists now,
so rounds added at the end of `info_exp.csv` are processed without redoing the earlier ones. Each step's skip
and refuse rules, and `fouri.status`, which applies them for the Status page, are in
`.claude/rules/skip-rules.md`.

Nothing tracks what an output was made from, so after a fix in 03 or a new 01 run, 06 needs `overwrite=True`.
Every output a step writes goes through `io.written_whole`: it is written to its own name plus `.partial`
(`cell_data_B02_df.csv.partial`), renamed over the output when the write ends cleanly, and removed when it doesn't
(an error or Ctrl+C), so a stopped run never leaves a half-written file that would count as done and an earlier
file stays as it was. Readers match exact names or the `.tif` suffix, so a `.partial` left by a killed process is
never read. `save_transforms` hands `numpy.save` an open file, since it adds `.npy` to any other name.
Every step takes `wells=` (checked and put in plate order by `Experiment.pick_wells`); `None` means every well.

## Fine-tuning Cellpose

Fine-tune Cellpose (`fouri.finetune.finetune_cellpose(exp, n_epochs=100, wells=None)`; the step `finetune` in
`fouri.run`, absent from the Run page, the Status page and the Experiment page's plate, and passed wells as the
per-well steps are, through `status.WELL_STEPS`) trains a copy of Cellpose-SAM on the hand-edited rounds of the
chosen wells.
`training_squares` cuts, for each 4-connected region where a round's saved labels differ from its Cellpose original,
a 300 × 300 px square of the round's alignment-channel image and saved labels, centered on the region's bounding box
and moved inside the image at its edges; every square found is trained on, none held back, with Cellpose's
`train_seg` at Cellpose's default settings except `n_epochs`. The step refuses an `n_epochs` below 1, then a PC without a CUDA
GPU, before any square is read, and then a run with no square in any chosen well; a chosen well with no square is
named and passed over. Cellpose drops a square with fewer than 5 nuclei (`min_train_masks`): a logging handler on
the `cellpose.train` logger states the count and refuses the step when every square is dropped. The same handler
moves the bar from Cellpose's epoch lines (epoch 0, epoch 5 and every 10th), so Stop ends training at the next such
epoch; when no epoch line is recognized, the step states so, with the Cellpose version.

Each fine-tuned model is kept until deleted, in `models/cellpose/<name>` under the app-data directory
(`io.app_data_dir`: `%LOCALAPPDATA%\4i Pipeline` on Windows, `~/.local/share/4i Pipeline` elsewhere; NiceGUI's
storage is its `nicegui` directory, `fouri.app.storage_dir`), the name being the creation time in ISO's compact
form, `20261117T204206`. The directory is written as `<name>.partial`, renamed when training ends cleanly and
removed on any failure or Stop; a name already in use, kept or `.partial`, refuses the step before training. The
directory holds Cellpose's file `models/model` (1.2 GB) and `model.json`: `created` (ISO time), `experiment_key`,
`config` (the experiment.json path), `wells` (each well with the rounds its squares come from), `squares` (the count
found) and `n_epochs`. `fouri.models.kept_models` returns the readable kept models, newest first, and the names of
the directories whose `model.json` cannot be read (not JSON, a key missing, or a value of the wrong type);
`model_path(name)` returns a kept model's file and raises `FileNotFoundError` for a name not kept or not readable;
`shown_name` gives the name the app shows, `Fine-tuned 2026-11-17 20:42 [4i_local_test: B02]`.

Segment nuclei's `model` setting is `None` for Cellpose-SAM, the default, or a kept model's name: `segment_wells`
refuses a name `model_path` refuses before examining any well, and the run record keeps the name. The Run page's
Model row lists Cellpose-SAM (built in) and each kept model by its shown name, and names under the row the models
left out for an unreadable `model.json`.

The app's Models page (`fouri/app/pages/models.py`) starts fine-tuning as a run of one step, with the wells chosen
on the page's plate (`fouri.app.plate.wells_card`, the Run page's well chooser, every well chosen at first) and Epochs, a
whole number of 1 or more, then opens the Run page, which shows the step's bar, Stop and log as for any run.
Fine-tune is disabled while a run is going, while the GPU check has not answered and on a PC with no GPU. The page
counts the corrected regions saved (`fouri.app.finetuning.corrected_regions`: the squares `training_squares` finds
in every well, a well whose round table is missing or predates `raw_file_name` passed over) and names each well's
rounds that hold them. The page lists the kept models with their created time, experiment key and wells, and the
left-out line of the Run page's Model row. Delete asks, then removes the model's directory whole through
`fouri.app.finetuning.delete_model`, which raises `FileNotFoundError`, removing nothing, for a name holding a path
separator, a name ending in `.partial`, or a directory without `model.json`; Delete is disabled while a run is going.

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
  signal)`, e.g. `R0_pRB_nuc_mean`, `_ring_mean`, `_total_nuc_protein`, `_nuc_cyto_ratio` and `_covered`;
  plus the anchor's shape columns from skimage's `regionprops` (`axis_major_length`, … with `area` renamed
  `nuc_area` and `ring_area`). `metadata.channel_names` builds its suggestions with the same `feature_prefix`.
- Cells are outlined once, on the anchor round's segmentation (`exp.anchor_round`, so 02, 03 and 06
  cannot disagree). `fouri.measure.measure_well` moves those outlines and their rings into each other
  round with `align.move_to_round` (inverse of the round's transform, nearest pixel, cv2; `move_to_anchor`
  goes the other way), blanks pixels where every channel of the round is 0 (no image data in the montage),
  and measures 00's TIFFs. A cell is dropped if its nucleus or ring, in any round, falls outside the image
  or on such a blank area, or within `edge_margin` px of one (Measure cells' setting, 1 by default; 0 drops
  only cells that reach a blank area): each round's has-data area is moved into the anchor's frame, the areas
  are combined, and the result is eroded `edge_margin` times (3 × 3). `measure_wells` refuses a negative
  margin before measuring any well. `measure_well` returns the count with its path and `measure_wells` prints
  it per well. Cells with no ring are dropped by the inner joins.
  Per channel of each round, `measure_well` writes `{prefix}_covered`: true when the channel's `final_{signal}`
  mask, read by `fouri.artifacts.final_masks`, shares a pixel with the cell's nucleus or ring as moved into that
  round's frame. A round with no `.npz` is uncovered: every cover column of the round is false. A well is
  refused, naming the round, when a round's `.npz` lacks `final_{signal}` for a channel of the round table or
  holds a mask of another shape than the round's images; the other wells are measured.
- Cytoplasmic rings come from `fouri.rings.make_rings(labels, width, gap)` (06's `ring_width`/`ring_gap`
  settings, 6 and 1): a pixel joins a ring only if exactly one nucleus lies within `width`, so pixels two
  cells could claim are dropped. cv2 dilate/erode, identical to the original skimage version.

## Metadata and notebook 07

- `sample_ID` is filled by the app's Experiment page, only for the wells with none (`wells.fill_sample_ids`,
  in `src/fouri/app/CLAUDE.md`).
- `features.build_fullest_df` joins each well's CSV (via `io.cell_data_path`, dropping an old `Unnamed: 0`
  column if present) with its `well_metadata`; `treatment_time` is renamed
  `treatment time` in `fullest_df` and `adata.obs` (`_OBS_NAMES`), the name 07 has always used.
  `sample_ID` stays a category of numbers (08 selects samples with `isin([1, 2, …])`), other metadata text.
- `derived_features` are `pandas.eval` formulas applied in order, so one may use an earlier one; a formula
  naming a missing column is skipped with a printed message rather than raising, and any other error stops
  Build cell table. The app's Experiment page checks them the same way (`fouri.app.derived.check_formulas`) on
  `Cells.base`, the measured wells' cells as `build_fullest_df` joins them, and refuses to save a row that would
  stop Build cell table, a blank row, a name used twice, a measured column's name or a cell-page group's.
- Build cell table (`run.build_cell_table(exp, filters=(), labels=())`) removes, after the derived features,
  every cell covered in any channel (`fouri.artifacts.remove_covered`; an empty cover value, in cell data from
  before cover columns, counts as uncovered), states the count, 0 included, and drops every `_covered` column.
  The saved filters named in the `filters` setting are then applied to the cells left, so a filter may use a
  derived feature, and the result is added as the column `included`; the saved labels named in the `labels`
  setting are applied to every cell left, included or not, one column per label. How each rule is computed and
  what Build cell table refuses are in `.claude/rules/filters-labels.md`.
- 07's pRB thresholds are its own cells, unchanged until the app. `fouri.adata.build_adata(exp, fullest_df,
  labels=())` then builds the AnnData and saves `fullest_df.csv` and `standard_adata.h5ad`; what `.obs`, `.X`,
  `.var` and `.uns` hold, and which columns `.X` drops, are in `.claude/rules/adata.md`.

## Embeddings (08)

`fouri/embedding.py`, one call per step, all on `standard_adata_sub.h5ad` (`io.adata_sub_path`): `subsample`,
`rank_features`, `embed`, `make_runs` and `embed_grid`, with `plots.plot_embedding` for the figures. What each
does, the embedding keys (`run_key`) and the `app.embedding` format are in `.claude/rules/embeddings.md`.

## Dependencies

- `torch`/`torchvision` come from the PyTorch CUDA 13.0 index on Windows only (`[tool.uv.sources]`);
  other platforms take PyPI builds.
- `delve-fs` is pinned to a tag on the omar-merhebi fork until the upstream change is merged.
- OpenCV is `opencv-python-headless`, the build Cellpose also uses; the plain `opencv-python` bundles its
  own Qt, which can clash with napari's.
- OmegaConf is used only by `python -m fouri.run`, to read `--settings-file` and the `key=value` overrides;
  PyYAML, installed with OmegaConf, supplies the error caught for a file that is not YAML or JSON, and
  `types-PyYAML` in the dev group types PyYAML for mypy.

## Register of app text, messages and slides

- **UI strings (labels, tooltips, errors, logs):** Target audience is wet-lab scientists. State only what is
  shown, what was computed, or what a control does using standard scientific terms ("Segment nuclei",
  "Variance explained by batch", "Computed for N of M cells").
- **UI bans:**
  - Never explain internal software mechanics in UI text unless the user must take action on it.
  - No promotional/chatty phrases ("whenever you want", "never hold up a run", "Tip:").
  - Dropdowns name the option only ("Condition only", not "Only this condition"). "(default)" is the only
    allowed note.
- **Spelling:** American, in UI text and in code names (`color`, `colored`, `canceled`); names defined by
  Python or a library (`asyncio.CancelledError`) stay as defined.
- **Slides & presentations (if generated here):** Data presentation for wet-lab biologists. No "we", no
  progress timelines, no internal code/dataset nicknames (`Daugrid`, `H1`), no cute titles. Define every
  computational metric and dataset on first appearance.

## Writing for this repo

- During the refactor (the `fouri-refactor` branch) the step-by-step browser guide
  (`docs/pipeline_guide.html`, still on main) is removed and `README.md` is not updated per change: both
  are rewritten once at the end, before the merge into main.
- Logic used by more than one notebook goes in `fouri`, not copied between notebooks. Notebooks are
  moving toward settings, one `fouri` call per step, and displays (napari, plots); 00 to 03 and 06 have
  moved, into `fouri/preprocess.py`, `segment.py`, `align.py` and `measure.py`, 07's AnnData into
  `adata.py`, and 08 into `embedding.py`; figures go in `plots.py` (plotly). Torch, Cellpose, PHATE and UMAP
  are imported inside the functions that use them, so importing `fouri` stays fast; napari is imported at the
  top of `fouri/viewers/` only, which nothing else in `fouri` imports.
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
- `.gitignore` ignores every dotfile except `.gitignore`, `.gitattributes`, `.python-version` and
  `.claude/rules/` (the rest of `.claude/` stays ignored), so a new dot-folder (e.g. `.vscode/`) is not committed
  unless that rule changes.

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
