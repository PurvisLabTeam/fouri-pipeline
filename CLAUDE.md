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

# run one notebook top to bottom without touching the file in the repo
uv run jupyter nbconvert --to notebook --execute --output-dir /tmp/nbrun 06_calculate_cell_properties.ipynb
```

There is no test suite yet. ruff and mypy are configured in `pyproject.toml`: ruff uses foursight's rule
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
`setup_experiment.ipynb` calls `fouri.setup_form` (an ipywidgets plate form), which writes the file
through `save_config` and keeps the previous version as `experiment.json.bak`.

- `base_dir` is `mac_base_dir` on non-Windows if set, else `base_dir`, else the folder holding the config.
- Folder names are `DEFAULT_PATHS` in `config.py`; each can be overridden in the config's `paths`
  (relative to `base_dir` or absolute). A new folder needs an entry there and a property on `Experiment`.
- The format stays readable as Lauryn's older `project_config` files (`experiment_key`, `base_dir`,
  `mac_base_dir`, `well_list`, `well_metadata`); `Experiment.__getitem__`/`.get` exist for code written
  against that dict. New fields must be optional.
- Optional `anchor_round`: the round every other round is aligned onto, counted from 0 like `round_index`;
  default 0, the first round in `info_exp.csv`. It is not in the setup form, which keeps any key it does
  not manage when it saves.
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
| 02 | `output_df/tmat_{well}.npy`: rigid transforms (rounds × 3 × 3) indexed by `round_index`, identity for the anchor | 03, 06 |
| 03 | replaces one entry of `tmat_{well}.npy` per saved round with a hand alignment: matching cells clicked in the alignment viewer (`fouri.viewers.AlignmentViewer`); creates the file if 02 never ran | 06 |
| 04 | removed: 06 moves the anchor's outlines into each round instead of measuring aligned copies of the images | — |
| 05 | not used for now: it draws masks on 04's aligned images and its masks never removed cells (07 dropped `nuc_mask`/`ring_mask`). Masks return with the artifact U-Net, and will remove the cells they cover | — |
| 06 | `cell_data/cell_data_{well}_df.csv` (no row-number column; older ones have `Unnamed: 0`) | 07 |
| 07 | `fullest_df.csv`, `standard_adata.h5ad`, `standard_adata_sub.h5ad` in `base_dir` | 08 |
| 08 | `standard_adata_sub_filtered.h5ad`, PowerPoint and PNG plots in `base_dir` | — |

`NNN` is the 0-based `round_index`, zero-padded to 3 digits.

File names live in `fouri/io.py` (`round_table_path`, `tmat_path`, `cell_data_path`, `label_paths`), with the
functions that read and write those files.

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
notebook, default `False`) is `True`: 00 skips a well whose `round_table_{well}.csv` exists (not the old
`.pkl`), 01 a round whose labels exist, 02 a round whose matrix isn't NaN (rounds fixed in 03 are kept), 06 a
well whose CSV exists. Nothing tracks what an output was made from, so after a fix in 03 or a new 01 run, 06
needs `overwrite=True`. 02 stops if a saved file has a different number of rounds, or a non-identity matrix at
`anchor_round` (the anchor changed). The check runs in the main process before work is handed out;
`report.say_skipped` prints what was skipped. `io.write_tiff` writes to a `.partial` name and renames, so a
stopped run never leaves a half-written TIFF that would count as done.

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
  round with `align.move_labels` (inverse of the round's transform, nearest pixel, cv2), blanks pixels
  where every channel of the round is 0 (no image data in the montage), and measures 00's TIFFs. Cells
  missing from any round's table (moved outside the image, or no ring) are dropped by the inner joins.
- Cytoplasmic rings come from `fouri.rings.make_rings(labels, width, gap)` (06's `ring_width`/`ring_gap`
  settings, 6 and 1): a pixel joins a ring only if exactly one nucleus lies within `width`, so pixels two
  cells could claim are dropped. cv2 dilate/erode, identical to the original skimage version.

## Metadata and notebook 07

- `sample_ID` is filled by `metadata.assign_sample_ids` unless every well already has one: wells with the
  same values in `condition_columns` share an ID, numbered in plate order.
- `features.build_fullest_df` joins each well's CSV (via `io.cell_data_path`, dropping an old `Unnamed: 0`
  column if present) with its `well_metadata`; `treatment_time` is renamed
  `treatment time` in `fullest_df` and `adata.obs` (`_OBS_NAMES`), the name 07 has always used.
- `derived_features` are `pandas.eval` formulas applied in order, so one may use an earlier one; a formula
  naming a missing column is skipped with a printed message rather than raising.
- 07 puts metadata in `.obs`, z-scores the numeric features into `.X`, drops any feature column with NaN,
  and subsamples with sketchKH grouped by `sample_ID`.

## Dependencies

- `torch`/`torchvision` come from the PyTorch CUDA 13.0 index on Windows only (`[tool.uv.sources]`);
  other platforms take PyPI builds.
- `delve-fs` is pinned to a tag on the omar-merhebi fork until the upstream change is merged.
- OpenCV is `opencv-python-headless`, the build Cellpose also uses; the plain `opencv-python` bundles its
  own Qt, which can clash with napari's.

## Writing for this repo

- The readers are lab members, not programmers. `README.md` and `docs/pipeline_guide.html` (a
  step-by-step guide opened in a browser) describe the same workflow; a change to setup or to how a
  notebook is run needs both updated. Messages printed by `fouri` say in plain words what to do next.
- Logic used by more than one notebook goes in `fouri`, not copied between notebooks. Notebooks are
  moving towards settings, one `fouri` call per step, and displays (napari, plots); 00 to 03 and 06 have
  moved, into `fouri/preprocess.py`, `segment.py`, `align.py` and `measure.py`. Torch and Cellpose are
  imported inside the functions that use them, so importing `fouri` stays fast; napari is imported at the
  top of `fouri/viewers/` only, which nothing else in `fouri` imports.
- Hand-work viewers live in `fouri/viewers/`. `base.BaseViewer` (abstract) opens napari with a docked panel:
  well and round dropdowns, a ▲▼◀▶ pad, the viewer's own tools, Undo and Save, and a shortcut list; it binds
  the keys in `KeyboardShortcuts` (Up/Down round, Left/Right well, Shift for first/last, Ctrl+Z, Shift+S;
  napari 0.9 already uses Ctrl+S and most bare letters) and shows prompts in the canvas text overlay.
  Subclasses fill hooks (`_build_layers`, `_tool_widgets`, `_round_choices`, `_load_well`, `_load_round`,
  `save`, `undo`, `_extra_keys`). `alignment.AlignmentViewer` is 02's optional check and 03: it starts each
  round from its saved transform or from none, clicks snap to cell centres, Fit (Shift+F) needs 3 pairs with
  two clicks a third of the image apart. A failed well or round switch leaves the old one on screen and puts
  the reason in the prompt. 03 no longer uses Affinder, whose 0.5.0 calls `viewer._canvas_size`, removed in
  napari 0.9.
- Work run side by side: `fouri.parallel.parallel_map(function, items, …)` runs a top-level function (or a
  `functools.partial` of one) over items in spawned processes with a progress bar, results in order; 00, 02
  and 06 use it. `fouri.worker_count()` picks how many at once on the current PC (memory at
  8 GB per process, and cores, never more than the items) and prints why; an empty list returns `[]`
  without starting any. Worker processes start with `spawn` on every platform,
  so a run on Linux behaves like the Windows lab PCs; a worker is handed file paths, not arrays, and
  prints nothing (on Windows its prints never reach the notebook). Progress bars use `tqdm.auto`, and
  nothing prints while one is running.
- `.gitignore` ignores every dotfile except `.gitignore`, `.gitattributes` and `.python-version`, so a new
  dot-folder (e.g. `.claude/`) is not committed unless that rule changes.

## Checking a change

On Omar's workstation, `~/data/4i_refactor_test/experiment.json` is a one-well (B02) experiment whose
`paths` point at the raw data and pre-refactor segmentation in `~/data/4i_baseline/`.
`~/data/4i_baseline/old/` holds the pre-refactor outputs to compare against, e.g.
`old/cell_data/cell_data_B02_df.csv` against `~/data/4i_refactor_test/cell_data/cell_data_B02_df.csv`.

`~/data/4i_local_test/experiment.json` is the 5-well experiment (B02, B09, C02, E02, E09) run through the
new 00–06; its outputs are safe to overwrite. Its segmentation is a newer Cellpose run than `old/`, so cell
numbers don't match the old tables. To compare 06 with `old/cell_data/cell_data_B02_df.csv`, use a scratch
experiment whose `segmented` is `old/segmented`, whose `output_df` holds the new `round_table_B02.csv` plus
`old/output_df/tmat_B02.pkl`, and whose `output_images` is the local test's. Expected (checked 2026-11-05):
same 5,388 cells and columns, shape and R0 nucleus means identical, R0 ring means identical except rings on
no-data pixels; rounds 1–5 away from the edge differ by a median 0.3 % (nuclei) and 0.4 % (rings).
