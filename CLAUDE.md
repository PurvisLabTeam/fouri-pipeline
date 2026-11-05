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
`allow_cpu` setting is `True`: about 20 minutes per image on the processor); `03` and `05`
are hand work in napari; `02` ends with an optional napari check; `07` has two alternative threshold
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
| 00 | `output_df/round_table_{well}.csv`: one row per round × channel listed in `info_exp.csv`, with its ND2 file, acquisition settings, `is_align_channel` and `raw_file_name`. `output_images/{well}/Round_{round_name}_well{well}_{signal}.tif` (the `raw_file_name`): every channel of every round, cropped to the smallest image, with an OME header (pixel size, wavelengths, exposure); 0.3–0.9 GB per well | table: 01–06; images: 01 (the `is_align_channel` ones) |
| 01 | `segmented/{well}/labels_{well}_round_{NNN}.tif` (Cellpose-SAM) | 02, 03, 06 |
| 02 | `output_df/tmat_{well}.npy`: rigid transforms (rounds × 3 × 3) indexed by `round_index`, identity for the anchor | 03, 04 |
| 03 | replaces one entry of `tmat_{well}.npy` with a hand alignment: matching cells clicked in napari (`fouri.align.start_hand_alignment`) | 04 |
| 04 | `aligned_tiffs/{well}/Round_{nameRound}_well{well}_{signal}.tif`; rewrites `df_{well}.pkl` adding `aligned_file_name` | 05, 06 |
| 05 | `masks/mask_{well}.pkl`: napari shape polygons to exclude (optional) | 06 |
| 06 | `cell_data/cell_data_{well}_df.{pkl,csv}` | 07 |
| 07 | `fullest_df.csv`, `standard_adata.h5ad`, `standard_adata_sub.h5ad` in `base_dir` | 08 |
| 08 | `standard_adata_sub_filtered.h5ad`, PowerPoint and PNG plots in `base_dir` | — |

`NNN` is the 0-based `round_index`, zero-padded to 3 digits.

The round table is only read through `fouri.preprocess.read_round_table` (the name lives in
`round_table_path`). Reading falls back to the `df_{well}.pkl` that 00 wrote before the switch, renaming
`nameRound` → `round_name`, `alignRound` → `round_index` and `alignIm` → `is_align_channel`, so 02–06 still
run on older experiments. 01 needs `raw_file_name`, which only the new table has: `alignment_paths` stops
with "rerun 00" for an older one.

The transforms file is only read and written through `fouri.align.load_transforms` and `save_transforms`
(the name lives in `tmat_path`). Reading falls back to the `tmat_{well}.pkl` that 02 and 03 wrote before
the switch to `.npy`, so older experiments still align without rerunning 02.

## Inputs, naming and couplings

- `info_exp.csv`: one row per round; the first column is the round name (`R0`, `R1`, …), read as
  `round_name` whatever its header (`myRound` in existing files); every other column is an ND2 optical
  configuration name and its cell is the signal (antibody) in that round, empty if unused. Every channel
  in an ND2 file needs a column. 00 finds the round folder in `data/` by the substring
  `Round_{round_name}_` and the file by `Well{well}_` and the `.nd2` suffix.
- 00's `align_signal` setting (default `"DNA"`) marks each round's alignment channel as
  `is_align_channel`; `round_table` refuses a well unless every round has exactly one. 01 segments those
  channels, listed in round order by `fouri.preprocess.alignment_paths`.
- Feature columns from 06 are `{nameRound}_{signal}_{suffix}`, e.g. `R0_pRB_nuc_mean`, `_ring_mean`,
  `_total_nuc_protein`, `_nuc_cyto_ratio`, plus `nuc_area`, `ring_area`. `metadata.channel_names` rebuilds
  the same prefixes from `info_exp.csv` to suggest derived features; keep the two in step.
- Cells are outlined once, on the anchor round's segmentation, and measured in every aligned channel.
  The anchor round is `exp.anchor_round`, so 02, 03 and 06 cannot disagree: `fouri.align.find_transforms`
  aligns every round onto it, 03 aligns the hand-fixed round onto it, and 06 outlines cells on it
  (`label_round`).
- Cytoplasmic rings in 06 come from `fouri.rings.make_rings(labels, width=6, gap=1)`, which drops pixels
  that could belong to two cells.

## Metadata and notebook 07

- `sample_ID` is filled by `metadata.assign_sample_ids` unless every well already has one: wells with the
  same values in `condition_columns` share an ID, numbered in plate order.
- `features.build_fullest_df` joins each well's CSV with its `well_metadata`; `treatment_time` is renamed
  `treatment time` in `fullest_df` and `adata.obs` (`_OBS_NAMES`), the name 07 has always used.
- `derived_features` are `pandas.eval` formulas applied in order, so one may use an earlier one; a formula
  naming a missing column is skipped with a printed message rather than raising.
- 07 puts metadata in `.obs`, z-scores the numeric features into `.X`, drops any feature column with NaN,
  and subsamples with sketchKH grouped by `sample_ID`.

## Dependencies

- `torch`/`torchvision` come from the PyTorch CUDA 13.0 index on Windows only (`[tool.uv.sources]`);
  other platforms take PyPI builds.
- `delve-fs` is pinned to a tag on the omar-merhebi fork until the upstream change is merged.

## Writing for this repo

- The readers are lab members, not programmers. `README.md` and `docs/pipeline_guide.html` (a
  step-by-step guide opened in a browser) describe the same workflow; a change to setup or to how a
  notebook is run needs both updated. Messages printed by `fouri` say in plain words what to do next.
- Logic used by more than one notebook goes in `fouri`, not copied between notebooks. Notebooks are
  moving towards settings, one `fouri` call per step, and displays (napari, plots); 00 to 03 have
  moved, into `fouri/preprocess.py`, `segment.py` and `align.py`. Napari and its plugins are imported
  inside the functions that use them, so importing `fouri` stays fast.
- 03 no longer uses the Affinder plugin: Affinder 0.5.0 calls `viewer._canvas_size`, which napari 0.9
  removed, so its Start button fails.
- Work run side by side: `fouri.worker_count()` picks how many at once on the current PC (memory at
  8 GB per process, and cores) and prints why. Worker processes start with `spawn` on every platform,
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
