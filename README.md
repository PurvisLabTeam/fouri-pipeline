# 4i_pipeline
Notebooks for running the Purvis lab 4i pipeline, from stitched ND2 images to per-cell data, AnnData and PHATE.

A step-by-step guide for lab members is in [`docs/pipeline_guide.html`](docs/pipeline_guide.html) (open it in a browser).

# Setup (once per computer)
1. Install [uv](https://docs.astral.sh/uv/) and Git.
2. Clone this repository, then in the repository folder run:
   ```
   uv sync
   uv run nbstripout --install
   git config filter.nbstripout.extrakeys metadata.language_info.version
   ```
   `uv sync` creates the `.venv` environment with Python 3.12 and every package the notebooks need
   (Cellpose-SAM with the CUDA build of PyTorch on Windows, napari, scanpy, PHATE, delve, ...).
   `nbstripout --install` makes git drop notebook outputs when you commit. The `git config` line also
   drops the Python version a notebook records each time it runs, so running a notebook never shows it
   as changed. Set up before that line was added? Run it once in your copy of the repository.
3. Open the notebooks in VS Code (or `uv run jupyter lab`) and select the `.venv` kernel.

To pick up changes later: `git pull`, then `uv sync`.

# Running an experiment
1. Open `setup_experiment.ipynb` and load the experiment folder. The form reads the wells from `data/` and the
   rounds from `info_exp.csv`; choose wells and conditions on the plate (or import a metadata CSV) and save.
   This writes `experiment.json` into the experiment folder.
2. In each pipeline notebook, set the `CONFIG` line at the top to that file, e.g.
   ```python
   CONFIG = r'Z:\Clara\1.3_20260310_Working Folder 4i\experiment.json'
   ```
   Every folder and the well list come from the config; nothing else needs editing.
3. Run the notebooks in order (03 and 05 only when needed).

## Experiment folder layout
```
<experiment folder>/
    experiment.json      written by setup_experiment.ipynb
    info_exp.csv         optical configuration -> channel names, one row per round
    data/                one subfolder per round (Round_<name>_...) with Well<W>_Channel...nd2 files
    output_df/ output_images/ segmented/ aligned_tiffs/ masks/ cell_data/    created by the notebooks
    fullest_df.csv  standard_adata.h5ad  standard_adata_sub.h5ad             created by 07
```
If the raw data or the info file live somewhere else, set *data folder* / *info file* in the form
(stored as `paths` in `experiment.json`, relative to the experiment folder or as a full path).

Every round is aligned onto the first round in `info_exp.csv`, and cells are outlined on it. To use
another round, add `"anchor_round": 1` to `experiment.json` by hand (counting from 0, so 1 is the second
round). Notebooks 02, 03 and 06 all read it, and the form keeps it when it saves.

## Metadata CSV
One row per well; only `well` is required (`B2` and `B02` both work):
```
well,treatment,dose,treatment_time,sample_ID
B02,control,0,96h,
B03,abema_fulv,100nM,96h,
```
- `sample_ID` is optional: wells with the same treatment, dose and treatment_time get the same ID, numbered in
  plate order. Extra columns (e.g. `replicate`, `cell_line`) are kept and end up in `adata.obs`; tick them in the
  form if they define a condition.
- *Export CSV template* in the form writes a CSV listing the wells with data, ready to fill in with Excel.

# Expected outputs
The pipeline takes images from the Purvis lab Photon microscope system and outputs .csv and .pkl files for each well.
These contain feature data for each round of imaging and are used for downstream analysis.

# Notebook Descriptions
`setup_experiment` creates or edits the experiment's `experiment.json` (wells, conditions, derived features).

Notebook 00 converts the ND2 files into one TIFF per channel in `output_images`, all cut to the same size and keeping the
microscope's pixel size, wavelengths and exposure, and saves a table per well listing every round and channel. Its one
setting, `align_signal` (default `DNA`), names the channel used to align the rounds. It needs 0.3 to 0.9 GB of disk per well.

Notebook 01 performs the segmentation with Cellpose-SAM on the GPU. It stops with an error if no GPU is available, unless
`allow_cpu` is set to `True` (about 20 minutes per image instead of 1). Experiments run through an older version of 00 need 00
run again before 01.

Notebook 02 uses the segmentation masks to determine the transformation required to align all channels of the images

Notebook 03 is only used if the automatic transformations are incorrect, usually as a result of underlying issues with the data

Notebook 04 applies the transformation matrixes generated in notebook 02 or 03 to all channels of the images, generating a set of aligned tiffs

Notebook 05 uses a Napari interface to mask out debris and other artifacts ensuring the quality of the data

Notebook 06 calculates the properties of individual cells and outputs a .csv and a .pkl file for each well of the 96 well plate imaged.

Notebook 07 combines the wells into one table with each well's metadata, adds the derived features from the config
(e.g. pRB/RB ratio), optionally labels cells pRB high/low, z-scores the features and writes `standard_adata.h5ad`
plus a kernel-herding subsample `standard_adata_sub.h5ad`.

Notebook 08 ranks features with delve's Laplacian score, computes a 3D PHATE embedding and exports plots to PowerPoint.

The shared code (config loading, the setup form, file helpers, cytoplasmic rings) is the `fouri` package in `src/fouri/`.
