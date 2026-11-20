# fouri.app

Read with the root `CLAUDE.md` when a file under `src/fouri/app/` is read.

## Install, update and shortcuts

Lab members install the app by the README's steps; there is no installer and no setup script. On Windows, Git for
Windows must be installed first, by the user or IT; then, in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # then a new PowerShell window
git clone https://github.com/PurvisLabTeam/fouri-pipeline "$env:LOCALAPPDATA\fouri-pipeline"
cd "$env:LOCALAPPDATA\fouri-pipeline"
uv sync
uv run fouri-shortcuts
```

On Linux and macOS, git must be installed first (on macOS: `xcode-select --install`):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # then a new terminal
git clone https://github.com/PurvisLabTeam/fouri-pipeline ~/.local/share/fouri-pipeline
cd ~/.local/share/fouri-pipeline
uv sync
uv run fouri-shortcuts
```

To update: in the clone's directory (`$env:LOCALAPPDATA\fouri-pipeline` on Windows, `~/.local/share/fouri-pipeline`
on Linux and macOS), `git pull`, then `uv sync`. The app never checks for or offers updates.

`fouri-shortcuts` (`fouri.app.shortcuts`, a `[project.scripts]` console command) makes the 4i Pipeline shortcuts to
the launcher of the clone the command runs from (`.venv\Scripts\fouri-pipeline.exe` on Windows,
`.venv/bin/fouri-pipeline` on Linux and macOS), with the clone as working directory, replacing any already there:
`4i Pipeline.lnk` in the Start menu's Programs and on the Desktop on Windows, both directories found through
WScript.Shell (pywin32, a dependency on Windows only); `4i Pipeline.desktop` in `$XDG_DATA_HOME/applications` and in
the directory `xdg-user-dir DESKTOP` gives on Linux, the Desktop copy executable and marked trusted with `gio`;
`~/Applications/4i Pipeline.app` and a symbolic link to the bundle on the Desktop on macOS. The menu directory is
created when missing, a Desktop directory never; with no Desktop directory on Linux only the menu entry is written.
The command exits 1, writing no shortcut, on any other system or when the launcher is missing (`uv sync` writes it).

The lab's icon is in `src/fouri/app/icons/`, made once and committed: `fouri-pipeline.ico` (7 images, 16 to 256 px)
is the `.lnk`'s icon and the app's favicon in both modes on Windows (`ui.run(favicon=…)` in `fouri.app.main`), which
NiceGUI's native window sets on its title bar and taskbar button; `fouri-pipeline.png` (256 px) is the desktop
entry's `Icon=` on Linux and any other system, the app's favicon in both modes on every system but Windows and, in the
native window on Linux, pywebview's window icon (`app.native.start_args["icon"]`); `fouri-pipeline.icns` (made from
the `.ico` with Pillow) is copied into the bundle's `Contents/Resources/` and named in `CFBundleIconFile` on macOS,
and is the Dock icon of the app opened from the bundle; `fouri.app.main` sets no icon on macOS.
`fouri.app.shortcuts.icon_file` gives this system's file, found from `fouri/app/`'s own directory.

## Experiment page

- New experiment (`fouri.app.wells.new_config`) writes experiment.json in a results folder, which it creates when
  missing: every well with ND2 files, each with `sample_ID` 1 and no other metadata, `DEFAULT_CONDITION_COLUMNS`,
  `anchor_round` 0, the folder's name as `experiment_key` and the paths written as the page writes them (below). It
  refuses a folder that already has an experiment.json, an ND2 folder with no well's files, or an info_exp.csv that
  can't be read.
- The details card (`wells.save_details`) writes `experiment_key`, `experiment_name`, `cell_type`, `passage`,
  `date_completed` and `notes`. The wells card (`wells.save_wells`) writes `well_list`, `well_metadata`
  (`metadata.table_to_config`) and `condition_columns`, and is refused while `check_metadata` finds an error or the
  table lists a well twice (`B02 is listed twice.`). The standard three keep their Condition ticks whatever their
  values; an added column with no value in any well is saved in neither. With no ND2 files found, wells can be removed
  but not added, and the two ND2 checks are skipped. Import CSV replaces only the CSV's own columns for the wells it
  lists (`wells.imported_rows`), a blank `sample_ID` cell read as no ID, so Save fills it as for any well without one.
- The page keeps every sample ID already set, except one held by wells of two or more conditions (values in the
  condition columns): on Save the ID stays with the wells whose condition is unchanged since the last Save, or, when
  none or more than one of those conditions is unchanged, with the condition first in plate order, and is cleared on
  the others (`wells.split_wells` names them). The saved conditions are each well's condition values as experiment.json
  holds them, read again at Save (`wells.read_saved_conditions`); a well absent from them counts as changed. A sample ID
  may be typed in the table or imported (a whole number above 0; a cleared one counts as none). On Save,
  `wells.fill_sample_ids` gives each well with none, in plate order, the ID of the first well with the same condition,
  else one more than the largest ID in use; after the write a notification names the wells whose ID was split off
  (`Sample IDs changed: B09, C02, whose condition differs from the other wells with their ID.`).
- The derived-features section's Save refuses a name ending in `_covered`, a name `features.reserved_names` gives
  (measured columns, metadata columns as the cell table names them, `sample_ID`, `well`, `label`, `included`) whether
  or not a well is measured, and a column of the measured cells, as Build cell table does
  (`treatment: the cell table already has a column with this name.`).
- The app's Experiment page (`config_to_save` and `path_entry` in `fouri/app/wells.py`; `config_to_save` reads
  experiment.json again before the paths card's keys are written over it) writes the results folder
  to the key it is read from on that PC (`mac_base_dir` on non-Windows if set, else `base_dir`), left out
  only when the same folder would be read without it (the folder holding the config, with no other key to
  take over: on non-Windows `mac_base_dir` stays while `base_dir` is set), and the ND2 folder and info_exp.csv as
  `paths.data`/`paths.info_file`, each left out when its box is empty or it is base_dir's own `data` or
  `info_exp.csv`, relative to `base_dir` when inside it, in full otherwise. A path typed without a drive or
  root is read as relative to the results folder.
- Every Browse opens the app's own picker (`fouri.app.dialogs`), in the app's window and in a browser tab: a
  listing of directories and, for a file, of the files with the suffix sought. Enter on a typed or pasted
  directory holding the file sought (`pick_file`'s `name`: experiment.json beside experiment.json, info_exp.csv
  beside each info_exp.csv box, matched without regard to case, the exact spelling first) selects that file; any
  other directory, typed or clicked, is listed. Import CSV seeks no name.

## Embedding page

- The set-up tab's ranking k box takes a whole number, 4 or more (`fouri.embedding.MIN_RANKING_K`). A k at or above
  the tab's cell count (the subsample's cells of the samples chosen, or cells per sample times samples with no
  subsample) is refused for the ranking and for Save with the step's message, rechecked on every change of the samples.

## Run page

- Before a run whose Convert images step has changed wells among those chosen (`fouri.preprocess.changed_wells`),
  the Run page shows a question naming each well's changed rounds, its hand-edited rounds and the outputs to
  delete, `This can't be undone.` in bold red, and Cancel and Delete and reconvert; Cancel starts no step, and
  Delete and reconvert starts the run with Convert images' `reconvert_changed` true.

## Choices kept between pages

The Run, Export, Models and Batch effects pages keep the choices made on them, saved or not, in `STATE.kept`
(`fouri.app.state.kept_choices(page)`, keyed `run`, `export`, `models` and `batch`), in memory only and never
written to disk. `open_experiment` empties `STATE.kept` when the file opened is not the open experiment's (another
`config_path`: the same file under another spelling, a mapped drive against its network path or other capitals on
Windows, counts as another) and keeps the choices when the same file is opened again, as after a Save. A new start
of the app begins with none.

- Run: the wells chosen, the steps ticked, each step's Overwrite and shown settings, and the wells processed in
  parallel. On return a well no longer in the experiment is dropped, a new well is chosen, a setting no longer
  shown is dropped, a new setting is at its default, and a value no longer among a choice setting's options (a
  fine-tuned model deleted since) is at its default.
- Export: the settings Export would send, written on every change; on return the form is refilled through the
  path of Load settings (`_ExportPage.load`), which names the features and embeddings no longer present.
- Models: the wells chosen, reconciled as on the Run page, and Epochs as shown.
- Batch effects: Batch, Biology to keep and Log-transform intensities, written on every change; on return the page
  starts from the kept values before the saved set-up, a batch no longer offered falling back to `well` and a Biology to keep
  column no longer offered dropped.

## Saves of app keys

The Saves of the filters, the labels, the batch set-up and the embedding set-up write through
`fouri.app.state.save_app_value(exp, key, value)`, which reads experiment.json again, sets the one key under `app`
and saves through `save_config`; an `app` value that is not a mapping is refused with no write
(`experiment.json's app entry can't be read; nothing was saved.`).
