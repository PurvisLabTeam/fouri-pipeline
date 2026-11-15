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

`.github/workflows/shortcuts.yml` runs the install steps, `fouri-shortcuts` and the update steps on GitHub's
`windows-latest` and `macos-latest` runners, one job each, and opens the app from the Desktop shortcut. A push to any
branch that changes `src/fouri/app/shortcuts.py`, `src/fouri/app/__init__.py`, `pyproject.toml`, `uv.lock` or the
workflow file starts the workflow, as does Run workflow in the repository's Actions tab (`workflow_dispatch`, offered
once the workflow is on `main`); each job clones the pushed branch from GitHub. The workflow's runs are read in the Actions tab or with
`gh run list --workflow shortcuts.yml` and `gh run view <run id>`; each job's screenshots, of the Desktop before the app
opens and of the open app, are the run's artifact `screenshot-windows` or `screenshot-macos`, downloaded from the run's
page or with `gh run download <run id>`.

## Experiment page

- New experiment (`fouri.app.wells.new_config`) writes experiment.json in a results folder, which it creates
  when missing: every well with ND2 files, no metadata, `DEFAULT_CONDITION_COLUMNS`, `anchor_round` 0, the
  folder's name as `experiment_key` and the paths written as the page writes them (below). It refuses a
  folder that already has an experiment.json, an ND2 folder with no well's files, or an info_exp.csv that
  can't be read.
- The details card (`wells.save_details`) writes `experiment_key`, `experiment_name`, `cell_type`, `passage`,
  `date_completed` and `notes`, as the setup form does. The wells card (`wells.save_wells`) writes
  `well_list`, `well_metadata` (`metadata.table_to_config`) and `condition_columns`, and is refused while
  `check_metadata` finds an error. The standard three keep their Condition ticks whatever their values; an
  added column with no value in any well is saved in neither. With no ND2 files found, wells can be removed
  but not added, and the two ND2 checks are skipped. Import CSV replaces only the CSV's own columns for
  the wells it lists.
- The page never changes a sample ID already set, only one typed in its table or imported (a whole number
  above 0; a cleared one counts as none). On Save, `wells.fill_sample_ids` gives a well with none the ID of
  the first well in plate order with the same values in the condition columns, else one more than the
  largest ID, in plate order; it does not call `assign_sample_ids`, which numbers every well again.
- The app's Experiment page (`config_to_save` in `fouri/app/pages/experiment.py`, with `path_entry` in
  `fouri/app/wells.py`) writes the results folder
  to the key it is read from on that PC (`mac_base_dir` on non-Windows if set, else `base_dir`), left out
  only when the same folder would be read without it (the folder holding the config, with no other key to
  take over: on non-Windows `mac_base_dir` stays while `base_dir` is set), and the ND2 folder and info_exp.csv as
  `paths.data`/`paths.info_file`, each left out when its box is empty or it is base_dir's own `data` or
  `info_exp.csv`, relative to `base_dir` when inside it, in full otherwise. A path typed without a drive or
  root is read as relative to the results folder.
