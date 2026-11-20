"""Loading, saving, and directory resolution for `experiment.json` files.

`load_config` reads the file into an `Experiment`. `Experiment` resolves the base directory and
the data directories. `save_config` writes the file atomically, with the previous version kept as
`experiment.json.bak`. Legacy `project_config` files load without conversion.
"""

import json
import os
import platform
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .metadata import normalize_well, plate_sort_key, to_sample_id
from .report import say

CONFIG_NAME = "experiment.json"

# Folder layout inside an experiment's base_dir. Any of these can be
# overridden in the config's "paths" section.
DEFAULT_PATHS = {
    "info_file": "info_exp.csv",
    "data": "data",
    "output_df": "output_df",
    "output_images": "output_images",
    "segmented": "segmented",
    "masks": "masks",
    "cell_data": "cell_data",
    "exports": "exports",
}

DEFAULT_CONDITION_COLUMNS = ["treatment", "dose", "treatment_time"]


class Experiment:
    """Experiment configuration loaded from `experiment.json`, with data directories resolved.

    The base directory is `mac_base_dir` on platforms other than Windows, if set; else `base_dir`,
    if set; else the directory holding `experiment.json`. Each data directory is the name in
    `DEFAULT_PATHS`, or the override in the config's `paths`, relative to the base directory or
    absolute.

    Parameters
    ----------
    config : dict[str, Any]
        The parsed contents of `experiment.json`.
    config_path : str | Path
        The path of the `experiment.json` file.

    Attributes
    ----------
    config : dict[str, Any]
        The parsed contents of `experiment.json`.
    config_path : str
        The absolute path of the `experiment.json` file.
    base_dir : str
        The base directory.
    paths : dict[str, str]
        The path of each data directory, keyed by the names in `DEFAULT_PATHS`.
    """

    config: dict[str, Any]
    config_path: str
    base_dir: str
    paths: dict[str, str]

    def __init__(self, config: dict[str, Any], config_path: str | Path) -> None:
        self.config = config
        self.config_path = os.path.abspath(config_path)
        self.base_dir = _resolve_base_dir(config, self.config_path)
        overrides = config.get("paths") or {}
        self.paths = {
            name: os.path.normpath(os.path.join(self.base_dir, overrides.get(name) or default))
            for name, default in DEFAULT_PATHS.items()
        }

    # -- identity -------------------------------------------------------------
    @property
    def key(self) -> str:
        """The experiment key, `experiment_key` in the config. Empty if not set."""
        return self.config.get("experiment_key", "")

    @property
    def name(self) -> str:
        """The experiment name, `experiment_name` in the config. Empty if not set."""
        return self.config.get("experiment_name", "")

    # -- wells and metadata ---------------------------------------------------
    @property
    def well_list(self) -> list[str]:
        """The wells listed in `well_list`, in file order. Empty if not set."""
        return list(self.config.get("well_list", []))

    @property
    def well_metadata(self) -> dict[str, dict[str, Any]]:
        """Metadata of each well in `well_list`, an integer `sample_ID` read as an int."""
        meta = self.config.get("well_metadata") or {}
        out = {}
        for well in self.well_list:
            m = dict(meta.get(well, {}))
            if "sample_ID" in m:
                m["sample_ID"] = to_sample_id(m["sample_ID"])
            out[well] = m
        return out

    def pick_wells(self, wells: str | Iterable[str] | None = None) -> list[str]:
        """Return the chosen wells sorted by plate row, then by column.

        A name that is not a well raises `ValueError` from `fouri.metadata.normalize_well`.

        Parameters
        ----------
        wells : str | Iterable[str] | None, default=None
            A well name or well names. None for every well in `well_list`.

        Returns
        -------
        list[str]
            Well names normalized to the zero-padded form (e.g., `c2` becomes `C02`), without
            repeats.

        Raises
        ------
        ValueError
            If a well is not in the experiment's `well_list`.
        """
        if wells is None:
            wells = self.well_list
        if isinstance(wells, str):
            wells = [wells]
        chosen = {normalize_well(well) for well in wells}
        unknown = sorted(chosen - {normalize_well(well) for well in self.well_list}, key=plate_sort_key)
        if unknown:
            raise ValueError(
                f"Not in this experiment: {', '.join(unknown)}. Its wells are listed in {self.config_path}."
            )
        return sorted(chosen, key=plate_sort_key)

    @property
    def metadata_columns(self) -> list[str]:
        """The metadata columns that are used by at least one well in `well_list`.

        Ordered as `sample_ID`, then the default condition columns (`treatment`, `dose`,
        `treatment_time`), then the remaining columns. Each group is in order of first appearance,
        reading the wells in `well_list` order.
        """
        cols = []
        for m in self.well_metadata.values():
            for c in m:
                if c not in cols:
                    cols.append(c)
        return sorted(cols, key=lambda c: (c != "sample_ID", c not in DEFAULT_CONDITION_COLUMNS, cols.index(c)))

    # -- alignment ------------------------------------------------------------
    @property
    def anchor_round(self) -> int:
        """The imaging round that all other rounds are aligned to.

        The value is the round's 0-based index position in `info_exp.csv`. Read from `anchor_round`
        in the config. 0 if not set. Cells are outlined on this round's segmentation.

        Raises
        ------
        ValueError
            If `anchor_round` in the config is not an integer of 0 or more.
        """
        value = self.config.get("anchor_round", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f'"anchor_round" in {self.config_path} is {value!r}. It must be a whole number counting '
                "from 0: 0 is the first round in info_exp.csv, 1 the second."
            )
        return value

    @property
    def condition_columns(self) -> list[str]:
        """The config's condition columns. `DEFAULT_CONDITION_COLUMNS` if not set."""
        return list(self.config.get("condition_columns", DEFAULT_CONDITION_COLUMNS))

    @property
    def derived_features(self) -> list[dict[str, str]]:
        """The derived features, as dicts with `name` and `formula` keys. Empty if not set."""
        return list(self.config.get("derived_features", []))

    # -- folders --------------------------------------------------------------
    @property
    def info_file(self) -> str:
        """The path to the `info_exp.csv` file."""
        return self.paths["info_file"]

    @property
    def data_dir(self) -> str:
        """The path to the directory of raw ND2 files (`data`)."""
        return self.paths["data"]

    @property
    def output_df(self) -> str:
        """The path to the directory of round tables and transforms (`output_df`)."""
        return self.paths["output_df"]

    @property
    def output_images(self) -> str:
        """The path to the directory of converted TIFF images (`output_images`)."""
        return self.paths["output_images"]

    @property
    def segmented(self) -> str:
        """The path to the directory of label (segmentation mask) files (`segmented`)."""
        return self.paths["segmented"]

    @property
    def masks(self) -> str:
        """The path to the directory of artifact mask files (`masks`)."""
        return self.paths["masks"]

    @property
    def cell_data(self) -> str:
        """The path to the directory of per-well cell measurement tables (`cell_data`)."""
        return self.paths["cell_data"]

    @property
    def exports(self) -> str:
        """The path to the directory of exported cell tables (`exports`)."""
        return self.paths["exports"]

    # dict-style access for code written against the older project_config
    def __getitem__(self, item: str) -> Any:
        """Return a config value, as in a legacy `project_config` dict, `base_dir` resolved."""
        if item == "base_dir":
            return self.base_dir
        return self.config[item]

    def get(self, item: str, default: Any = None) -> Any:
        """Return a config value or `default`, as in a legacy `project_config` dict."""
        try:
            return self[item]
        except KeyError:
            return default

    def __repr__(self) -> str:
        """Return the experiment key, base directory, and well count as text."""
        return f"Experiment({self.key!r}, base_dir={self.base_dir!r}, {len(self.well_list)} wells)"


def _resolve_base_dir(config: dict[str, Any], config_path: str) -> str:
    """Return the base directory from `mac_base_dir`, `base_dir`, or `config_path`'s directory."""
    if platform.system() != "Windows" and (config.get("mac_base_dir") or "").strip():
        return os.path.normpath(config["mac_base_dir"].strip())
    if (config.get("base_dir") or "").strip():
        return os.path.normpath(config["base_dir"].strip())
    return os.path.dirname(config_path)


def config_path_for(path: str | Path) -> str:
    """Return the `experiment.json` path for a file or an experiment directory, `~` expanded."""
    path = os.path.expanduser(str(path))
    if os.path.isdir(path):
        return os.path.join(path, CONFIG_NAME)
    return path


def load_config(path: str | Path, announce: bool = True) -> Experiment:
    """Read an experiment's `experiment.json` file into an `Experiment`.

    Parameters
    ----------
    path : str | Path
        The path to the `experiment.json` file or to the experiment directory holding it.
    announce : bool, default=True
        If True, report the experiment's details through `announce_experiment`.

    Returns
    -------
    Experiment
        The experiment found in the file.

    Raises
    ------
    FileNotFoundError
        If the `experiment.json` file does not exist.
    """
    config_path = config_path_for(path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No experiment.json at {config_path}.")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    exp = Experiment(config, config_path)
    if announce:
        announce_experiment(exp)
    return exp


def announce_experiment(exp: Experiment) -> None:
    """Report the experiment's key, paths, and well count, and warn of a missing input path."""
    say(f"Experiment: {exp.key or '(no key)'}" + (f" - {exp.name}" if exp.name else ""))
    say(f"  config:   {exp.config_path}")
    say(f"  base_dir: {exp.base_dir}")
    say(f"  wells:    {len(exp.well_list)}")
    if not os.path.isdir(exp.base_dir):
        say(f"  WARNING: base_dir does not exist: {exp.base_dir}")
    for name in ("data", "info_file"):
        if not os.path.exists(exp.paths[name]):
            say(f"  WARNING: {name} not found: {exp.paths[name]}")


def save_config(config: dict[str, Any], path: str | Path) -> str:
    """Write a config to an `experiment.json` file.

    Sorts `well_list` by plate row, then by column. Drops the `well_metadata` entries of wells not
    in `well_list`. Copies an existing file to `experiment.json.bak` before writing. Writes
    atomically through `fouri.io.written_whole`.

    Parameters
    ----------
    config : dict[str, Any]
        The config to write, as held in `Experiment.config`.
    path : str | Path
        The path to the `experiment.json` file or to the experiment directory holding it.

    Returns
    -------
    str
        The path of the file written.

    Notes
    -----
    `config` itself is not modified.
    """
    from .io import written_whole

    config = dict(config)
    config["well_list"] = sorted(config.get("well_list", []), key=plate_sort_key)
    meta = config.get("well_metadata") or {}
    config["well_metadata"] = {w: meta[w] for w in config["well_list"] if w in meta}
    config_path = config_path_for(path)
    if os.path.exists(config_path):
        shutil.copyfile(config_path, config_path + ".bak")
    with written_whole(config_path) as partial, open(partial, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return config_path
