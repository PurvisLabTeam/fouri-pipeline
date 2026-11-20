from __future__ import annotations

import json
import os
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath
from typing import Any

import numpy as np
import pandas as pd

from fouri.app.state import open_experiment
from fouri.config import CONFIG_NAME, DEFAULT_CONDITION_COLUMNS, DEFAULT_PATHS, Experiment, save_config
from fouri.io import read_info_file
from fouri.metadata import config_to_table, plate_sort_key, table_to_config, to_sample_id, wells_in_data

SAMPLE_ID = "sample_ID"
DETAILS = ("experiment_key", "experiment_name", "cell_type", "passage", "date_completed", "notes")
ALREADY_THERE = "This folder already has an experiment.json. Open it instead."
NO_ND2 = "No ND2 files for any well in {folder}."
CANT_READ_INFO = "Can't read info_exp.csv: {reason}"
LISTED_TWICE = "{wells} {verb} listed twice."


def same_path(first: str, second: str) -> bool:
    return os.path.normcase(os.path.normpath(first)) == os.path.normcase(os.path.normpath(second))


def in_full(value: str, base: str) -> str:
    path = os.path.expanduser(value.strip())
    pure = PurePath(path)
    return os.path.normpath(path if pure.drive or pure.root else os.path.join(base, path))


def _inside(path: str, folder: str) -> bool:
    try:
        return same_path(os.path.commonpath([path, folder]), folder)
    except ValueError:
        return False


def path_entry(value: str, base_dir: str, default: str) -> str | None:
    if not value.strip():
        return None
    path = in_full(value, base_dir)
    if same_path(path, os.path.join(base_dir, default)):
        return None
    if _inside(path, base_dir):
        return PurePath(os.path.relpath(path, base_dir)).as_posix()
    return path


def _results_key(config: dict[str, Any]) -> str:
    """Return the key Experiment reads the results folder from on this PC."""
    if platform.system() != "Windows" and (config.get("mac_base_dir") or "").strip():
        return "mac_base_dir"
    return "base_dir"


def config_to_save(exp: Experiment, *, data: str, info_file: str, results: str, anchor: int | None) -> dict[str, Any]:
    """Return the experiment's settings with the page's paths and anchor round, as Save writes them.

    The results folder goes to the key it is read from on this PC: ``mac_base_dir`` on a PC that
    isn't Windows when that is set, ``base_dir`` otherwise. That key is left out only when the
    folder would be read the same without it: the folder holding experiment.json, with no other
    key to take over, so on such a PC ``mac_base_dir`` stays while ``base_dir`` is set. The ND2
    folder and info_exp.csv go under ``paths``, each left out when its box is empty or it is the
    results folder's own ``data`` or ``info_exp.csv``, relative to the results folder when
    inside it, and in full otherwise. A path typed without a drive or root is relative to the
    results folder. ``anchor`` None keeps the saved anchor round.
    """
    config = _read(exp)
    home = os.path.dirname(exp.config_path)
    base_dir = in_full(results, home) if results.strip() else home
    key = _results_key(config)
    config[key] = base_dir
    without = {name: value for name, value in config.items() if name != key}
    if same_path(str(Experiment(without, exp.config_path).base_dir), base_dir):
        config = without
    paths = dict(config.get("paths") or {})
    for name, value in (("data", data), ("info_file", info_file)):
        entry = path_entry(value, base_dir, DEFAULT_PATHS[name])
        if entry is None:
            paths.pop(name, None)
        else:
            paths[name] = entry
    if paths:
        config["paths"] = paths
    else:
        config.pop("paths", None)
    if anchor is not None:
        config["anchor_round"] = anchor
    return config


def _reason(error: Exception) -> str:
    if isinstance(error, OSError) and error.strerror:
        return f"{error.strerror}: {error.filename}" if error.filename else error.strerror
    return str(error).rstrip(".")


def new_config(folder: Path, *, data: str, info_file: str) -> dict[str, Any]:
    base_dir = os.path.normpath(folder)
    config_path = os.path.join(base_dir, CONFIG_NAME)
    if os.path.exists(config_path):
        raise ValueError(ALREADY_THERE)
    paths: dict[str, str] = {}
    for name, value in (("data", data), ("info_file", info_file)):
        entry = path_entry(value, base_dir, DEFAULT_PATHS[name])
        if entry is not None:
            paths[name] = entry
    key = Path(base_dir).name
    exp = Experiment({"paths": paths}, config_path)
    wells = wells_in_data(exp.data_dir)
    if not wells:
        raise ValueError(NO_ND2.format(folder=exp.data_dir))
    try:
        read_info_file(exp)
    except (OSError, ValueError) as error:
        raise ValueError(CANT_READ_INFO.format(reason=_reason(error))) from error
    config: dict[str, Any] = {
        "experiment_key": key,
        "well_list": wells,
        "well_metadata": {well: {SAMPLE_ID: 1} for well in wells},
        "condition_columns": list(DEFAULT_CONDITION_COLUMNS),
    }
    if paths:
        config["paths"] = paths
    config["anchor_round"] = 0
    return config


def blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip()


def as_sample_id(value: object) -> object:
    if blank(value):
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    return to_sample_id(value)


def wells_table(exp: Experiment) -> pd.DataFrame:
    wells = sorted(exp.well_list, key=plate_sort_key)
    table = config_to_table(wells, exp.well_metadata)
    table[SAMPLE_ID] = pd.Series([as_sample_id(value) for value in table[SAMPLE_ID]], index=table.index, dtype=object)
    return table


def _condition(row: pd.Series, columns: list[str]) -> tuple[str, ...]:
    return tuple("" if blank(row[column]) else str(row[column]).strip() for column in columns)


def _ids_and_conditions(
    table: pd.DataFrame, condition_columns: Sequence[str]
) -> tuple[list[object], list[tuple[str, ...]], list[int]]:
    columns = [column for column in condition_columns if column in table.columns]
    given = table[SAMPLE_ID] if SAMPLE_ID in table.columns else pd.Series("", index=table.index)
    ids: list[object] = [as_sample_id(value) for value in given]
    conditions = [_condition(row, columns) for _, row in table.iterrows()]
    wells = table["well"].tolist()
    order = sorted(range(len(wells)), key=lambda index: plate_sort_key(wells[index]))
    return ids, conditions, order


def _unchanged(
    table: pd.DataFrame, conditions: list[tuple[str, ...]], saved_conditions: Mapping[str, tuple[str, ...]]
) -> list[bool]:
    wells = [str(well) for well in table["well"]]
    return [saved_conditions.get(well) == condition for well, condition in zip(wells, conditions, strict=True)]


def _split_off(
    ids: list[object], conditions: list[tuple[str, ...]], order: list[int], unchanged: list[bool]
) -> list[int]:
    holders: dict[object, list[int]] = {}
    for index in order:
        if ids[index] != "":
            holders.setdefault(ids[index], []).append(index)
    split: list[int] = []
    for indices in holders.values():
        held = list(dict.fromkeys(conditions[index] for index in indices))
        if len(held) < 2:
            continue
        kept = list(dict.fromkeys(conditions[index] for index in indices if unchanged[index]))
        keeper = kept[0] if len(kept) == 1 else held[0]
        split.extend(index for index in indices if conditions[index] != keeper)
    place = {index: position for position, index in enumerate(order)}
    return sorted(split, key=place.__getitem__)


def _conditions_in(config: Mapping[str, Any], condition_columns: Sequence[str]) -> dict[str, tuple[str, ...]]:
    metadata = config.get("well_metadata") or {}
    return {
        str(well): tuple(
            "" if blank(entry.get(column)) else str(entry.get(column)).strip() for column in condition_columns
        )
        for well, entry in metadata.items()
    }


def read_saved_conditions(exp: Experiment, condition_columns: Sequence[str]) -> dict[str, tuple[str, ...]]:
    return _conditions_in(_read(exp), condition_columns)


def split_wells(
    table: pd.DataFrame, condition_columns: Sequence[str], saved_conditions: Mapping[str, tuple[str, ...]]
) -> list[str]:
    ids, conditions, order = _ids_and_conditions(table, condition_columns)
    wells = table["well"].tolist()
    unchanged = _unchanged(table, conditions, saved_conditions)
    return [str(wells[index]) for index in _split_off(ids, conditions, order, unchanged)]


def fill_sample_ids(
    table: pd.DataFrame, condition_columns: Sequence[str], saved_conditions: Mapping[str, tuple[str, ...]]
) -> pd.DataFrame:
    filled = table.copy()
    if SAMPLE_ID not in filled.columns:
        filled[SAMPLE_ID] = ""
    ids, conditions, order = _ids_and_conditions(filled, condition_columns)
    unchanged = _unchanged(filled, conditions, saved_conditions)
    for index in _split_off(ids, conditions, order, unchanged):
        ids[index] = ""
    first: dict[tuple[str, ...], object] = {}
    for index in order:
        if ids[index] != "":
            first.setdefault(conditions[index], ids[index])
    numbers = [value for value in ids if isinstance(value, int) and not isinstance(value, bool)]
    next_id = max(numbers, default=0) + 1
    for index in order:
        if ids[index] == "":
            if conditions[index] not in first:
                first[conditions[index]] = next_id
                next_id += 1
            ids[index] = first[conditions[index]]
    filled[SAMPLE_ID] = pd.Series(ids, index=filled.index, dtype=object)
    return filled


def imported_rows(
    rows: Sequence[Mapping[str, Any]],
    kept: Mapping[str, Mapping[str, Any]],
    imported: pd.DataFrame,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    given = [str(column) for column in imported.columns]
    listed = list(dict.fromkeys(str(well) for well in imported["well"]))
    replaced: list[dict[str, Any]] = []
    for well in listed:
        existing = next((row for row in rows if row["well"] == well), None)
        start: dict[str, Any] = dict(existing or kept.get(well) or {"well": well, SAMPLE_ID: ""})
        for column in columns:
            start.setdefault(column, "")
        for _, line in imported[imported["well"] == well].iterrows():
            row = dict(start)
            for column in given:
                if column == SAMPLE_ID:
                    row[SAMPLE_ID] = as_sample_id(line[column])
                elif column != "well":
                    row[column] = line[column]
            replaced.append(row)
    listed_set = set(listed)
    return [dict(row) for row in rows if row["well"] not in listed_set] + replaced


def _read(exp: Experiment) -> dict[str, Any]:
    with open(exp.config_path, encoding="utf-8") as file:
        config: dict[str, Any] = json.load(file)
    return config


def save_wells(exp: Experiment, table: pd.DataFrame, condition_columns: Sequence[str]) -> Experiment:
    twice = sorted(set(table.loc[table["well"].duplicated(), "well"]), key=plate_sort_key)
    if twice:
        raise ValueError(LISTED_TWICE.format(wells=", ".join(twice), verb="is" if len(twice) == 1 else "are"))
    config = _read(exp)
    filled = fill_sample_ids(table, condition_columns, _conditions_in(config, condition_columns))
    well_list, well_metadata = table_to_config(filled)
    kept = [
        column
        for column in condition_columns
        if column in DEFAULT_CONDITION_COLUMNS or (column in filled.columns and not filled[column].map(blank).all())
    ]
    config["well_list"] = well_list
    config["well_metadata"] = well_metadata
    config["condition_columns"] = kept
    save_config(config, exp.config_path)
    return open_experiment(exp.config_path)


def save_details(exp: Experiment, details: Mapping[str, str]) -> Experiment:
    config = _read(exp)
    for key in DETAILS:
        if key in details:
            config[key] = details[key] if key == "notes" else details[key].strip()
    save_config(config, exp.config_path)
    return open_experiment(exp.config_path)
