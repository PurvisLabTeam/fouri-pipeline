from __future__ import annotations

import json
import os
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
        "well_metadata": {},
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


def fill_sample_ids(table: pd.DataFrame, condition_columns: Sequence[str]) -> pd.DataFrame:
    filled = table.copy()
    if SAMPLE_ID not in filled.columns:
        filled[SAMPLE_ID] = ""
    columns = [column for column in condition_columns if column in filled.columns]
    ids = [as_sample_id(value) for value in filled[SAMPLE_ID]]
    conditions = [_condition(row, columns) for _, row in filled.iterrows()]
    wells = filled["well"].tolist()
    order = sorted(range(len(wells)), key=lambda index: plate_sort_key(wells[index]))
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


def _read(exp: Experiment) -> dict[str, Any]:
    with open(exp.config_path, encoding="utf-8") as file:
        config: dict[str, Any] = json.load(file)
    return config


def save_wells(exp: Experiment, table: pd.DataFrame, condition_columns: Sequence[str]) -> Experiment:
    filled = fill_sample_ids(table, condition_columns)
    well_list, well_metadata = table_to_config(filled)
    kept = [
        column
        for column in condition_columns
        if column in DEFAULT_CONDITION_COLUMNS or (column in filled.columns and not filled[column].map(blank).all())
    ]
    config = _read(exp)
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
