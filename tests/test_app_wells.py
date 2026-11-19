import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from builders import INFO_EXP, ND2_FILE, ND2_FOLDER, four_wells, with_filters, with_metadata

from fouri.app import wells
from fouri.app.wells import fill_sample_ids, in_full, new_config, path_entry, save_wells, wells_table
from fouri.config import Experiment

SMALL_AREAS = {"name": "small areas", "use": True, "rule": "between", "feature": "nuc_area", "low": 0, "high": 40}
CONDITIONS = ["treatment", "treatment_time"]


def _saved(exp: Experiment) -> dict[str, Any]:
    config: dict[str, Any] = json.loads(Path(exp.config_path).read_text(encoding="utf-8"))
    return config


def _sample_ids(exp: Experiment) -> dict[str, Any]:
    return {well: entry.get("sample_ID") for well, entry in _saved(exp)["well_metadata"].items()}


def _set(table: pd.DataFrame, well: str, **values: Any) -> pd.DataFrame:
    table = table.copy()
    for column, value in values.items():
        if column not in table.columns:
            table[column] = ""
        table[column] = table[column].astype(object)
        table.loc[table["well"] == well, column] = value
    return table


def test_the_paths_save_keeps_keys_saved_since(experiment: Experiment) -> None:
    with_filters(experiment, [SMALL_AREAS])

    config = wells.config_to_save(experiment, data="", info_file="", results="", anchor=None)

    assert config.get("app", {}).get("filters") == [SMALL_AREAS]


def test_a_well_listed_twice_is_refused(experiment: Experiment) -> None:
    table = wells_table(experiment)
    twice = pd.concat([table, _set(table, "B02", treatment="palbo").iloc[[0]]], ignore_index=True)
    before = Path(experiment.config_path).read_bytes()

    with pytest.raises(ValueError):
        save_wells(experiment, twice, CONDITIONS)

    assert Path(experiment.config_path).read_bytes() == before


def test_an_id_held_by_two_conditions_is_split_on_save(tmp_path: Path) -> None:
    exp = four_wells(tmp_path)
    exp = with_metadata(exp, "B03", treatment="palbo", sample_ID=1)
    exp = with_metadata(exp, "C02", sample_ID=2)
    exp = with_metadata(exp, "C03", treatment_time="96h", sample_ID=2)

    save_wells(exp, wells_table(exp), CONDITIONS)

    assert _sample_ids(exp) == {"B02": 1, "B03": 2, "C02": 2, "C03": 3}


def test_fill_sample_ids_keeps_ids_and_numbers_in_plate_order() -> None:
    table = pd.DataFrame(
        [
            ("C03", "", "palbo", "24h"),
            ("B02", 1, "control", "24h"),
            ("C04", "", "palbo", "24h"),
            ("C02", 2, "palbo", "96h"),
            ("B04", "", "dmso", "24h"),
            ("B03", "", "control", "24h"),
        ],
        columns=["well", "sample_ID", "treatment", "treatment_time"],
    )

    filled = fill_sample_ids(table, CONDITIONS)

    assert dict(zip(filled["well"], filled["sample_ID"], strict=True)) == {
        "B02": 1,
        "B03": 1,
        "B04": 3,
        "C02": 2,
        "C03": 4,
        "C04": 4,
    }
    assert filled["well"].tolist() == table["well"].tolist()


def test_save_wells_keeps_condition_columns_and_metadata(experiment: Experiment) -> None:
    table = wells_table(experiment)
    table = _set(table, "B02", cell_line="RPE1", plate_lot="L1")
    table = _set(table, "B03", cell_line="RPE1", plate_lot="L2")
    table = _set(table, "C02", cell_line="MCF10A", plate_lot="L3")
    conditions = ["treatment", "dose", "treatment_time", "cell_line"]

    save_wells(experiment, table.iloc[::-1].reset_index(drop=True), conditions)

    saved = _saved(experiment)
    assert set(saved["condition_columns"]) == set(conditions)
    assert saved["well_list"] == ["B02", "B03", "C02"]
    assert saved["well_metadata"] == {
        "B02": {
            "sample_ID": 1,
            "treatment": "control",
            "treatment_time": "24h",
            "cell_line": "RPE1",
            "plate_lot": "L1",
        },
        "B03": {
            "sample_ID": 2,
            "treatment": "palbo",
            "treatment_time": "24h",
            "cell_line": "RPE1",
            "plate_lot": "L2",
        },
        "C02": {
            "sample_ID": 3,
            "treatment": "palbo",
            "treatment_time": "96h",
            "cell_line": "MCF10A",
            "plate_lot": "L3",
        },
    }


def test_sample_ids_are_saved_as_numbers(experiment: Experiment) -> None:
    exp = with_metadata(experiment, "B02", sample_ID="3")
    exp = with_metadata(exp, "C02", sample_ID="4")
    table = _set(wells_table(exp), "B03", treatment="control", sample_ID=3)

    save_wells(exp, table, CONDITIONS)

    ids = _sample_ids(exp)
    assert ids == {"B02": 3, "B03": 3, "C02": 4}
    assert all(type(value) is int for value in ids.values())


def test_new_config_refuses_an_existing_experiment_json(experiment: Experiment) -> None:
    config_path = Path(experiment.config_path)
    before = config_path.read_bytes()

    with pytest.raises(ValueError):
        new_config(config_path.parent, data="", info_file="")

    assert config_path.read_bytes() == before


def test_new_config_gives_every_well_sample_id_1(tmp_path: Path) -> None:
    (tmp_path / "info_exp.csv").write_text(INFO_EXP, encoding="utf-8")
    folder = tmp_path / "data" / ND2_FOLDER.format(round_name="R0")
    folder.mkdir(parents=True)
    for well in ("B02", "B03", "C02"):
        (folder / ND2_FILE.format(well=well)).touch()

    config = new_config(tmp_path, data="", info_file="")

    assert config["well_metadata"] == {well: {"sample_ID": 1} for well in ("B02", "B03", "C02")}


def test_path_entry_writes_paths_relative_to_base_dir(tmp_path: Path) -> None:
    base = str(tmp_path / "results")
    inside = os.path.join(base, "raw", "nd2")
    outside = str(tmp_path / "elsewhere" / "nd2")
    sibling = os.path.join(base + "_old", "nd2")

    assert path_entry("", base, "data") is None
    assert path_entry("data", base, "data") is None
    assert path_entry(os.path.join(base, "data"), base, "data") is None
    assert path_entry(inside, base, "data") == "raw/nd2"
    assert path_entry(os.path.join("raw", "nd2"), base, "data") == "raw/nd2"
    assert path_entry(outside, base, "data") == outside
    assert path_entry(sibling, base, "data") == sibling
    assert path_entry(os.path.join(base, "..", "elsewhere", "nd2"), base, "data") == outside
    assert in_full(os.path.join("raw", "nd2"), base) == inside
    assert in_full(outside, base) == outside

    elsewhere = str(tmp_path / "configs" / "experiment.json")
    with_base_dir = Experiment({"base_dir": base, "paths": {"data": path_entry(inside, base, "data")}}, elsewhere)
    in_folder = Experiment(
        {"paths": {"data": path_entry(inside, base, "data")}}, os.path.join(base, "experiment.json")
    )
    assert with_base_dir.data_dir == inside
    assert in_folder.data_dir == inside
