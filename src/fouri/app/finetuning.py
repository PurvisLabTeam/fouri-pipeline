from __future__ import annotations

import shutil
from dataclasses import dataclass

from fouri.config import Experiment
from fouri.finetune import training_squares
from fouri.io import read_round_table, require_raw_files
from fouri.models import models_dir

SEPARATORS = ("/", "\\")


@dataclass(frozen=True)
class Regions:
    count: int
    rounds: dict[str, list[str]]


def corrected_regions(exp: Experiment) -> Regions:
    count = 0
    rounds: dict[str, list[str]] = {}
    for well in exp.pick_wells(None):
        try:
            table = read_round_table(exp, well)
        except FileNotFoundError:
            continue
        try:
            require_raw_files(table, well)
        except ValueError:
            continue
        squares = training_squares(exp, [well])
        if not squares:
            continue
        names = table.query("is_align_channel").sort_values("round_index")["round_name"].astype(str).tolist()
        count += len(squares)
        rounds[well] = [names[index] for index in sorted({square.round_index for square in squares})]
    return Regions(count, rounds)


def delete_model(name: str) -> None:
    directory = models_dir() / name
    if (
        name in ("", ".", "..")
        or directory.name != name
        or any(separator in name for separator in SEPARATORS)
        or name.endswith(".partial")
        or not (directory / "model.json").is_file()
    ):
        raise FileNotFoundError(f"Model {name} is not on this PC.")
    shutil.rmtree(directory)
