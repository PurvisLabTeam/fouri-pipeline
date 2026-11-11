"""Write what each pipeline step would have written, for a small made-up experiment.

A test arranges a state in its temporary folder with these, without ND2 files, Cellpose or a GPU.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fouri import io, measure
from fouri.config import Experiment, load_config, save_config

INFO_EXP = """\
myRound,DAPI,Cy3,Cy5
R0,DNA,pRB,RB
R1,DNA,cMyc,
R2,DNA,CyclinA2,EdU
"""

CONFIG = {
    "experiment_key": "test",
    "experiment_name": "made-up",
    "well_list": ["B02", "B03", "C02"],
    "well_metadata": {
        "B02": {"treatment": "control", "treatment_time": "24h", "sample_ID": 1},
        "B03": {"treatment": "palbo", "treatment_time": "24h", "sample_ID": 2},
        "C02": {"treatment": "palbo", "treatment_time": "96h", "sample_ID": 3},
    },
    "condition_columns": ["treatment", "treatment_time"],
    "derived_features": [{"name": "pRB_RB_ratio", "formula": "R0_pRB_nuc_mean / R0_RB_nuc_mean"}],
}

SIZE = 64  # every image is SIZE x SIZE pixels

# Four 6 x 6 nuclei, one in each quarter of the image. Each sits in a bright square reaching 8 px
# past it: room for its ring (6 px wide after a 1 px gap) and the 1 px margin Measure cells keeps
# from pixels with no image data, so no cell is dropped.
NUCLEUS_STARTS = (13, 45)
NUCLEUS_SIDE = 6
SQUARE_MARGIN = 8

# Made-up acquisition values per optical configuration: excitation nm, emission nm, exposure ms.
ACQUISITION = {"DAPI": (405.0, 450.0, 100.0), "Cy3": (555.0, 590.0, 200.0), "Cy5": (640.0, 690.0, 300.0)}


def _squares(margin: int) -> list[tuple[slice, slice]]:
    """Return each nucleus's rows and columns, widened by ``margin`` on every side."""
    return [
        (slice(row - margin, row + NUCLEUS_SIDE + margin), slice(col - margin, col + NUCLEUS_SIDE + margin))
        for row in NUCLEUS_STARTS
        for col in NUCLEUS_STARTS
    ]


def _round_count(exp: Experiment) -> int:
    return len(io.read_info_file(exp))


def make_experiment(folder: Path) -> Experiment:
    """Write info_exp.csv and experiment.json for three wells and three rounds, then load both."""
    folder = Path(folder)
    (folder / "info_exp.csv").write_text(INFO_EXP, encoding="utf-8")
    config_path = folder / "experiment.json"
    config_path.write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")
    return load_config(config_path, announce=False)


def converted(exp: Experiment, well: str, rounds: int | None = None) -> None:
    """Write the round table and each channel's 64 x 64 image for the first ``rounds`` rounds."""
    info = io.read_info_file(exp)
    rows = []
    brightness = {}  # each image's file name and the value of its bright squares
    for round_index, (_, round_info) in enumerate(info.iloc[:rounds].iterrows()):
        round_name = str(round_info["round_name"])
        for channel_in_file, channel in enumerate(info.columns[1:]):
            cell = round_info[channel]
            if pd.isna(cell) or not str(cell).strip():
                continue
            signal = str(cell).strip()
            raw_file_name = f"Round_{round_name}_well{well}_{signal}.tif"
            brightness[raw_file_name] = 1000 * (channel_in_file + 1) + 100 * round_index
            excitation, emission, exposure = ACQUISITION[channel]
            rows.append(
                {
                    "dir": exp.data_dir,
                    "sub_dir": f"Round_{round_name}_made_up",
                    "file": f"Well{well}_ChannelDAPI,Cy3,Cy5_Seq0000.nd2",
                    "round_name": round_name,
                    "round_index": round_index,
                    "well": well,
                    "width": SIZE,
                    "height": SIZE,
                    "pixel_size_um": 0.65,
                    "objective": "Plan Apo 20x",
                    "bit_depth": 16,
                    "acquired_at": f"2026-11-0{round_index + 1} 10:00:00",
                    "channel_in_file": channel_in_file,
                    "channel_name": channel,
                    "signal": signal,
                    "excitation_nm": excitation,
                    "emission_nm": emission,
                    "exposure_ms": exposure,
                    "raw_file_name": raw_file_name,
                }
            )
    table = pd.DataFrame(rows)
    table["width_min"] = table["width"].min()
    table["height_min"] = table["height"].min()
    table["is_align_channel"] = table["signal"] == "DNA"

    Path(exp.output_df).mkdir(parents=True, exist_ok=True)
    io.save_round_table(exp, well, table)

    folder = Path(exp.output_images) / well
    folder.mkdir(parents=True, exist_ok=True)
    for raw_file_name, value in brightness.items():
        image = np.zeros((SIZE, SIZE), dtype=np.uint16)
        for square in _squares(SQUARE_MARGIN):
            image[square] = value
        io.write_tiff(folder / raw_file_name, image)


def segmented(exp: Experiment, well: str, rounds: Sequence[int] | None = None) -> None:
    """Write label images with four square nuclei for the given rounds, every round if None."""
    labels = np.zeros((SIZE, SIZE), dtype=np.uint16)
    for label, square in enumerate(_squares(0), start=1):
        labels[square] = label
    for round_index in range(_round_count(exp)) if rounds is None else rounds:
        path = io.label_path(exp, well, round_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        io.write_tiff(path, labels)


def aligned(exp: Experiment, well: str, rounds: Sequence[int] | None = None) -> None:
    """Write transforms: identity for the anchor and the given rounds (all if None), else NaN."""
    count = _round_count(exp)
    identity = set(range(count) if rounds is None else rounds) | {exp.anchor_round}
    transforms = np.full((count, 3, 3), np.nan)
    for round_index in identity:
        transforms[round_index] = np.eye(3)
    Path(exp.output_df).mkdir(parents=True, exist_ok=True)
    io.save_transforms(exp, well, transforms)


def measured(exp: Experiment, well: str) -> None:
    """Run ``measure.measure_well`` on a well already converted, segmented and aligned."""
    measure.measure_well(exp, well)


def measured_values(exp: Experiment, well: str, column: str, values: Sequence[float]) -> None:
    path = io.cell_data_path(exp, well)
    table = pd.read_csv(path)
    table[column] = list(values)
    table.to_csv(path, index=False)


def with_filters(exp: Experiment, filters: Sequence[Mapping[str, Any]]) -> Experiment:
    app = {**exp.config.get("app", {}), "filters": [dict(rule) for rule in filters]}
    save_config({**exp.config, "app": app}, exp.config_path)
    return load_config(exp.config_path, announce=False)
