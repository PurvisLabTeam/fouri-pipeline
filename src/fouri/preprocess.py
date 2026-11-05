"""Build each channel's round table and convert ND2 files to single-channel tiffs."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import pandas as pd

from fouri.config import Experiment
from fouri.io import (
    nd2_metadata,
    read_info_file,
    read_nd2,
    save_round_table,
    write_tiff,
)
from fouri.parallel import parallel_map


def convert_well(exp: Experiment, well: str, align_signal: str = "DNA") -> None:
    table = round_table(exp, well, align_signal)

    output_images = Path(exp.output_images) / well
    output_images.mkdir(parents=True, exist_ok=True)

    for (datadir, subdir, file), rows in table.groupby(["dir", "sub_dir", "file"]):
        image = read_nd2(Path(str(datadir)) / str(subdir) / str(file))
        for _, row in rows.iterrows():
            channel = image[int(row["channel_in_file"])]
            channel = channel[: int(row["height_min"]), : int(row["width_min"])]
            write_tiff(output_images / str(row["raw_file_name"]), channel, metadata=_ome_metadata(row))

    save_round_table(exp, well, table)


def preprocess(exp: Experiment, align_signal: str = "DNA", n_workers: int | None = None) -> None:
    path_df = Path(exp.output_df)
    path_images = Path(exp.output_images)

    path_df.mkdir(parents=True, exist_ok=True)
    path_images.mkdir(parents=True, exist_ok=True)

    parallel_map(
        function=functools.partial(convert_well, exp, align_signal=align_signal),
        items=exp.well_list,
        desc="Converting wells",
        unit="well",
        n_workers=n_workers,
    )


def round_table(exp: Experiment, well: str, align_signal: str = "DNA") -> pd.DataFrame:
    rounds = read_info_file(exp)
    rows = []
    for round_index, (_, round_info) in enumerate(rounds.iterrows()):
        rows.extend(_round_rows(exp, well, round_index, round_info))
    df = pd.DataFrame(rows)
    df["width_min"] = df["width"].min()
    df["height_min"] = df["height"].min()
    df["is_align_channel"] = df["signal"] == align_signal
    duplicated = df[df.duplicated(["round_name", "signal"], keep=False)]
    if not duplicated.empty:
        message = f"Duplicate round/signal names in well {well} rounds table:\n\n"
        duplicated = duplicated.drop_duplicates(["round_name", "signal"])
        for row in duplicated.itertuples():
            message += f"{row.round_name} duplicates {row.signal}\n"
        message += "Correct info_exp.csv then run again"
        raise ValueError(message)

    align_counts = df.groupby("round_name")["is_align_channel"].sum().reindex(rounds["round_name"], fill_value=0)
    invalid = align_counts[align_counts != 1]
    if not invalid.empty:
        message = f"Some rounds have an invalid number of '{align_signal}' channels. Each must have exactly 1:\n"
        for round_name, n_channels in invalid.items():
            message += f"\tWell {well}: {round_name} has {n_channels} '{align_signal}' channels\n"
        message += "Check info_exp.csv or align_signal."
        raise ValueError(message)

    return df


def _round_rows(exp: Experiment, well: str, round_index: int, round_info: pd.Series[Any]) -> list[dict[str, Any]]:
    round_name = str(round_info["round_name"])
    subdir = next((d for d in Path(exp.data_dir).iterdir() if f"Round_{round_name}_" in d.name and d.is_dir()), None)
    if subdir is None:
        raise FileNotFoundError(f"No folder for round {round_name}")
    fpath = next((f for f in subdir.iterdir() if f"Well{well}_" in f.name and f.name.endswith(".nd2")), None)
    if fpath is None:
        raise FileNotFoundError(f"No file found for round {round_name}, well {well}")

    metadata = nd2_metadata(fpath)
    shared = {
        "dir": str(exp.data_dir),
        "sub_dir": subdir.name,
        "file": fpath.name,
        "round_name": round_name,
        "round_index": round_index,
        "well": well,
        "width": metadata.width,
        "height": metadata.height,
        "pixel_size_um": metadata.pixel_size_um,
        "objective": metadata.objective,
        "bit_depth": metadata.bit_depth,
        "acquired_at": metadata.acquired_at,
    }
    rows = []
    for c, channel in enumerate(metadata.channels):
        if channel.name not in round_info.index:
            raise ValueError(f"info_exp.csv: column for channel {channel.name} does not exist.")
        cell = round_info[channel.name]
        signal = str(cell).strip()
        if pd.isna(cell) or signal == "":
            continue
        rows.append(
            {
                **shared,
                "channel_in_file": c,
                "channel_name": channel.name,
                "signal": signal,
                "excitation_nm": channel.excitation_nm,
                "emission_nm": channel.emission_nm,
                "exposure_ms": channel.exposure_ms,
                "raw_file_name": f"Round_{round_name}_well{well}_{signal}.tif",
            }
        )

    return rows


def _ome_metadata(row: pd.Series[Any]) -> dict[str, Any]:
    row_asdict = row.dropna().to_dict()
    header = {
        "axes": "YX",
        "Name": Path(str(row_asdict.get("raw_file_name"))).stem,
        "PhysicalSizeX": row_asdict.get("pixel_size_um"),
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": row_asdict.get("pixel_size_um"),
        "PhysicalSizeYUnit": "µm",
        "Channel": {
            "Name": row_asdict.get("signal"),
            "ExcitationWavelength": row_asdict.get("excitation_nm"),
            "ExcitationWavelengthUnit": "nm",
            "EmissionWavelength": row_asdict.get("emission_nm"),
            "EmissionWavelengthUnit": "nm",
        },
        "Plane": {"ExposureTime": row_asdict.get("exposure_ms"), "ExposureTimeUnit": "ms"},
    }

    return header
