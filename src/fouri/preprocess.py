"""Conversion of ND2 files to single-channel TIFF files, with a round table for each well.

`preprocess` is the Convert images step. `round_table` builds the round table for a well from
`info_exp.csv` and the well's ND2 files. `convert_well` writes the TIFF files and the round table
file for a well. `changed_wells` finds the converted wells whose rounds no longer match
`info_exp.csv`.
"""

from __future__ import annotations

import functools
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import tifffile

from fouri.artifacts import is_mask_edited
from fouri.config import Experiment
from fouri.io import (
    cell_data_path,
    label_path,
    label_paths,
    mask_paths,
    mask_record_path,
    nd2_metadata,
    original_label_path,
    read_info_file,
    read_nd2,
    read_round_table,
    round_table_path,
    save_round_table,
    tmat_path,
    write_npz,
    write_tiff,
)
from fouri.parallel import Failed, check_results, parallel_map
from fouri.report import say, say_skipped


@dataclass(frozen=True)
class ChangedWell:
    """A converted well with rounds changed in `info_exp.csv`.

    Attributes
    ----------
    well : str
        The well, zero-padded (e.g., `B02`).
    rounds : list[str]
        The names of the changed rounds as listed in the round table, in round order. For a well
        with a `df_<well>.pkl` file and no round table file, every round in `info_exp.csv`.
    hand_edited : list[str]
        The well's hand-edited rounds, changed or not, by name in round order. For a well with a
        `df_<well>.pkl` file and no round table file, the hand-edited rounds among those with a
        label file or an artifact mask file, each named `round N`, with `N` the 0-based round index.
        A round is hand-edited when the round has an original Cellpose label file (`fouri.segment`)
        or a mask record with `edited` true (`fouri.artifacts`).
    outputs : list[Path]
        The well's outputs that exist, ordered as the round table file, the `df_<well>.pkl` file,
        `<output_images>/<well>`, `<segmented>/<well>`, the transforms files (`.npy`, then `.pkl`),
        `<masks>/<well>`, and the cell data file.
    """

    well: str
    rounds: list[str]
    hand_edited: list[str]
    outputs: list[Path]


def _hand_edited(exp: Experiment, well: str, round_index: int) -> bool:
    return original_label_path(exp, well, round_index).exists() or is_mask_edited(
        mask_record_path(exp, well, round_index)
    )


def changed_wells(exp: Experiment, wells: Sequence[str] | None = None) -> list[ChangedWell]:
    """Find the converted wells whose rounds no longer match `info_exp.csv`.

    A well is changed when a round of the well's round table differs in name, channels, or signals
    from the round at the same 0-based index in `info_exp.csv`, or when `info_exp.csv` does not
    have a round at that index. A round added at the end of `info_exp.csv` is not a change. A well
    without a round table file is changed when the `df_<well>.pkl` file exists. A well with neither
    file is not changed. Reads the mask records of each changed well with `is_mask_edited`, which
    raises for a record that is not valid JSON or does not have an `edited` of true or false.

    Parameters
    ----------
    exp : Experiment
        The Experiment that holds the files.
    wells : Sequence[str] | None, default=None
        The wells to check. Must be in `well_list`. None checks every well in `well_list`.

    Returns
    -------
    list[ChangedWell]
        One `ChangedWell` per changed well, sorted by plate row, then by column.
    """
    wells = exp.pick_wells(wells)
    table = read_info_file(exp)
    listed: dict[int, tuple[str, dict[str, str]]] = {}
    for row_index, row in table.iterrows():
        channel_signals = {
            str(col).strip(): stripped
            for col, cell in row.items()
            if col != "round_name" and not pd.isna(cell) and (stripped := str(cell).strip())
        }
        listed[cast(int, row_index)] = (str(row["round_name"]), channel_signals)

    changed = []
    for well in wells:
        rtable_path = round_table_path(exp, well)
        old_table_path = Path(exp.output_df) / f"df_{well}.pkl"
        if rtable_path.exists():
            round_info = read_round_table(exp, well)
            saved: dict[int, tuple[str, dict[str, str]]] = {}
            for idx, rows in round_info.groupby("round_index"):
                rname = str(rows["round_name"].iloc[0])
                channel_signals = {
                    str(channel): str(signal)
                    for channel, signal in zip(rows["channel_name"], rows["signal"], strict=True)
                }
                saved[cast(int, idx)] = (rname, channel_signals)

            rounds_changed = [entry[0] for idx, entry in saved.items() if idx not in listed or entry != listed[idx]]
            if not rounds_changed:
                continue
            hand_edited = [round_name for idx, (round_name, _) in saved.items() if _hand_edited(exp, well, idx)]
        elif old_table_path.exists():
            rounds_changed = [round_name for round_name, _ in listed.values()]
            masks_dir = Path(exp.masks) / well
            round_files = [
                *label_paths(exp.segmented, well),
                *(mask_paths(exp.masks, well) if masks_dir.is_dir() else []),
            ]
            indices = sorted({int(path.stem[-3:]) for path in round_files})
            hand_edited = [f"round {idx}" for idx in indices if _hand_edited(exp, well, idx)]
        else:
            continue

        outputs: list[Path] = [
            rtable_path,
            old_table_path,
            Path(exp.output_images) / well,
            Path(exp.segmented) / well,
            tmat_path(exp, well, ".npy"),
            tmat_path(exp, well, ".pkl"),
            Path(exp.masks) / well,
            cell_data_path(exp, well),
        ]
        changed.append(
            ChangedWell(
                well=well,
                rounds=rounds_changed,
                hand_edited=hand_edited,
                outputs=[output for output in outputs if output.exists()],
            )
        )

    return changed


def trim(image: npt.NDArray[Any], width: int, height: int) -> npt.NDArray[Any]:
    """Crop an image to the first `height` rows and the first `width` columns."""
    return image[:height, :width]


def convert_well(exp: Experiment, well: str, align_signal: str = "DNA", overwrite: bool = False) -> None:
    """Write the TIFF files and the round table file for a single well.

    Builds the round table with `round_table`, whose refusals raise `ValueError` or
    `FileNotFoundError`. Writes each channel of the round table to
    `<output_images>/<well>/<raw_file_name>`, cropped to the top-left `width_min` by `height_min`
    pixels. Without `overwrite`, when a round table file exists with the same crop size (`width_min`
    and `height_min`), only the rounds not listed in the file are written.

    When the crop size differs from the file's, every round is written. The well's label files,
    original Cellpose label files, and every array of the well's artifact mask files are cropped to
    the new size after every TIFF file is written. The mask record files are unchanged. The round
    table file is written last.

    Parameters
    ----------
    exp : Experiment
        The Experiment that holds the files.
    well : str
        The well, zero-padded (e.g., `B02`).
    align_signal : str, default="DNA"
        The signal of each round's alignment channel, passed to `round_table`.
    overwrite : bool, default=False
        If True, every round is written when the crop size is unchanged.

    Notes
    -----
    Each TIFF file holds a single channel, with an OME header holding the pixel size and the
    channel's acquisition settings (`fouri.io.ND2Metadata`), and the signal as the channel name.
    """
    table = round_table(exp, well, align_signal)
    round_table_pth = round_table_path(exp, well)
    rounds_idx = table["round_index"].unique()
    to_write = table
    cut_to: tuple[int, int] | None = None
    if round_table_pth.exists():
        existing_round_table = read_round_table(exp, well)
        existing_dims = (
            int(existing_round_table["width_min"].iloc[0]),
            int(existing_round_table["height_min"].iloc[0]),
        )
        dims = (int(table["width_min"].iloc[0]), int(table["height_min"].iloc[0]))
        if existing_dims == dims and not overwrite:
            to_write = table[~table["round_name"].isin(existing_round_table["round_name"])]
        elif existing_dims != dims:
            cut_to = dims

    output_images = Path(exp.output_images) / well
    output_images.mkdir(parents=True, exist_ok=True)

    for (datadir, subdir, file), rows in to_write.groupby(["dir", "sub_dir", "file"]):
        image = read_nd2(Path(str(datadir)) / str(subdir) / str(file))
        for _, row in rows.iterrows():
            channel = image[int(row["channel_in_file"])]
            channel = trim(channel, width=int(row["width_min"]), height=int(row["height_min"]))
            write_tiff(output_images / str(row["raw_file_name"]), channel, metadata=_ome_metadata(row))

    if cut_to is not None:
        width, height = cut_to
        if (Path(exp.masks) / well).exists():
            for path in mask_paths(exp.masks, well):
                with np.load(path) as fh:
                    masks = {name: trim(arr, width, height) for name, arr in fh.items()}
                write_npz(path, masks)

        labels_to_trim: list[Path] = []
        labels_to_trim.extend(path for idx in rounds_idx if (path := label_path(exp, well, idx)).exists())
        labels_to_trim.extend(path for idx in rounds_idx if (path := original_label_path(exp, well, idx)).exists())
        for path in labels_to_trim:
            label = tifffile.imread(path)
            write_tiff(path, trim(label, width, height))

    save_round_table(exp, well, table)


def preprocess(
    exp: Experiment,
    align_signal: str = "DNA",
    n_workers: int | None = None,
    overwrite: bool = False,
    wells: Sequence[str] | None = None,
    reconvert_changed: bool = False,
) -> None:
    """Convert the ND2 files of the chosen wells to single-channel TIFF files.

    Converts each well with `convert_well`, in spawned processes. A well without a round table file
    is converted. A well whose round table lists the rounds of `info_exp.csv` is skipped. A well
    whose round table lists the first rounds of `info_exp.csv` is converted for the rounds added at
    the end. Without `reconvert_changed`, a changed well, as found by `changed_wells`, is refused
    regardless of `overwrite`. The refusal message names the changed rounds.

    Reports the wells skipped, the wells with added rounds, and the outputs deleted for each changed
    well. With no well to convert and no well refused, reports so and returns.
    `fouri.parallel.check_results` raises `StepFailed` naming each refused and failed well, after
    the other wells are converted, or `Stopped` when the step is stopped.

    Parameters
    ----------
    exp : Experiment
        The Experiment that holds the files.
    align_signal : str, default="DNA"
        The signal of each round's alignment channel. Must be the signal of exactly one channel in
        each round.
    n_workers : int | None, default=None
        The number of worker processes, capped at the number of wells to convert. None for
        `fouri.parallel.worker_count`.
    overwrite : bool, default=False
        If True, every well that is not changed is converted again, with every round written.
    wells : Sequence[str] | None, default=None
        The wells to convert. Must be in `well_list`. None for every well in `well_list`.
    reconvert_changed : bool, default=False
        If True, the outputs of each changed well (`ChangedWell.outputs`) are deleted, a directory
        with its contents, before any well is converted. The changed wells are converted again.
    """
    info = read_info_file(exp)
    round_names = info["round_name"].tolist()

    changed = {well.well: well for well in changed_wells(exp, wells)}
    worklist = []
    skipped = []
    added = []
    refused = []
    for well in exp.pick_wells(wells):
        changed_info = changed.get(well)
        if changed_info is not None:
            if not reconvert_changed:
                if round_table_path(exp, well).exists():
                    reason = (
                        f"Converted rounds no longer match info_exp.csv: rounds {', '.join(changed_info.rounds)} "
                        "have changed. Convert images again from the Run page and confirm or run python -m "
                        "fouri.run reconvert."
                    )
                else:
                    reason = (
                        "Converted by an older version of the pipeline (every round). Convert images again from "
                        "the Run page and confirm, or run python -m fouri.run reconvert."
                    )
                refused.append(Failed(well, reason))
                continue
            for output in changed_info.outputs:
                if output.is_dir():
                    shutil.rmtree(output)
                else:
                    output.unlink(missing_ok=True)
            say(
                f"{well}: rounds changed in info_exp.csv ({', '.join(changed_info.rounds)}). "
                "Removed its files from: "
                f"{', '.join(f'{o.parent.name}/{o.name}' for o in changed_info.outputs)} and reconverting..."
            )
            worklist.append(well)
            continue
        round_table = round_table_path(exp, well)
        if overwrite or not round_table.exists():
            worklist.append(well)
            continue

        well_round_table = read_round_table(exp, well)
        well_rounds = well_round_table.sort_values("round_index")["round_name"].unique().tolist()
        if well_rounds == round_names:
            skipped.append(well)
            continue

        if well_rounds == round_names[: len(well_rounds)]:
            worklist.append(well)
            added.append(well)
            continue

    if not worklist:
        if refused:
            check_results(refused, unit="well")  # raises StepFailed naming them
        say("Every well is already converted, nothing to convert. To overwrite them from scratch, use overwrite=True")
        return

    say_skipped(done="converted", skipped=skipped)
    if added:
        say(text=f"{', '.join(added)}: info_exp.csv has new rounds, converting only those.")

    path_df = Path(exp.output_df)
    path_images = Path(exp.output_images)

    path_df.mkdir(parents=True, exist_ok=True)
    path_images.mkdir(parents=True, exist_ok=True)

    results = parallel_map(
        function=functools.partial(convert_well, exp, align_signal=align_signal, overwrite=overwrite),
        items=worklist,
        desc="Converting wells",
        unit="well",
        n_workers=n_workers,
    )
    check_results([*refused, *results], unit="well")


def round_table(exp: Experiment, well: str, align_signal: str = "DNA") -> pd.DataFrame:
    """Build the round table for a single well from `info_exp.csv` and the well's ND2 files.

    The round table has one row per channel with a signal in each round of `info_exp.csv`, in round
    order, then in ND2 channel order. A blank value in `info_exp.csv` is a channel without a signal.
    Each round's directory is the directory in `<data_dir>` whose name contains
    `Round_<round_name>_`. The round's ND2 file is the file in that directory whose name contains
    `Well<well>_` and ends in `.nd2`.

    A missing round directory or ND2 file raises `FileNotFoundError`. Two round directories or two
    ND2 files matching a round raise `ValueError`. A signal listed under a channel that the ND2 file
    does not contain, or an ND2 channel without a column in `info_exp.csv`, raises `ValueError`.

    Parameters
    ----------
    exp : Experiment
        The Experiment that holds the files.
    well : str
        The well, zero-padded (e.g., `B02`).
    align_signal : str, default="DNA"
        The signal of each round's alignment channel.

    Returns
    -------
    pd.DataFrame
        The round table as a pandas DataFrame.

    Raises
    ------
    ValueError
        If a round names the same signal twice, or a round does not have exactly one channel with
        signal `align_signal`. The message names each such round.

    Notes
    -----
    The columns of the round table:

    - `dir`, `sub_dir`, and `file`: `<data_dir>`, the round directory's name, and the ND2 file's
      name.
    - `round_name` and `round_index`: the round's name and 0-based index in `info_exp.csv`.
    - `well`: the well.
    - `width`, `height`, `pixel_size_um`, `objective`, `bit_depth`, and `acquired_at`: the image
      metadata of the ND2 file (`fouri.io.ND2Metadata`).
    - `channel_in_file`, `channel_name`, and `signal`: the channel's 0-based index in the ND2 file,
      the ND2 channel name, and the signal.
    - `excitation_nm`, `emission_nm`, and `exposure_ms`: the channel's acquisition settings
      (`fouri.io.ChannelMetadata`).
    - `raw_file_name`: the converted image's file name,
      `Round_<round_name>_well<well>_<signal>.tif`.
    - `width_min` and `height_min`: the smallest image width and height across the well's rounds.
    - `is_align_channel`: true for the channel with signal `align_signal`.
    """
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
        message = f"Duplicate round/signal names in well {well} rounds table:\n"
        duplicated = duplicated.drop_duplicates(["round_name", "signal"])
        for row in duplicated.itertuples():
            message += f"{row.round_name} duplicates {row.signal}\n"
        message += "\nCorrect info_exp.csv then run again"
        raise ValueError(message)

    align_counts = df.groupby("round_name")["is_align_channel"].sum().reindex(rounds["round_name"], fill_value=0)
    invalid = align_counts[align_counts != 1]
    if not invalid.empty:
        message = f"Some rounds have an invalid number of '{align_signal}' channels. Each must have exactly 1:\n"
        for round_name, n_channels in invalid.items():
            message += f"Well {well}: {round_name} has {n_channels} '{align_signal}' channels\n"
        message += "\nCheck info_exp.csv or align_signal."
        raise ValueError(message)

    return df


def _round_rows(exp: Experiment, well: str, round_index: int, round_info: pd.Series[Any]) -> list[dict[str, Any]]:
    round_name = str(round_info["round_name"])
    subdirs = sorted(d for d in Path(exp.data_dir).iterdir() if f"Round_{round_name}_" in d.name and d.is_dir())
    if subdirs == []:
        raise FileNotFoundError(f"No folder for round {round_name}")
    if len(subdirs) > 1:
        raise ValueError(f"Multiple round folders found matching {round_name}: {', '.join(s.name for s in subdirs)}")
    subdir = subdirs[0]
    fpaths = sorted(f for f in subdir.iterdir() if f"Well{well}_" in f.name and f.name.endswith(".nd2"))
    if fpaths == []:
        raise FileNotFoundError(f"No file found for round {round_name}, well {well}")
    if len(fpaths) > 1:
        raise ValueError(f"Multiple ND2 files match {round_name}: {', '.join(f.name for f in fpaths)}")
    fpath = fpaths[0]
    metadata = nd2_metadata(fpath)
    channel_names = [channel.name for channel in metadata.channels]
    for column, cell in round_info.items():
        listed_signal = "" if pd.isna(cell) else str(cell).strip()
        if column != "round_name" and listed_signal and column not in channel_names:
            raise ValueError(
                f"{round_name} lists {listed_signal} under {column}, but the ND2 file of {round_name} for well {well} "
                f"has no {column} channel (it has {', '.join(channel_names)})."
            )
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
