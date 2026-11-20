"""Readers, writers, and paths for the pipeline's input and output files.

Paths derive from the directories of a `fouri.config.Experiment`. Writes use `written_whole` for
atomic writes. `read_round_table` and `load_transforms` read legacy `df_{well}.pkl` and
`tmat_{well}.pkl` when the current file is absent.
"""

import os
import pickle
import re
import sys
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import nd2
import numpy as np
import numpy.typing as npt
import pandas as pd
from tifffile import imread, imwrite

from fouri.config import Experiment

APP_DIR = "4i Pipeline"
EXPOSURE_UNIT_VALUE: Final[dict[str, float]] = {
    "s": 1e3,
    "ms": 1,
    "µs": 1e-3,
    "ns": 1e-6,
}
COMBAT_LAYER: Final[str] = "combat"


@dataclass(frozen=True)
class ChannelMetadata:
    """Acquisition settings for a single channel, as read from an ND2 file.

    Attributes
    ----------
    name : str
        The channel's name. Must match the column headers of `info_exp.csv`.
    excitation_nm : float | None
        Excitation wavelength in nm. None if not recorded.
    emission_nm : float | None
        Emission wavelength in nm. None if not recorded.
    exposure_ms : float | None
        Exposure time in milliseconds. None if not recorded or recorded with an unrecognized unit
        (not one of seconds, milliseconds, microseconds, or nanoseconds).
    """

    name: str
    excitation_nm: float | None
    emission_nm: float | None
    exposure_ms: float | None


@dataclass(frozen=True)
class ND2Metadata:
    """Image metadata derived from the ND2 files.

    Attributes
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    pixel_size_um : float
        Pixel size along `x` in micrometers.
    objective : str
        The name of the objective.
    bit_depth : int | None
        The number of significant bits per pixel.
    acquired_at : str
        The acquisition date and time of the first image. `None` as text if it was not recorded.
    channels : list[ChannelMetadata]
        Acquisition settings for each channel in file order.
    """

    width: int
    height: int
    pixel_size_um: float
    objective: str
    bit_depth: int | None
    acquired_at: str
    channels: list[ChannelMetadata]


def make_dir(path: str | Path) -> None:
    """Create a directory including any missing parent directories."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _normalize_exposure(exposure: float | None, exposure_time_unit: str | None) -> float | None:
    if exposure is None or exposure_time_unit is None:
        return None
    factor = EXPOSURE_UNIT_VALUE.get(exposure_time_unit)
    if factor is None:
        return None
    return exposure * factor


def app_data_dir() -> Path:
    """Return the app-data directory for the current platform.

    On Windows: `%LOCALAPPDATA%/4i Pipeline`, or `~/AppData/Local/4i Pipeline` if `LOCALAPPDATA` is
    unset or empty. On other platforms: `~/.local/share/4i Pipeline`.

    Returns
    -------
    Path
        The path of the app-data directory.

    Notes
    -----
    The directory is not created.
    """
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
    else:
        base = Path.home() / ".local" / "share"
    return base / APP_DIR


def fullest_df_path(exp: Experiment) -> Path:
    """Return the path to the `fullest_df.csv` file in the experiment's base directory."""
    return Path(exp.base_dir) / "fullest_df.csv"


def adata_path(exp: Experiment) -> Path:
    """Return the path to the `standard_adata.h5ad` file in the experiment's base directory."""
    return Path(exp.base_dir) / "standard_adata.h5ad"


def adata_sub_path(exp: Experiment) -> Path:
    """Return the path to the `standard_adata_sub.h5ad` file in the experiment's base directory."""
    return Path(exp.base_dir) / "standard_adata_sub.h5ad"


def run_record_path(exp: Experiment) -> Path:
    """Return the path to the `run_record.jsonl` file in the experiment's base directory."""
    return Path(exp.base_dir) / "run_record.jsonl"


def export_path(exp: Experiment, name: str, file_format: str) -> Path:
    """Return the path of `exports/<name>.<file_format>`, `file_format` without its dot (`h5ad`)."""
    return Path(exp.exports) / f"{name}.{file_format}"


def read_info_file(exp: Experiment) -> pd.DataFrame:
    """Read the `info_exp.csv` file, with the first column renamed to `round_name`.

    Parameters
    ----------
    exp : Experiment
        The Experiment that holds the file.

    Returns
    -------
    pd.DataFrame
        info_exp.csv as a pandas DataFrame.

    Notes
    -----
    The first column contains the round names, regardless of its header. Every other column is an
    ND2 channel name. Each value is the signal (stain, marker, etc.) in that channel for the round.
    An empty value is an unused channel.
    """
    info = Path(exp.info_file)
    df = pd.read_csv(info)
    df.rename(columns={df.columns[0]: "round_name"}, inplace=True)
    return df


def nd2_metadata(path: str | Path) -> ND2Metadata:
    """Read the image metadata of an ND2 file.

    Parameters
    ----------
    path : str | Path
        The path to the ND2 file.

    Returns
    -------
    ND2Metadata
        The image metadata found in the file.

    Raises
    ------
    ValueError
        If the file does not contain channel metadata.
    """
    path = Path(path)
    with nd2.ND2File(path) as fh:
        sizes = fh.sizes
        pixel_size_um = fh.voxel_size().x
        metadata = fh.metadata
        ome_metadata = fh.ome_metadata()

    channels = metadata.channels
    if not channels:
        raise ValueError(f"{path.name} has no channel information, so its channels cannot be matched to info_exp.csv")

    return ND2Metadata(
        width=sizes["X"],
        height=sizes["Y"],
        pixel_size_um=pixel_size_um,
        objective=str(channels[0].microscope.objectiveName),
        bit_depth=int(channels[0].volume.bitsPerComponentSignificant),
        acquired_at=str(ome_metadata.images[0].acquisition_date),
        channels=[
            ChannelMetadata(
                name=ch.channel.name,
                excitation_nm=ch.channel.excitationLambdaNm,
                emission_nm=ch.channel.emissionLambdaNm,
                exposure_ms=_normalize_exposure(
                    ome_metadata.images[0].pixels.planes[c].exposure_time,
                    ome_metadata.images[0].pixels.planes[c].exposure_time_unit.value,
                ),
            )
            for c, ch in enumerate(channels)
        ],
    )


def read_nd2(path: str | Path) -> npt.NDArray:
    """Read all channels of an ND2 file into memory and close it.

    Parameters
    ----------
    path : str | Path
        The path to the ND2 file.

    Returns
    -------
    npt.NDArray
        The image found in the file, shaped (channels, height, width).

    Raises
    ------
    ValueError
        If the file contains any axes other than `C`, `Y`, and `X`.

    Notes
    -----
    If a file has no channel axis, the image is still returned as (C, H, W) with a channel axis
    length of 1.
    """
    path = Path(path)
    with nd2.ND2File(path) as fh:
        sizes = fh.sizes
        if invalid_axes := sorted(set(sizes) - {"C", "Y", "X"}):
            raise ValueError(f"{path.name} holds axes that are not valid: {invalid_axes}")

        image = np.asarray(fh.asarray())

    if "C" not in sizes:
        image = np.expand_dims(image, 0)

    return image


def label_paths(labels_path: str | Path, well: str) -> list[Path]:
    """List the label (segmentation mask) files for a well, in round order.

    Parameters
    ----------
    labels_path : str | Path
        The path to the experiment's `segmented` directory.
    well : str
        The well, zero-padded (e.g., `B02`).

    Returns
    -------
    list[Path]
        Paths to the label files, sorted by name. Empty if the well's directory does not exist.

    Notes
    -----
    Paths are `<labels_path>/<well>/labels_<well>_round_NNN.tif` sorted lexicographically. Files
    with names that do not match exactly are excluded, including any temporary files created by
    fouri-pipeline (ending in `.partial`) and renamed copies of files.
    """
    name_pattern = rf"labels_{re.escape(well)}_round_[0-9]{{3}}\.tif"
    folder = Path(labels_path) / well
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if re.fullmatch(name_pattern, path.name))


def label_file_name(well: str, round_index: int) -> str:
    """Return the name of the label file for a single (round, well)."""
    return f"labels_{well}_round_{round_index:03d}.tif"


def label_path(exp: Experiment, well: str, round_index: int) -> Path:
    """Return the path to the label file for a single (round, well)."""
    return Path(exp.segmented) / well / label_file_name(well, round_index)


def original_label_path(exp: Experiment, well: str, round_index: int) -> Path:
    """Return the path to the original Cellpose label file for a single (round, well)."""
    p = label_path(exp, well, round_index)
    return p.parent / "cellpose_original" / p.name


def read_labels(labels_path: str | Path, well: str, indices: Sequence[int] | None = None) -> npt.NDArray[Any]:
    """Read the label (segmentation mask) images for a well.

    Returns labels stacked in the order of `indices` or in round order when `indices` is None.

    Parameters
    ----------
    labels_path : str | Path
        The path to the experiment's `segmented` directory.
    well : str
        The well, zero-padded (e.g., `B02`).
    indices : Sequence[int] | None, default=None
        The 0-based round indices to read. None reads every label file for the well.

    Returns
    -------
    npt.NDArray[Any]
        The label images found in the files, shaped (rounds, height, width). Empty if `indices` is
        None and the well does not have any label files.

    Raises
    ------
    FileNotFoundError
        If the label file for a round in `indices` does not exist.

    Notes
    -----
    With `indices` None, the files are those listed by `label_paths`. Otherwise each file is
    `<labels_path>/<well>/labels_<well>_round_NNN.tif` for a round index in `indices`.
    """
    labels_path = Path(labels_path)
    if indices is None:
        paths = label_paths(labels_path, well)

    else:
        paths = []
        for round_idx in indices:
            path = labels_path / well / label_file_name(well, round_idx)
            if not path.exists():
                raise FileNotFoundError(f"Well {well}, round {round_idx}: could not find label file.")
            paths.append(path)

    return np.array([imread(pth) for pth in paths])


def mask_path(exp: Experiment, well: str, round_index: int) -> Path:
    """Return the path to the artifact mask file for a single (round, well)."""
    return Path(exp.masks) / well / f"masks_{well}_round_{round_index:03d}.npz"


def mask_paths(masks_path: str | Path, well: str) -> list[Path]:
    """List the artifact mask files for a well.

    Parameters
    ----------
    masks_path : str | Path
        The path to the experiment's `masks` directory.
    well : str
        The well, zero-padded (e.g., `B02`).

    Returns
    -------
    list[Path]
        Paths to the mask files, sorted lexicographically.
    """
    folder = Path(masks_path) / well
    return sorted(path for path in folder.iterdir() if path.suffix == ".npz")


def mask_record_path(exp: Experiment, well: str, round_index: int) -> Path:
    """Return the path to the mask record JSON file for a single (round, well)."""
    return mask_path(exp, well, round_index).with_suffix(".json")


@contextmanager
def written_whole(path: str | Path) -> Generator[Path]:
    """Yield a temporary `.partial` file path to write to and rename over `path`.

    Makes writes to `path` atomic. The temporary path is `path` with `.partial` appended to the name
    (e.g., `cell_data_B02_df.csv.partial`). If the block raises, the temporary file is removed. A
    file already at `path` is unaffected by a failed write. The exception is re-raised.

    Parameters
    ----------
    path : str | Path
        The path of the output file.

    Yields
    ------
    Path
        The temporary path to write to.
    """
    path = Path(path)
    partial_path = path.with_name(f"{path.name}.partial")
    try:
        yield partial_path
        Path.replace(partial_path, path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


def write_npz(path: Path, masks_to_write: Mapping[str, npt.NDArray[Any]]) -> None:
    """Write named arrays to a compressed `.npz` file."""
    with written_whole(path) as partial, open(partial, "wb") as fh:
        np.savez_compressed(fh, **masks_to_write, allow_pickle=False)


def write_tiff(path: str | Path, array: npt.NDArray[Any], metadata: dict[str, Any] | None = None) -> None:
    """Write an image to a zlib-compressed TIFF, with an OME header when `metadata` is given."""
    path = Path(path)
    with written_whole(path) as partial:
        imwrite(partial, array, compression="zlib", ome=metadata is not None, metadata=metadata)


def round_table_path(exp: Experiment, well: str) -> Path:
    """Return the path to the round table file for a single well."""
    return Path(exp.output_df) / f"round_table_{well}.csv"


def save_round_table(exp: Experiment, well: str, table: pd.DataFrame) -> None:
    """Write the round table for a single well to the round table file, without the index."""
    path = round_table_path(exp, well)
    with written_whole(path) as partial:
        table.to_csv(partial, index=False)


def read_round_table(exp: Experiment, well: str) -> pd.DataFrame:
    current = round_table_path(exp, well)
    deprecated = current.parent / f"df_{well}.pkl"

    if current.is_file():
        return pd.read_csv(current)

    if deprecated.is_file():
        table = pd.read_pickle(deprecated)
        table = table.rename(
            columns={"nameRound": "round_name", "alignRound": "round_index", "alignIm": "is_align_channel"}
        )
        table["is_align_channel"] = table["is_align_channel"].astype(bool)
        table["round_index"] = table["round_index"].astype(int)
        return table

    raise FileNotFoundError(f"Could not find the round table .csv or .pkl for well {well}")


def require_raw_files(table: pd.DataFrame, well: str) -> None:
    if "raw_file_name" not in table.columns:
        raise ValueError(
            f"Well {well}'s round table was made using an old version of the pipeline. Rerun Convert images first."
        )


def cell_data_path(exp: Experiment, well: str) -> Path:
    """Return the path to the cell data file for a single well."""
    return Path(exp.cell_data) / f"cell_data_{well}_df.csv"


def alignment_paths(exp: Experiment, well: str) -> list[Path]:
    table = read_round_table(exp, well)
    require_raw_files(table, well)
    parent = Path(exp.output_images) / well
    return [
        parent / str(row.raw_file_name)
        for row in table.query("is_align_channel").sort_values("round_index").itertuples()
    ]


def tmat_path(exp: Experiment, well: str, suffix: str | None = None) -> Path:
    """Return the path of `output_df/tmat_<well>` plus `suffix`, given with its dot (`.npy`)."""
    pth = Path(exp.output_df) / f"tmat_{well}"
    if suffix is not None:
        pth = pth.with_suffix(suffix)
    return pth


def save_transforms(exp: Experiment, well: str, transforms: npt.NDArray[np.float64]) -> Path:
    """Write the alignment transforms for a single well to a `.npy` transforms file.

    Parameters
    ----------
    exp : Experiment
        The Experiment holding the file.
    well : str
        The well, zero-padded (e.g., `B02`).
    transforms : npt.NDArray[np.float64]
        Rigid transform matrices, with shape (rounds, 3, 3) and indexed by round index. Each matrix
        maps (x, y, 1) pixel coordinates in the anchor round to those in its round. The anchor
        round's matrix is the identity matrix. A round that is not yet aligned is a matrix of NaN.

    Returns
    -------
    Path
        The path of the transforms file written.
    """
    save_path = tmat_path(exp, well, ".npy")
    with written_whole(save_path) as partial, open(partial, "wb") as fh:
        np.save(fh, transforms)
    return save_path


def load_transforms(exp: Experiment, well: str) -> npt.NDArray[np.float64]:
    current_path = tmat_path(exp, well, ".npy")
    deprecated_path = tmat_path(exp, well, ".pkl")

    if current_path.exists():
        transforms = np.load(current_path)
    elif deprecated_path.exists():
        with open(deprecated_path, "rb") as fh:
            transforms = pickle.load(fh)
        transforms = np.array([getattr(transform, "params", transform) for transform in transforms])
    else:
        raise FileNotFoundError(f"Could not find transforms for well {well}")

    return transforms
