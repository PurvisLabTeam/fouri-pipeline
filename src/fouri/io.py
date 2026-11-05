"""File helpers shared by the pipeline notebooks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nd2
import numpy as np
import numpy.typing as npt
import pandas as pd
from tifffile import imread, imwrite

from fouri.config import Experiment

EXPOSURE_UNIT_VALUE: dict[str, float] = {
    "s": 1e3,
    "ms": 1,
    "µs": 1e-3,
    "ns": 1e-6,
}


@dataclass(frozen=True)
class ChannelMetadata:
    name: str
    excitation_nm: float | None
    emission_nm: float | None
    exposure_ms: float | None


@dataclass(frozen=True)
class ND2Metadata:
    width: int
    height: int
    pixel_size_um: float
    objective: str
    bit_depth: int | None
    acquired_at: str
    channels: list[ChannelMetadata]


def make_dir(path):
    """Create ``path`` (and parents) if needed."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _normalize_exposure(exposure: float | None, exposure_time_unit: str | None) -> float | None:
    if exposure is None or exposure_time_unit is None:
        return None
    factor = EXPOSURE_UNIT_VALUE.get(exposure_time_unit)
    if factor is None:
        return None
    return exposure * factor


def read_info_file(exp: Experiment) -> pd.DataFrame:
    info = Path(exp.info_file)
    df = pd.read_csv(info)
    df.rename(columns={df.columns[0]: "round_name"}, inplace=True)
    return df


def nd2_metadata(path: str | Path) -> ND2Metadata:
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
    """Read all channels of ND2 file into memory and close the file."""
    path = Path(path)
    with nd2.ND2File(path) as fh:
        sizes = fh.sizes
        if invalid_axes := sorted(set(sizes) - {"C", "Y", "X"}):
            raise ValueError(f"{path.name} holds axes that are not valid: {invalid_axes}")

        image = np.asarray(fh.asarray())

    if "C" not in sizes:
        image = np.expand_dims(image, 0)

    return image


def read_nd2_channel(path, channel):
    """Read one channel of an ND2 file into memory and close the file."""
    with nd2.ND2File(path) as fh:
        return np.asarray(fh.to_dask()[channel, :, :].compute())


def label_paths(labels_path, well):
    folder = Path(labels_path) / well
    return sorted(pth for pth in folder.iterdir() if pth.suffix in (".tiff", ".tif"))


def read_labels(labels_path, well, indices=None):
    """Stack of a well's label images (sorted by file name).

    ``indices`` picks specific images, e.g. [anchor, moving] in notebook 03.
    """
    paths = label_paths(labels_path, well)
    if indices is not None:
        paths = [paths[int(i)] for i in indices]
    return np.array([imread(pth) for pth in paths])


def write_tiff(path: str | Path, array: npt.NDArray[Any], metadata: dict[str, Any] | None = None):
    imwrite(path, array, compression="zlib", ome=metadata is not None, metadata=metadata)
