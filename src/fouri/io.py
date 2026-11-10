"""File helpers shared by the pipeline notebooks."""

import pickle
from collections.abc import Generator
from contextlib import contextmanager
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


def fullest_df_path(exp: Experiment) -> Path:
    return Path(exp.base_dir) / "fullest_df.csv"


def adata_path(exp: Experiment) -> Path:
    return Path(exp.base_dir) / "standard_adata.h5ad"


def adata_sub_path(exp: Experiment) -> Path:
    return Path(exp.base_dir) / "standard_adata_sub.h5ad"


def run_record_path(exp: Experiment) -> Path:
    return Path(exp.base_dir) / "run_record.jsonl"


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


def label_paths(labels_path, well):
    folder = Path(labels_path) / well
    return sorted(pth for pth in folder.iterdir() if pth.suffix in (".tiff", ".tif"))


def label_path(exp: Experiment, well: str, round_index: int) -> Path:
    return Path(exp.segmented) / well / f"labels_{well}_round_{round_index:03d}.tif"


def read_labels(labels_path, well, indices=None):
    """Stack of a well's label images (sorted by file name).

    ``indices`` picks specific images, e.g. [anchor, moving] in notebook 03.
    """
    paths = label_paths(labels_path, well)
    if indices is not None:
        paths = [paths[int(i)] for i in indices]
    return np.array([imread(pth) for pth in paths])


@contextmanager
def written_whole(path: str | Path) -> Generator[Path]:
    """Yield a .partial name to write to.

    The .partial is renamed over path when the block ends without error.
    """
    path = Path(path)
    partial_path = path.with_name(f"{path.name}.partial")
    try:
        yield partial_path
        Path.replace(partial_path, path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


def write_tiff(path: str | Path, array: npt.NDArray[Any], metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    with written_whole(path) as partial:
        imwrite(partial, array, compression="zlib", ome=metadata is not None, metadata=metadata)


def round_table_path(exp: Experiment, well: str) -> Path:
    return Path(exp.output_df) / f"round_table_{well}.csv"


def save_round_table(exp: Experiment, well: str, table: pd.DataFrame) -> None:
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
            f"Well {well}'s round table was made using an old version of the pipeline. "
            "Rerun the preprocessing step first."
        )


def cell_data_path(exp: Experiment, well: str) -> Path:
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
    pth = Path(exp.output_df) / f"tmat_{well}"
    if suffix is not None:
        pth = pth.with_suffix(suffix)
    return pth


def save_transforms(exp: Experiment, well: str, transforms: npt.NDArray[np.float64]) -> Path:
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
