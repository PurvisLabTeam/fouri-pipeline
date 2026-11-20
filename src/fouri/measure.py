"""Per-cell intensity and shape measurement for the Measure cells step.

`measure_wells` measures the chosen wells in spawned processes. `measure_well` measures the cells of
a single well and writes the well's cell data file.

Notes
-----
The cell data file has one row per cell and no index column. The columns, in order:

- `label`: the nucleus label in the anchor round's label (segmentation mask) image.
- `nuc_area`, `centroid-0`, `centroid-1`, `orientation`, `axis_major_length`, `axis_minor_length`,
  and `bbox-0` to `bbox-3`: the nucleus area in pixels and the shape properties from
  `skimage.measure.regionprops_table`, in the anchor round's pixel coordinates.
- `<prefix>_nuc_mean`: the mean intensity over the nucleus.
- `ring_area`: the cytoplasmic ring area in pixels.
- `<prefix>_ring_mean`: the mean intensity over the ring.
- `<prefix>_covered`: True when the channel's artifact mask covers a pixel of the nucleus or ring.
- `<prefix>_total_nuc_protein`: `<prefix>_nuc_mean` times `nuc_area`.
- `<prefix>_nuc_cyto_ratio`: `<prefix>_nuc_mean` divided by `<prefix>_ring_mean`.
- `well`: the well, zero-padded (e.g., `B02`).

`<prefix>` is the feature prefix for a signal in a round, from `feature_prefix` (e.g., `R0_pRB`).
Each `<prefix>` group has one column per channel of each round, in round order, then in the round
table's row order.
"""

import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.measure import regionprops_table
from skimage.segmentation import clear_border
from tifffile import imread

from fouri.align import move_to_anchor, move_to_round
from fouri.artifacts import final_masks
from fouri.config import Experiment
from fouri.io import cell_data_path, load_transforms, read_labels, read_round_table, written_whole
from fouri.metadata import feature_prefix
from fouri.parallel import Dropped, Failed, check_results, parallel_map
from fouri.report import say, say_skipped
from fouri.rings import make_rings


def measure_wells(
    exp: Experiment,
    ring_width: int = 6,
    ring_gap: int = 1,
    edge_margin: int = 1,
    n_workers: int | None = None,
    overwrite: bool = False,
    wells: Sequence[str] | None = None,
) -> list[Path]:
    """Measure the cells of the chosen wells and write a cell data file for each well.

    A well whose cell data file has a `<prefix>_nuc_mean` column for every channel of the well's
    round table is skipped unless `overwrite` is True. A well whose cell data file does not contain
    all of these columns is measured again for every round, with a message. Reports when no well is
    left to measure.

    Each well is measured by `measure_well` in a spawned process (`parallel_map`). The number of
    cells dropped in each well is reported when above 0. A well that raises an error is recorded as
    failed. The remaining wells are measured. After the last well, `check_results` raises
    `StepFailed` naming the failed wells, or `Stopped` when the step is stopped before every well's
    job is submitted.

    With `overwrite` False, a well with a cell data file and without a round table is refused,
    recorded as failed, and excluded from the work. The other wells are measured.

    `Experiment.pick_wells` raises `ValueError` for a well not in `well_list`.

    Parameters
    ----------
    exp : Experiment
        The Experiment that holds the wells.
    ring_width : int, default=6
        The reach of each cytoplasmic ring from the nucleus edge, in pixels, passed to `make_rings`.
        Must be 1 or more.
    ring_gap : int, default=1
        The distance from a nucleus, in pixels, within which pixels are excluded from every ring,
        passed to `make_rings`. Must be 0 or more and less than `ring_width`.
    edge_margin : int, default=1
        The margin, in pixels, around pixels without image data within which a cell is dropped. Must
        be 0 or more.
    n_workers : int | None, default=None
        The number of worker processes, capped at the number of wells. None for `worker_count`.
    overwrite : bool, default=False
        If True, measure every chosen well, including wells already measured.
    wells : Sequence[str] | None, default=None
        The wells to measure. None for every well in `well_list`.

    Returns
    -------
    list[Path]
        The paths of the cell data files written, sorted by plate row, then by column. Empty if no
        well is left to measure.

    Raises
    ------
    ValueError
        If `ring_width` is below 1, `ring_gap` is below 0, `ring_gap` is not below `ring_width`, or
        `edge_margin` is below 0. The check runs before any well is measured.
    """
    if ring_width < 1:
        raise ValueError(f"Ring width must be >= 1. Got {ring_width}")
    if ring_gap < 0:
        raise ValueError(f"Ring gap must be >= 0. Got {ring_gap}")
    if ring_gap >= ring_width:
        raise ValueError(f"Ring gap must be less than ring width. Got ring_gap={ring_gap}, ring_width={ring_width}")
    if edge_margin < 0:
        raise ValueError(f"Edge margin must be >= 0. Got {edge_margin}")

    chosen = exp.pick_wells(wells)
    worklist = []
    refused: list[Failed] = []
    for well in chosen:
        try:
            if overwrite or not _is_measured(exp, well):
                worklist.append(well)
        except FileNotFoundError as error:
            refused.append(Failed(well, str(error)))
    if not worklist:
        if refused:
            check_results(refused, unit="well")
        say("Every well already has cell data, nothing to measure. To overwrite it from scratch, use overwrite=True")
        return []

    refused_wells = {failure.item for failure in refused}
    say_skipped(
        done="measured", skipped=[well for well in chosen if well not in worklist and well not in refused_wells]
    )
    remeasured = [well for well in worklist if not overwrite and cell_data_path(exp, well).exists()]
    for well in remeasured:
        say(f"{well}: cell data has no columns for some rounds, measuring again.")

    results = parallel_map(
        function=functools.partial(
            measure_well, exp, ring_width=ring_width, ring_gap=ring_gap, edge_margin=edge_margin
        ),
        items=worklist,
        desc="Measuring cells",
        unit="well",
        n_workers=n_workers,
    )
    paths = []
    for result, well in zip(results, worklist, strict=True):
        if isinstance(result, (Failed, Dropped)):
            continue
        path, n_cells_lost = result
        paths.append(path)
        if n_cells_lost > 0:
            say(f"{well}: {n_cells_lost} cell(s) dropped.")

    check_results([*refused, *results], unit="well")

    return paths


def _is_measured(exp: Experiment, well: str) -> bool:
    cell_data_csv = cell_data_path(exp, well)
    if not cell_data_csv.exists():
        return False
    columns = pd.read_csv(cell_data_csv, nrows=0).columns
    round_table = read_round_table(exp, well)
    expected_cols = {
        feature_prefix(round_name, signal) + "_nuc_mean"
        for round_name, signal in zip(round_table["round_name"], round_table["signal"], strict=True)
    }
    return expected_cols.issubset(columns)


def measure_well(
    exp: Experiment, well: str, ring_width: int = 6, ring_gap: int = 1, edge_margin: int = 1
) -> tuple[Path, int]:
    """Measure the cells of a single well and write the well's cell data file.

    Cells are the nuclei of the anchor round's label (segmentation mask) image, without the nuclei
    that touch the image border (`clear_border`). Cytoplasmic rings around the nuclei are built by
    `make_rings`. For each round, the nuclei and rings are moved into the round's frame with
    `move_to_round`. Pixels without image data (0 in every channel of the round) are removed from
    the nuclei and rings. The mean intensity of each channel is measured over each nucleus and each
    ring in the round's converted images.

    A cell is dropped when a pixel of its nucleus or ring, in the anchor round's frame, lies on or
    within `edge_margin` pixels (chessboard distance) of a pixel without image data in any round.
    The outermost rows and columns of each round's images, and the area outside the images, count as
    pixels without image data. A cell without a ring is dropped.

    For each channel, `<prefix>_covered` is True when the channel's artifact mask, read with
    `final_masks`, shares a pixel with the cell's nucleus or ring in the round's frame. The cover
    columns of a round without an artifact mask file are False. Build cell table removes every cell
    with a True cover column (`fouri.artifacts.remove_covered`).

    Writes the cell data file named by `cell_data_path` atomically (`written_whole`). The columns
    are listed in the module's `Notes`. `read_round_table`, `load_transforms`, and `read_labels`
    raise `FileNotFoundError` if the round table, the transforms file, or the anchor round's label
    file does not exist. `final_masks` raises `ValueError` if an artifact mask file does not contain
    a mask for a channel of the round.

    Parameters
    ----------
    exp : Experiment
        The Experiment that holds the well.
    well : str
        The well, zero-padded (e.g., `B02`).
    ring_width : int, default=6
        The reach of each cytoplasmic ring from the nucleus edge, in pixels, passed to `make_rings`.
    ring_gap : int, default=1
        The distance from a nucleus, in pixels, within which pixels are excluded from every ring,
        passed to `make_rings`.
    edge_margin : int, default=1
        The margin, in pixels, around pixels without image data within which a cell is dropped.

    Returns
    -------
    path : Path
        The path of the cell data file written.
    n_dropped : int
        The number of nuclei in the anchor round's label image without a row in the file.

    Raises
    ------
    ValueError
        If the transforms file has fewer rounds than the round table, the anchor round's matrix is
        not the identity matrix, a round is not aligned yet (the message names the rounds), an
        artifact mask's shape differs from the shape of the round's images (the message names the
        round), or no cell is left.

    Notes
    -----
    The ring settings and `edge_margin` are not checked. `measure_wells` checks the three settings.
    """
    round_table = read_round_table(exp, well)
    transforms = load_transforms(exp, well)
    if (tr := transforms.shape[0]) < (ta := round_table["round_index"].nunique()):
        raise ValueError(
            f"Not all rounds are aligned: found alignment for {tr} rounds but the round table has {ta} rounds."
        )
    if not np.allclose(transforms[exp.anchor_round], np.eye(3, 3)):
        raise ValueError(
            f"Transforms were saved with a different anchor round than round {exp.anchor_round}. "
            "Align this well again with overwrite. WARNING: this also replaces manually aligned rounds."
        )

    unaligned = set(np.flatnonzero(np.isnan(transforms).any(axis=(1, 2))).tolist())
    unaligned_round_names = [
        str(row.round_name)
        for row in round_table[round_table["is_align_channel"]].sort_values(by="round_index").itertuples()
        if row.round_index in unaligned
    ]
    if unaligned_round_names:
        raise ValueError(
            f"Well {well}: the following rounds have not been aligned yet: {', '.join(unaligned_round_names)}. "
            "Align these rounds before measuring."
        )
    labels = read_labels(exp.segmented, well, [exp.anchor_round])[0]
    unique_labels = np.unique(labels)
    starting_cells = len(unique_labels[unique_labels != 0])

    cleaned = clear_border(labels)
    rings = make_rings(cleaned, ring_width, ring_gap)
    table = pd.DataFrame(
        regionprops_table(
            cleaned,
            properties=(
                "label",
                "area",
                "centroid",
                "orientation",
                "axis_major_length",
                "axis_minor_length",
                "bbox",
            ),
        )
    )
    table = table.rename(columns={"area": "nuc_area"})

    rings_table = pd.DataFrame(regionprops_table(rings, properties=("label", "area")))
    rings_table = rings_table.rename(columns={"area": "ring_area"})

    covered = np.full(labels.shape, True)

    nuc_means = []
    ring_means = []
    col_prefixes = []
    covers = []
    for (index, name), rows in round_table.groupby(["round_index", "round_name"]):
        index = cast(int, index)
        name = str(name)
        transform = transforms[index]
        round_nuclei = move_to_round(cleaned, transform) if index != exp.anchor_round else cleaned
        round_rings = move_to_round(rings, transform) if index != exp.anchor_round else rings
        images = np.stack([imread(Path(exp.output_images) / well / fname) for fname in rows["raw_file_name"]], axis=-1)

        no_data = (images == 0).all(axis=-1)
        has_data = ~no_data
        has_data[[0, -1], :] = False
        has_data[:, [0, -1]] = False
        has_data_in_anchor = move_to_anchor(has_data, transform)
        covered &= has_data_in_anchor  # pixels are true only where this round and every one before it had data

        round_nuclei = np.where(no_data, 0, round_nuclei)
        round_rings = np.where(no_data, 0, round_rings)

        signals = list(rows["signal"])
        prefixes = [feature_prefix(name, signal) for signal in signals]
        masks = final_masks(exp, well, index, signals)
        cover = table["label"].copy().to_frame()
        cover[[f"{prefix}_covered" for prefix in prefixes]] = False

        if masks is not None:
            # mark cells covered by a mask in any (round, channel)
            for signal, prefix in zip(signals, prefixes, strict=True):
                mask = masks[signal]
                mask_shape = mask.shape
                image_shape = images.shape[:2]
                if mask_shape != image_shape:
                    raise ValueError(
                        f"Well {well}: the artifact masks of round {index} are {mask_shape[0]} x {mask_shape[1]}, "
                        f"its images are {image_shape[0]} x {image_shape[1]}. Run Find artifacts on this well again "
                        "with Overwrite on."
                    )
                under_artifact = set(np.unique(round_nuclei[mask & (round_nuclei != 0)]))
                under_artifact |= set(np.unique(round_rings[mask & (round_rings != 0)]))
                cover[f"{prefix}_covered"] = cover["label"].isin(under_artifact)

        covers.append(cover)

        nuc_means.append(_mean_per_cell(round_nuclei, images, columns=[prfx + "_nuc_mean" for prfx in prefixes]))
        ring_means.append(_mean_per_cell(round_rings, images, columns=[prfx + "_ring_mean" for prfx in prefixes]))
        col_prefixes.extend(prefixes)

    covered = cv2.erode(
        covered.astype(np.uint8),
        np.full((3, 3), 1, dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
        iterations=edge_margin,
    ).astype(bool)

    tables = [table, *nuc_means, rings_table, *ring_means, *covers]
    cells = functools.reduce(lambda left, right: pd.merge(left, right, on="label", how="inner"), tables)
    total_nuc_protein = pd.DataFrame(
        {f"{prfx}_total_nuc_protein": cells[f"{prfx}_nuc_mean"] * cells["nuc_area"] for prfx in col_prefixes}
    )
    nuc_cyto_ratio = pd.DataFrame(
        {f"{prfx}_nuc_cyto_ratio": cells[f"{prfx}_nuc_mean"] / cells[f"{prfx}_ring_mean"] for prfx in col_prefixes}
    )
    cells = pd.concat([cells, total_nuc_protein, nuc_cyto_ratio], axis=1)
    cells["well"] = well

    # drop cells not fully inside every round's image data (with a margin of edge_margin px)
    lost_nuclei = cleaned[~covered]
    lost_rings = rings[~covered]
    lost_labels = np.unique(np.concatenate([lost_nuclei, lost_rings]))
    lost_labels = lost_labels[lost_labels != 0]
    cells = cells[~cells["label"].isin(lost_labels)]

    if cells.empty:
        raise ValueError(
            f"Nothing written for well {well}: no cells left to measure after clearing border cells, cells within "
            f"edge_margin={edge_margin} and cells without a ring (ring_width={ring_width}, ring_gap={ring_gap})."
        )

    path = cell_data_path(exp, well)
    path.parent.mkdir(parents=True, exist_ok=True)
    with written_whole(path) as partial:
        cells.to_csv(partial, index=False)
    return path, starting_cells - len(cells)


def _mean_per_cell(labels: npt.NDArray[Any], images: npt.NDArray[Any], columns: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(regionprops_table(labels, images, properties=("label", "intensity_mean")))
    names = {f"intensity_mean-{i}": column for i, column in enumerate(columns)}
    return table.rename(columns=names)
