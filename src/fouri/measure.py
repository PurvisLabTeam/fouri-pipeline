"""@TODO."""

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
from fouri.io import cell_data_path, load_transforms, read_labels, read_round_table, require_raw_files, written_whole
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
    """Measure every well without cell data (every well with overwrite); return the CSVs written.

    @TODO.
    """
    if edge_margin < 0:
        raise ValueError(f"Edge margin must be >= 0. Got {edge_margin}")

    chosen = exp.pick_wells(wells)
    worklist = [well for well in chosen if overwrite or not _is_measured(exp, well)]
    if not worklist:
        say("Every well already has cell data, nothing to measure. To overwrite it from scratch, use overwrite=True")
        return []

    say_skipped(done="measured", skipped=[well for well in chosen if well not in worklist])
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

    check_results(results, unit="well")

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
    round_table = read_round_table(exp, well)
    require_raw_files(round_table, well)
    transforms = load_transforms(exp, well)
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

    covered = cv2.erode(covered.astype(np.uint8), np.full((3, 3), 1, dtype=np.uint8), iterations=edge_margin).astype(
        bool
    )

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

    path = cell_data_path(exp, well)
    path.parent.mkdir(parents=True, exist_ok=True)
    with written_whole(path) as partial:
        cells.to_csv(partial, index=False)
    return path, len(lost_labels)


def _mean_per_cell(labels: npt.NDArray[Any], images: npt.NDArray[Any], columns: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(regionprops_table(labels, images, properties=("label", "intensity_mean")))
    names = {f"intensity_mean-{i}": column for i, column in enumerate(columns)}
    return table.rename(columns=names)
