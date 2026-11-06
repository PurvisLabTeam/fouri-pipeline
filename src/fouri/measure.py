"""@TODO."""

import functools
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.measure import regionprops_table
from skimage.segmentation import clear_border
from tifffile import imread

from fouri.align import move_labels
from fouri.config import Experiment
from fouri.io import cell_data_path, load_transforms, read_labels, read_round_table, require_raw_files
from fouri.metadata import feature_prefix
from fouri.parallel import parallel_map
from fouri.report import say, say_skipped
from fouri.rings import make_rings


def measure_wells(
    exp: Experiment,
    ring_width: int = 6,
    ring_gap: int = 1,
    n_workers: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Measure every well without cell data (every well with overwrite); return the CSVs written."""
    worklist = [well for well in exp.well_list if overwrite or not cell_data_path(exp, well).exists()]
    if not worklist:
        say("Every well already has cell data, nothing to measure. To overwrite it from scratch, use overwrite=True")
        return []

    say_skipped(done="measured", skipped=[well for well in exp.well_list if well not in worklist])

    return parallel_map(
        function=functools.partial(measure_well, exp, ring_width=ring_width, ring_gap=ring_gap),
        items=worklist,
        desc="Measuring cells",
        unit="well",
        n_workers=n_workers,
    )


def measure_well(exp: Experiment, well: str, ring_width: int = 6, ring_gap: int = 1) -> Path:
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

    nuc_means = []
    ring_means = []
    col_prefixes = []
    for (index, name), rows in round_table.groupby(["round_index", "round_name"]):
        index = cast(int, index)
        name = str(name)
        round_nuclei = move_labels(cleaned, transforms[index]) if index != exp.anchor_round else cleaned
        round_rings = move_labels(rings, transforms[index]) if index != exp.anchor_round else rings
        images = np.stack([imread(Path(exp.output_images) / well / fname) for fname in rows["raw_file_name"]], axis=-1)

        no_data = (images == 0).all(axis=-1)
        round_nuclei = np.where(no_data, 0, round_nuclei)
        round_rings = np.where(no_data, 0, round_rings)

        prefixes = [feature_prefix(name, signal) for signal in rows["signal"]]
        nuc_means.append(_mean_per_cell(round_nuclei, images, columns=[prfx + "_nuc_mean" for prfx in prefixes]))
        ring_means.append(_mean_per_cell(round_rings, images, columns=[prfx + "_ring_mean" for prfx in prefixes]))
        col_prefixes.extend(prefixes)

    tables = [table, *nuc_means, rings_table, *ring_means]
    cells = functools.reduce(lambda left, right: pd.merge(left, right, on="label", how="inner"), tables)
    total_nuc_protein = pd.DataFrame(
        {f"{prfx}_total_nuc_protein": cells[f"{prfx}_nuc_mean"] * cells["nuc_area"] for prfx in col_prefixes}
    )
    nuc_cyto_ratio = pd.DataFrame(
        {f"{prfx}_nuc_cyto_ratio": cells[f"{prfx}_nuc_mean"] / cells[f"{prfx}_ring_mean"] for prfx in col_prefixes}
    )
    cells = pd.concat([cells, total_nuc_protein, nuc_cyto_ratio], axis=1)
    cells["well"] = well

    path = cell_data_path(exp, well)
    path.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(path, index=False)
    return path


def _mean_per_cell(labels: npt.NDArray[Any], images: npt.NDArray[Any], columns: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(regionprops_table(labels, images, properties=("label", "intensity_mean")))
    names = {f"intensity_mean-{i}": column for i, column in enumerate(columns)}
    return table.rename(columns=names)
