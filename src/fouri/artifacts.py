"""Artifact masks: computation, growth, reading, and the removal of covered cells.

`find_artifacts` is the Find artifacts step. `round_masks` computes the masks for a single
(round, well). `excess_over_background` and `grow_artifacts` compute the final mask for a channel
from the painted artifacts. `final_masks` reads the final masks for a single (round, well).
`is_mask_edited` reads whether the masks of a round were saved from the mask editor.
`remove_covered` removes the cells covered by an artifact mask from a cell table.

Notes
-----
Each (round, well) has an artifact mask file (`fouri.io.mask_path`) and a mask record JSON file
(`fouri.io.mask_record_path`). The mask file holds two arrays for each channel of the round, in the
shape of the round's images:

- `final_<signal>`: the final mask, true at each masked pixel.
- `outlines_<signal>`: the painted artifacts, each a distinct number, 0 for none.

The mask record holds `edited` and, under `masks`, a dict with `spread` and `pad` keys for each
signal (`grow_artifacts`). `edited` is true for masks saved from the mask editor
(`fouri.viewers.artifacts.ArtifactViewer`). The mask editor also saves the `spread` and `pad` of
each artifact under `artifacts`, keyed by artifact number:

    {"edited": true,
     "masks": {"DNA": {"spread": 2, "pad": 5, "artifacts": {"1": {"spread": 3, "pad": 5}}}}}
"""

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.ndimage as ndi
from tifffile import TiffFile

from fouri import io
from fouri.config import Experiment
from fouri.io import write_npz
from fouri.parallel import Failed, check_results, error_in_words
from fouri.report import Stopped, progress, say, say_skipped
from fouri.stats import robust_location_scale

BLOCK_SIZE: Final[int] = 64


@dataclass(frozen=True)
class _Plan:
    # well -> round_index -> signal : path
    # e.g. {'B02': {0: {'DNA': Path(...)}}}
    worklist: dict[str, dict[int, dict[str, Path]]]
    skipped: list[str]
    painted: list[str]
    refused: list[Failed]


def excess_over_background(image: npt.NDArray[Any]) -> tuple[npt.NDArray[np.float32], float]:
    """Compute an image's excess over the local background and the robust SD of the excess.

    The image is scaled linearly from the image's 1st and 99th percentiles to 0 and 1. Scaled values
    outside 0 to 1 are clipped. The local background is the 25th percentile of each `BLOCK_SIZE` by
    `BLOCK_SIZE` block of the scaled image, median-filtered over 5 by 5 blocks. Pixels in the
    partial blocks at the bottom and right edges take the background of the nearest full block. The
    excess is the scaled image minus the local background.

    Parameters
    ----------
    image : npt.NDArray[Any]
        A single-channel image, shaped (height, width). Must be at least `BLOCK_SIZE` pixels along
        each axis.

    Returns
    -------
    excess : npt.NDArray[np.float32]
        The excess over the local background, in the shape of `image`.
    noise : float
        The robust SD of `excess`, the median absolute deviation scaled by
        `fouri.stats.MAD_TO_SD`. At least 1e-6.
    """
    scaled = image.astype(np.float32)
    percentiles = np.percentile(scaled, (1, 99))
    low, high = map(float, percentiles)
    value_range = max(high - low, 1e-6)
    scaled = (scaled - low) / value_range
    scaled = np.clip(scaled, 0, 1)

    block_rows = scaled.shape[0] // BLOCK_SIZE
    block_cols = scaled.shape[1] // BLOCK_SIZE

    rows_trimmed = block_rows * BLOCK_SIZE
    cols_trimmed = block_cols * BLOCK_SIZE

    row_pad = image.shape[0] - rows_trimmed
    col_pad = image.shape[1] - cols_trimmed

    trimmed = scaled[:rows_trimmed, :cols_trimmed]
    trimmed = trimmed.reshape((block_rows, BLOCK_SIZE, block_cols, BLOCK_SIZE))
    trimmed = np.transpose(trimmed, axes=(0, 2, 1, 3))
    trimmed = trimmed.reshape((block_rows, block_cols, -1))

    perc_25 = np.percentile(trimmed, 25, axis=-1)
    filtered = ndi.median_filter(perc_25, size=(5, 5))
    background = np.repeat(filtered, repeats=BLOCK_SIZE, axis=0)
    background = np.repeat(background, repeats=BLOCK_SIZE, axis=1)
    background = np.pad(background, pad_width=((0, row_pad), (0, col_pad)), mode="edge")
    background = background.astype(np.float32)

    excess = np.asarray(scaled - background, dtype=np.float32)
    _, excess_robust_sd = robust_location_scale(excess)
    excess_robust_sd = max(excess_robust_sd, 1e-6)

    return excess, excess_robust_sd


def grow_artifacts(
    outlines: npt.NDArray[np.uint16],
    excess: npt.NDArray[np.float32],
    noise: float,
    values: Mapping[int, tuple[int, int]],
    *,
    min_size: int = 25,
) -> npt.NDArray[np.bool]:
    """Grow and pad each painted artifact of a channel and return the union as the final mask.

    Artifacts smaller than `min_size` pixels are left out. An artifact with a spread above 0 grows
    into each bright region that overlaps the artifact or is 4-adjacent to the artifact. The bright
    regions are the 4-connected regions (pixels sharing an edge excluding pixels on a diagonal) of
    pixels with an `excess > (7 - spread) * noise`. The grown artifact is dilated `pad` times with a
    4-connected structuring element.

    Parameters
    ----------
    outlines : npt.NDArray[np.uint16]
        The painted artifacts of a single channel, each a distinct number, 0 for none.
    excess : npt.NDArray[np.float32]
        The excess of the channel's image over the local background, as returned by
        `excess_over_background`. Must have the shape of `outlines`.
    noise : float
        The robust SD of `excess`, as returned by `excess_over_background`.
    values : Mapping[int, tuple[int, int]]
        The `(spread, pad)` tuple of each artifact, keyed by artifact number. Must have an entry for
        every artifact of at least `min_size` pixels.
    min_size : int, default=25
        The minimum artifact size in pixels.

    Returns
    -------
    npt.NDArray[np.bool]
        The final mask, true at each pixel of a grown and padded artifact, in the shape of
        `outlines`.

    Notes
    -----
    `outlines` is not modified.
    """
    output = np.zeros(outlines.shape, dtype=np.bool)
    boxes = ndi.find_objects(outlines)
    artifacts = {
        s: (box, *values[s])
        for s, box in enumerate(boxes, start=1)
        if box is not None and (outlines[box] == s).sum() >= min_size
    }
    spreads = {spread for _, spread, _ in artifacts.values() if spread > 0}

    regions_by_spread = {}
    for spread in spreads:
        bright = excess > (7 - spread) * noise
        _, regions, stats, _ = cv2.connectedComponentsWithStats(
            bright.view(np.uint8), ltype=cv2.CV_32S, connectivity=4
        )
        regions_by_spread[spread] = (regions, stats)

    for number, (box, spread, pad) in artifacts.items():
        grown = box
        regions_reached = np.empty(0, dtype=np.int32)
        if spread > 0:
            widened_box = _widen(box, 1, outlines.shape)
            artifact_pixels = outlines[widened_box] == number
            dilated = ndi.binary_dilation(artifact_pixels)
            regions, stats = regions_by_spread[spread]
            region_crop = regions[widened_box]
            numbers = np.unique(region_crop[dilated])
            regions_reached = cast(npt.NDArray[np.int32], numbers[numbers != 0])
            if regions_reached.size != 0:
                region_stats = stats[regions_reached]
                row_start = min(box[0].start, int(region_stats[:, cv2.CC_STAT_TOP].min()))
                row_stop = max(
                    box[0].stop,
                    int((region_stats[:, cv2.CC_STAT_TOP] + region_stats[:, cv2.CC_STAT_HEIGHT]).max()),
                )
                col_start = min(box[1].start, int(region_stats[:, cv2.CC_STAT_LEFT].min()))
                col_stop = max(
                    box[1].stop,
                    int((region_stats[:, cv2.CC_STAT_LEFT] + region_stats[:, cv2.CC_STAT_WIDTH]).max()),
                )
                grown = (slice(row_start, row_stop), slice(col_start, col_stop))

        final_box = _widen(grown, margin=pad, shape=outlines.shape)
        cropped_final = outlines[final_box]
        shape = cropped_final == number
        if regions_reached.size != 0:
            shape = shape | np.isin(regions[final_box], regions_reached)

        if pad > 0:
            shape = ndi.binary_dilation(shape, iterations=pad)
        output[final_box] |= shape

    return output


def _widen(box: tuple[slice, ...], margin: int, shape: tuple[int, ...]) -> tuple[slice, slice]:
    row_start = max(box[0].start - margin, 0)
    row_stop = min(box[0].stop + margin, shape[0])

    col_start = max(box[1].start - margin, 0)
    col_stop = min(box[1].stop + margin, shape[1])

    return (slice(row_start, row_stop), slice(col_start, col_stop))


def final_masks(
    exp: Experiment, well: str, round_index: int, signals: Sequence[str]
) -> dict[str, npt.NDArray[np.bool]] | None:
    """Read the final artifact mask of each signal for a single (round, well), or None when absent.

    Parameters
    ----------
    exp : Experiment
        The Experiment that holds the file.
    well : str
        The well, zero-padded (e.g., `B02`).
    round_index : int
        The 0-based round index.
    signals : Sequence[str]
        The signals whose masks to read.

    Returns
    -------
    dict[str, npt.NDArray[np.bool]] | None
        The final mask of each signal, keyed by signal. None if the artifact mask file does not
        exist.

    Raises
    ------
    ValueError
        If the artifact mask file does not contain `final_<signal>` for a signal in `signals`. The
        message names the well, the round, and the signal.

    Notes
    -----
    The file is the one `fouri.io.mask_path` names. The shape of each mask is not checked.
    """
    mask_path = io.mask_path(exp, well, round_index)
    if not mask_path.exists():
        return None

    masks: dict[str, npt.NDArray[np.bool]] = {}
    with np.load(mask_path) as fh:
        for signal in signals:
            key = f"final_{signal}"
            if key not in fh:
                raise ValueError(
                    f"Well {well}: the artifact masks of round {round_index} have no mask for {signal}. "
                    "Run Find artifacts on this well again with Overwrite."
                )
            arr = fh[key]
            masks[signal] = arr
    return masks


def remove_covered(table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove the cells covered by an artifact mask in any channel and drop the cover columns.

    A cell is removed when the final artifact mask of any channel, in any round, shares at least one
    pixel with the cell's nucleus or ring, moved into that round's frame. Measure cells records this
    for each channel in a cover column, a column whose name ends in `_covered`
    (`fouri.measure.measure_well`). A round without an artifact mask file covers no cell. An empty
    cover value counts as not covered.

    Parameters
    ----------
    table : pd.DataFrame
        The cell table, with a cover column for each channel of each round (e.g., `R0_DNA_covered`).

    Returns
    -------
    table : pd.DataFrame
        The cells not covered, without the cover columns.
    count : int
        The number of cells removed.

    Notes
    -----
    `table` is not modified.
    """
    cover_cols = table[[col for col in table.columns if col.endswith("_covered")]].fillna(False)
    mask = (cover_cols).any(axis=1)
    count = int(mask.sum())
    table = table[~mask]
    table = table.drop(columns=cover_cols.columns)
    return table, count


def round_masks(images: Mapping[str, Path]) -> dict[str, tuple[npt.NDArray[np.bool], npt.NDArray[np.uint16]]]:
    """Compute the final mask and the artifact outlines of each channel for a single (round, well).

    The final masks are false and the outlines 0 at every pixel. Each array has the shape of the
    channel's image, read from the TIFF file's header.

    Parameters
    ----------
    images : Mapping[str, Path]
        The path to the converted image of each channel, keyed by signal.

    Returns
    -------
    dict[str, tuple[npt.NDArray[np.bool], npt.NDArray[np.uint16]]]
        The `(final, outlines)` tuple of each channel, keyed by signal, in the layout of the mask
        file's `final_<signal>` and `outlines_<signal>`.
    """
    result = {}
    for signal, path in images.items():
        with TiffFile(path) as tiff:
            shape = tiff.series[0].shape
        result[signal] = (np.zeros(shape, dtype=np.bool), np.zeros(shape, dtype=np.uint16))
    return result


def find_artifacts(exp: Experiment, overwrite: bool = False, wells: Sequence[str] | None = None) -> None:
    """Compute and write the artifact masks for each round of the chosen wells.

    Computes the masks of a round with `round_masks`. Writes the mask record, with `edited` false
    and a `spread` and `pad` of 0 for each signal, then the mask file. A round with a mask file is
    skipped. A round whose mask record has `edited` true is never written, even with `overwrite`.
    The mask record is read again just before each write. A round with `edited` true by then is
    not written.

    A well is refused on any error while reading the well's round table or mask records (e.g., a
    missing round table file, or a mask record that is not valid JSON or does not have an `edited`
    of true or false). A well is failed on an error while computing or writing the well's masks.
    The rounds already written for a failed well are kept.

    Reports the hand-edited rounds kept, the wells skipped, the number of rounds and wells written,
    and the rounds found hand-edited just before a write. With no round to write and no well
    refused, reports so and returns. `fouri.parallel.check_results` raises `StepFailed` naming each
    refused and failed well, after the other wells are processed.

    Parameters
    ----------
    exp : Experiment
        The Experiment that holds the files.
    overwrite : bool, default=False
        If True, rounds with a mask file are written again.
    wells : Sequence[str] | None, default=None
        The wells to process. Must be in `well_list`. None for every well in `well_list`.

    Raises
    ------
    Stopped
        If the step is stopped from the app. The rounds already written are kept.
    """
    plan = _plan_wells(exp, overwrite, wells)
    if plan.painted:
        say(f"The following masks were hand-drawn so they are kept even with overwrite: {', '.join(plan.painted)}")
    if not plan.worklist:
        if plan.refused:
            check_results(plan.refused, unit="well")
        else:
            say("Nothing to do: all rounds already have masks. To remake them, turn overwrite on.")
            return
    say_skipped(done="masked", skipped=plan.skipped)

    outcomes: list[str | Failed] = []
    rounds_written = 0
    wells_written: set[str] = set()
    not_written: defaultdict[str, list[int]] = defaultdict(list)
    for well, work in progress(plan.worklist.items(), desc="Overall Progress", unit="well", total=len(plan.worklist)):
        try:
            with progress(work.items(), desc=well, unit="round", leave=False) as round_bar:
                for round_index, images in round_bar:
                    masks = round_masks(images)
                    written = _write_round(exp, well, round_index, masks)
                    if not written:
                        not_written[well].append(round_index)
                        continue
                    rounds_written += 1
                    wells_written.add(well)
        except Stopped:
            raise
        except Exception as err:
            outcomes.append(Failed(well, error_in_words(err)))
        else:
            outcomes.append(well)

    say(f"Wrote masks for {rounds_written} round(s) in {len(wells_written)} well(s).")
    if not_written:
        say(
            "The following masks were edited by hand while this step ran, so they were not overwritten: "
            f"{', '.join(f'{w} (round(s): {", ".join(map(str, rounds))})' for w, rounds in not_written.items())}"
        )
    check_results([*plan.refused, *outcomes], unit="well")


def _write_round(
    exp: Experiment, well: str, round_index: int, masks: dict[str, tuple[npt.NDArray[np.bool], npt.NDArray[np.uint16]]]
) -> bool:
    record_path = io.mask_record_path(exp, well, round_index)
    edited = is_mask_edited(record_path)
    if edited:
        return False

    path = io.mask_path(exp, well, round_index)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {"edited": False, "masks": {signal: {"spread": 0, "pad": 0} for signal in masks}}
    with (
        io.written_whole(record_path) as partial,
        open(partial, "w", encoding="utf-8") as fh,
    ):
        json.dump(record, fh, indent=2)

    masks_write = {
        f"{prefix}{signal}": arr
        for signal, arrs in masks.items()
        for prefix, arr in zip(("final_", "outlines_"), arrs, strict=True)
    }
    write_npz(path, masks_write)

    return True


def is_mask_edited(mask_record_path: Path) -> bool:
    """Return whether a mask record has `edited` true.

    A file that is not valid JSON raises `json.JSONDecodeError`. A file that does not contain an
    `edited` key raises `KeyError`.

    Parameters
    ----------
    mask_record_path : Path
        The path to the mask record JSON file.

    Returns
    -------
    bool
        The value of `edited`. False if the file does not exist.

    Raises
    ------
    ValueError
        If `edited` is not true or false. The message names the file.
    """
    edited = json.loads(mask_record_path.read_text(encoding="utf-8"))["edited"] if mask_record_path.exists() else False
    if not isinstance(edited, bool):
        raise ValueError(f"Expected a boolean value under 'edited' in file: {mask_record_path.name}. Got '{edited}'.")
    return edited


def _plan_wells(exp: Experiment, overwrite: bool = False, wells: Sequence[str] | None = None) -> _Plan:
    wells = exp.pick_wells(wells)

    # worklist = well -> round_index -> signal : path
    # e.g. {'B02': {0: {'DNA': Path(...)}}}
    worklist: dict[str, dict[int, dict[str, Path]]] = {}
    skipped: list[str] = []
    painted: list[str] = []
    refused: list[Failed] = []
    for well in wells:
        try:
            round_table = io.read_round_table(exp, well)

            to_write: dict[int, dict[str, Path]] = {}
            well_painted: list[int] = []
            well_skipped = 0
            for rnd_idx, rows in round_table.groupby("round_index"):
                rnd_idx = cast(int, rnd_idx)
                record_path = io.mask_record_path(exp, well, rnd_idx)
                mask_path = io.mask_path(exp, well, rnd_idx)

                edited = is_mask_edited(record_path)

                # e.g. {'DNA': Path(...)}
                out_paths = {
                    str(row.signal): Path(exp.output_images) / well / str(row.raw_file_name)
                    for row in rows.itertuples()
                }

                if edited:
                    well_painted.append(rnd_idx)
                elif mask_path.exists() and not overwrite:
                    well_skipped += 1
                else:
                    to_write[rnd_idx] = out_paths

        except Exception as error:  # note and move on if a well fails
            refused.append(Failed(well, error_in_words(error)))

        else:
            if to_write:
                worklist[well] = to_write
            if well_painted:
                painted_msg = (
                    f"{well} (round{'s' if len(well_painted) > 1 else ''} {', '.join(map(str, well_painted))})"
                )
                painted.append(painted_msg)
            if well_skipped > 0:
                skipped.append(f"{well} ({well_skipped} of {len(to_write) + len(well_painted) + well_skipped} rounds)")

    return _Plan(worklist=worklist, skipped=skipped, painted=painted, refused=refused)
