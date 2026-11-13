"""@TODO."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from tifffile import TiffFile

from fouri import io
from fouri.config import Experiment
from fouri.parallel import Failed, check_results, error_in_words
from fouri.report import Stopped, progress, say, say_skipped


@dataclass(frozen=True)
class _Plan:
    # well -> round_index -> signal : path
    # e.g. {'B02': {0: {'DNA': Path(...)}}}
    worklist: dict[str, dict[int, dict[str, Path]]]
    skipped: list[str]
    painted: list[str]
    refused: list[Failed]


def round_masks(images: Mapping[str, Path]) -> dict[str, tuple[npt.NDArray[np.bool], npt.NDArray[np.uint16]]]:
    """@TODO."""
    result = {}
    for signal, path in images.items():
        with TiffFile(path) as tiff:
            shape = tiff.series[0].shape
        result[signal] = (np.zeros(shape, dtype=np.bool), np.zeros(shape, dtype=np.uint16))
    return result


def find_artifacts(exp: Experiment, overwrite: bool = False, wells: Sequence[str] | None = None) -> None:
    """@TODO."""
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
    for well, work in progress(plan.worklist.items(), desc="Overall Progress", unit="well", total=len(plan.worklist)):
        try:
            with progress(work.items(), desc=well, unit="round", leave=False) as round_bar:
                for round_index, images in round_bar:
                    masks = round_masks(images)
                    _write_round(exp, well, round_index, masks)
                    rounds_written += 1
                    wells_written.add(well)
        except Stopped:
            raise
        except Exception as err:
            outcomes.append(Failed(well, error_in_words(err)))
        else:
            outcomes.append(well)

    say(f"Wrote masks for {rounds_written} round(s) in {len(wells_written)} well(s).")
    check_results([*plan.refused, *outcomes], unit="well")


def _write_round(
    exp: Experiment, well: str, round_index: int, masks: dict[str, tuple[npt.NDArray[np.bool], npt.NDArray[np.uint16]]]
) -> None:
    path = io.mask_path(exp, well, round_index)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {"edited": False, "masks": {signal: {"spread": 0, "pad": 0} for signal in masks}}
    with (
        io.written_whole(io.mask_record_path(exp, well, round_index)) as partial,
        open(partial, "w", encoding="utf-8") as fh,
    ):
        json.dump(record, fh, indent=2)

    masks_write = {
        f"{prefix}{signal}": arr
        for signal, arrs in masks.items()
        for prefix, arr in zip(("final_", "outlines_"), arrs, strict=True)
    }
    with io.written_whole(path) as partial, open(partial, "wb") as fh:
        np.savez_compressed(fh, **masks_write, allow_pickle=False)


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
            io.require_raw_files(round_table, well)

            to_write: dict[int, dict[str, Path]] = {}
            well_painted: list[int] = []
            well_skipped = 0
            for rnd_idx, rows in round_table.groupby("round_index"):
                rnd_idx = cast(int, rnd_idx)
                record_path = io.mask_record_path(exp, well, rnd_idx)
                mask_path = io.mask_path(exp, well, rnd_idx)

                edited = (
                    json.loads(record_path.read_text(encoding="utf-8"))["edited"] if record_path.exists() else False
                )
                if not isinstance(edited, bool):
                    raise ValueError(
                        f"Expected a boolean value under 'edited' in file: {record_path.name}. Got '{edited}'."
                    )
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
