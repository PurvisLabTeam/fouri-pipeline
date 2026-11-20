"""Utils for StackReg alignment."""

import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from pystackreg import StackReg
from skimage.transform import EuclideanTransform, downscale_local_mean
from tifffile import imread

from fouri.config import Experiment
from fouri.io import label_path, label_paths, load_transforms, read_round_table, save_transforms
from fouri.parallel import Dropped, Failed, check_results, parallel_map
from fouri.report import say, say_skipped


def register_round(
    anchor_path: str | Path, moving_path: str | Path, downscale_factor: int = 1
) -> npt.NDArray[np.float64]:
    """Rigid transform to align the labels of one round to the anchors.

    Written as a 3x3 affine transform matrix.
    """
    anchor = imread(anchor_path) > 0
    moving = imread(moving_path) > 0
    anchor = downscale_local_mean(anchor, (downscale_factor, downscale_factor))
    moving = downscale_local_mean(moving, (downscale_factor, downscale_factor))

    aligner = StackReg(StackReg.RIGID_BODY)
    matrix = aligner.register(anchor, moving)

    return rescale_transform(matrix, downscale_factor)


def _register_pair(paths: tuple[Path, Path], downscale_factor: int) -> npt.NDArray[np.float64]:
    """Register one (anchor, moving) pair.

    Allows parallel_map to hand each job as a single argument.
    """
    return register_round(*paths, downscale_factor=downscale_factor)


def rescale_transform(matrix: npt.NDArray[np.float64], factor: float) -> npt.NDArray[np.float64]:
    """Rescale the transform to ``factor`` times larger.

    Only the shift values (rows 0 and 1 of the last column) change. The rotation
    is the same at any size. Returns a new matrix.
    """
    rescaled = matrix.copy()
    rescaled[..., :2, -1] *= factor
    return rescaled


def move_to_round(labels: npt.NDArray[Any], transform: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
    return cv2.warpAffine(
        labels.astype(np.float32),
        np.linalg.inv(transform)[:2],
        labels.shape[::-1],
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
    ).astype(labels.dtype)


def move_to_anchor(labels: npt.NDArray[Any], transform: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
    return cv2.warpAffine(
        labels.astype(np.float32),
        transform[:2],
        labels.shape[::-1],
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
    ).astype(labels.dtype)


def find_transforms(
    exp: Experiment,
    downscale_factor: int = 2,
    n_workers: int | None = None,
    overwrite: bool = False,
    wells: Sequence[str] | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Align every round without a saved transform (every round with overwrite); return each well's.

    A round counts as aligned when its saved matrix isn't NaN, so rounds aligned by hand are kept
    unless overwrite is set. @TODO.
    """
    anchor_round = exp.anchor_round
    paths_of: dict[str, list[Path]] = {}
    round_names: dict[str, list[str]] = {}  # each well's round names, in round order
    transforms: dict[str, npt.NDArray[np.float64]] = {}
    planned: dict[str, npt.NDArray[np.float64]] = {}
    replaced: list[str] = []
    refused: list[Failed] = []  # wells set up wrongly, each with the first check it fails
    for well in exp.pick_wells(wells):
        labels = label_paths(exp.segmented, well)
        try:
            df = read_round_table(exp, well)
        except FileNotFoundError as error:
            refused.append(Failed(well, str(error)))
            continue

        rounds_df = df["round_index"].unique()
        expected_paths = [label_path(exp, well, index) for index in rounds_df]
        missing = [index for index, path in zip(rounds_df, expected_paths, strict=True) if path not in labels]
        extra = [path for path in labels if path not in expected_paths]
        if missing or extra:
            message = ""
            trailer = ""
            if missing:
                missing_names = [str(df[df["round_index"] == idx]["round_name"].iloc[0]) for idx in missing]
                message += f"Missing label file for round(s): {', '.join(missing_names)}. "
                trailer += "Segment this well again"
            if extra:
                message += (
                    "The following extra files were found for rounds not listed in the rounds table: "
                    f"{', '.join(ex.name for ex in extra)}. "
                )
                trailer = trailer + " and remove the extra file(s)" if trailer != "" else "Remove the extra file(s)"
            message = message + trailer + "."
            refused.append(Failed(well, message))
            continue

        if anchor_round not in rounds_df:
            refused.append(
                Failed(
                    well,
                    f"no label image for the anchor round, round {anchor_round}. Rounds count from 0: 0 is the "
                    "first round in info_exp.csv.",
                )
            )
            continue

        well_transforms = _unaligned(len(rounds_df), anchor_round)
        try:
            saved = load_transforms(exp, well)
        except FileNotFoundError:
            saved = None  # nothing aligned yet
        if saved is not None:
            if overwrite:
                replaced.append(well)
            elif len(saved) > len(rounds_df):
                refused.append(
                    Failed(
                        well,
                        f"{len(saved)} rounds saved, {len(rounds_df)} rounds in the well. Align this well again "
                        "with Overwrite; that also replaces rounds aligned by hand.",
                    )
                )
                continue
            elif anchor_round >= len(saved) or not np.allclose(saved[anchor_round], np.eye(3)):
                refused.append(
                    Failed(
                        well,
                        f"transforms saved with a different anchor round than round {anchor_round}. Align this "
                        "well again with Overwrite; that also replaces rounds aligned by hand.",
                    )
                )
                continue
            else:
                well_transforms[: len(saved)] = saved

        well_planned = _unaligned(len(rounds_df), anchor_round)
        if saved is not None:
            saved_rows = saved[: len(well_planned)]
            well_planned[: len(saved_rows)] = saved_rows

        paths_of[well] = sorted(expected_paths)
        round_names[well] = df.sort_values("round_index")["round_name"].unique().tolist()
        transforms[well] = well_transforms
        planned[well] = well_planned

    jobs = [
        (well, r, labels[anchor_round], label)
        for well, labels in paths_of.items()
        for r, label in enumerate(labels)
        if r != anchor_round and np.isnan(transforms[well][r]).any()
    ]

    if not jobs:
        if refused:
            check_results(refused, unit="well")  # raises StepFailed naming them
        say(
            "Every round is already aligned, nothing to align. To overwrite them from scratch, use "
            "overwrite=True (this also replaces rounds aligned by hand)"
        )
        return transforms

    skipped = []
    for well, labels in paths_of.items():
        n_rounds = len(labels) - 1  # the anchor is never aligned
        n_skipped = n_rounds - sum(job[0] == well for job in jobs)
        if n_skipped > 0:
            skipped.append(f"{well} ({n_skipped} of {n_rounds} rounds)")
    say_skipped(done="aligned", skipped=skipped)
    if replaced:
        say(f"Replacing the saved alignments of {', '.join(replaced)}, including any rounds aligned by hand")

    pairs = [(j[2], j[3]) for j in jobs]
    results = parallel_map(
        function=functools.partial(_register_pair, downscale_factor=downscale_factor),
        items=pairs,
        desc="Aligning images",
        unit="image",
        n_workers=n_workers,
        names=[j[0] for j in jobs],
    )
    outcomes: list[object] = []  # the results, with each failed pair's error naming its round
    computed: dict[str, dict[int, npt.NDArray[np.float64]]] = {}
    for job, result in zip(jobs, results, strict=True):
        well, rnd, _, _ = job
        if isinstance(result, Failed):
            outcomes.append(Failed(result.item, f"round {round_names[well][rnd]}: {result.error}"))
            continue
        outcomes.append(result)
        if isinstance(result, Dropped):
            continue
        computed.setdefault(well, {})[rnd] = result

    hand_saved: dict[str, list[int]] = {}
    for well, rounds in computed.items():
        try:
            current = load_transforms(exp, well)
        except FileNotFoundError:
            current = planned[well]
        written = _unaligned(len(planned[well]), anchor_round)
        current_rows = current[: len(written)]
        written[: len(current_rows)] = current_rows
        written[anchor_round] = np.eye(3)
        for rnd, matrix in rounds.items():
            if np.array_equal(written[rnd], planned[well][rnd], equal_nan=True):
                written[rnd] = matrix
            else:
                hand_saved.setdefault(well, []).append(rnd)
        save_transforms(exp, well, written)
        transforms[well] = written

    if hand_saved:
        say(
            "The following well(s) were aligned by hand while this step ran, so they were not overwritten: "
            f"{', '.join(f'{w} (round(s) {", ".join(str(r) for r in rounds)})' for w, rounds in hand_saved.items())}."
        )

    check_results([*refused, *outcomes], unit="well")
    return transforms


def _unaligned(n_rounds: int, anchor_round: int) -> npt.NDArray[np.float64]:
    """Return a not-yet-aligned well's transforms: NaN for every round, identity for the anchor."""
    transforms = np.full((n_rounds, 3, 3), np.nan)
    transforms[anchor_round] = np.eye(3)
    return transforms


def round_index_of(round_table: pd.DataFrame, round_name: str, anchor_round: int, well: str) -> int:
    matches = round_table[round_table["round_name"] == round_name]["round_index"]

    if matches.empty:
        raise ValueError(
            f"Round {round_name} was not found in the round table for well {well}. "
            "The table has the following rounds: "
            f"{', '.join(str(name) for name in round_table['round_name'].unique())}"
        )

    round_index = int(matches.iloc[0])
    if round_index == anchor_round:
        raise ValueError(
            f"{round_name} is the anchor round, which the other rounds are aligned onto, so it has nothing "
            "to correct. Set round_name to the round to correct."
        )
    return round_index


def fit_hand_alignment(
    anchor_points: npt.NDArray[np.float64], round_points: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], float] | None:
    """Shift and rotation that move the round's clicked cells onto the same cells in the anchor.

    Points are napari (row, column) positions, paired in the order they were clicked. Returns
    the 3x3 matrix for the round's napari layer and the average distance in pixels left between
    the pairs, or None until there are 2 complete pairs.
    """
    n_pairs = min(len(anchor_points), len(round_points))
    if n_pairs < 2:
        return None
    moving, fixed = round_points[:n_pairs], anchor_points[:n_pairs]
    transform = EuclideanTransform.from_estimate(moving, fixed)
    if not transform:
        return None
    mismatch = float(np.mean(transform.residuals(moving, fixed)))
    return transform.params, mismatch


def _swap_xy(matrix: npt.NDArray[Any]) -> npt.NDArray[Any]:
    swap_order = [1, 0, 2]
    return matrix[np.ix_(swap_order, swap_order)]


def save_manual_transform(
    exp: Experiment, well: str, round_index: int, napari_affine: npt.NDArray[np.float64]
) -> Path:
    manual = np.linalg.inv(_swap_xy(napari_affine))
    try:
        transforms = load_transforms(exp, well)
    except FileNotFoundError:
        n_rounds = read_round_table(exp, well)["round_index"].nunique()
        transforms = _unaligned(n_rounds, exp.anchor_round)
    if len(transforms) <= round_index:
        padding = np.full((round_index + 1 - len(transforms), 3, 3), np.nan)
        transforms = np.concatenate([transforms, padding])
    transforms[round_index] = manual
    return save_transforms(exp, well, transforms)


def transform_to_napari(transform: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return saved to on-screen position. @TODO."""
    return _swap_xy(np.linalg.inv(transform))
