"""Utils for StackReg alignment."""

import functools
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
from fouri.io import label_paths, load_transforms, read_round_table, save_transforms
from fouri.parallel import parallel_map


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


def move_labels(labels: npt.NDArray[Any], transform: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
    return cv2.warpAffine(
        labels.astype(np.float32),
        np.linalg.inv(transform)[:2],
        labels.shape[::-1],
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
    ).astype(labels.dtype)


def find_transforms(
    exp: Experiment,
    downscale_factor: int = 1,
    n_workers: int | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    anchor_round = exp.anchor_round
    paths_of: dict[str, list[Path]] = {}
    mismatched: dict[str, tuple[int, int]] = {}
    for well in exp.well_list:
        labels = label_paths(exp.segmented, well)
        paths_of[well] = labels

        df = read_round_table(exp, well)
        n_rounds_df = df["round_index"].nunique()

        if len(labels) != n_rounds_df:
            mismatched[well] = (len(labels), n_rounds_df)

    if mismatched:
        message = (
            f"Segmentation images and rounds don't match in {len(mismatched)} well(s), so no transforms were found:\n"
        )
        for well, (n_labels, n_rounds) in mismatched.items():
            message += (
                f"\t{well}: {n_labels} label images in segmented, {n_rounds} rounds listed in the rounds table\n"
            )
        message += (
            "Each round needs exactly one label image. Rerun 01 for these wells or remove "
            "extra .tif files from their folders."
        )
        raise ValueError(message)

    too_few = sorted(well for well, labels in paths_of.items() if anchor_round >= len(labels))
    if too_few:
        raise ValueError(
            f'"anchor_round" is {anchor_round} in experiment.json, but {", ".join(too_few)} '
            f"{'has' if len(too_few) == 1 else 'have'} no round {anchor_round}. Rounds count from 0: "
            "0 is the first round in info_exp.csv, 1 the second."
        )

    jobs = [
        (well, r, labels[anchor_round], label)
        for well, labels in paths_of.items()
        for r, label in enumerate(labels)
        if r != anchor_round
    ]
    pairs = [(j[2], j[3]) for j in jobs]
    transforms = {well: np.array([np.identity(3) for _ in range(len(paths))]) for well, paths in paths_of.items()}

    results = parallel_map(
        function=functools.partial(_register_pair, downscale_factor=downscale_factor),
        items=pairs,
        desc="Aligning images",
        unit="image",
        n_workers=n_workers,
    )
    for job, result in zip(jobs, results, strict=True):
        well, rnd, _, _ = job
        transforms[well][rnd] = result

    for well, matrix in transforms.items():
        save_transforms(exp, well, matrix)

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
        transforms = np.full((n_rounds, 3, 3), np.nan)
        transforms[exp.anchor_round] = np.eye(3)
    transforms[round_index] = manual
    return save_transforms(exp, well, transforms)


def transform_to_napari(transform: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return saved to on-screen position. @TODO."""
    return _swap_xy(np.linalg.inv(transform))
