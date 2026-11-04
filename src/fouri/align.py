"""@TODO."""

import functools
import pickle
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pystackreg import StackReg
from skimage.transform import downscale_local_mean, warp
from tifffile import imread
from tqdm.auto import tqdm

from fouri.config import Experiment
from fouri.io import label_paths, read_labels
from fouri.parallel import worker_count


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


def rescale_transform(matrix: npt.NDArray[np.float64], factor: float) -> npt.NDArray[np.float64]:
    """Rescale the transform to ``factor`` times larger.

    Only the shift values (rows 0 and 1 of the last column) change. The rotation
    is the same at any size. Returns a new matrix.
    """
    rescaled = matrix.copy()
    rescaled[..., :2, -1] *= factor
    return rescaled


def apply_transforms(matrices: npt.NDArray[Any], images: npt.NDArray) -> npt.NDArray[Any]:
    """Warp each image with the matrix at the same position in ``matrices``.

    For the visual check at the end of notebook 02; returns warped images as
    floats. Works on a single matrix or a stack.
    """
    return np.array(
        [warp(image, matrix, output_shape=image.shape) for image, matrix in zip(images, matrices, strict=True)]
    )


def find_transforms(
    exp: Experiment,
    downscale_factor: int = 1,
    n_workers: int | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Align every round of every well onto the experiment's anchor round and save.

    The anchor round is ``exp.anchor_round``, the one 03 and 06 also use. Saves to
    ``output_df/tmat_{well}.npy``. Each round is aligned in its own
    process, ``n_workers`` at a time. Returns
    ``{well: array of shape (rounds, 3, 3)}`` with the identity at the
    anchor round and saves those same arrays.
    """
    anchor_round = exp.anchor_round
    n_workers = n_workers if n_workers is not None else worker_count()
    paths_of: dict[str, list[Path]] = {}
    mismatched: dict[str, tuple[int, int]] = {}
    for well in exp.well_list:
        labels = label_paths(exp.segmented, well)
        paths_of[well] = labels

        df = pd.read_pickle(Path(exp.output_df) / f"df_{well}.pkl")
        n_rounds_df = df["alignRound"].nunique()

        if len(labels) != n_rounds_df:
            mismatched[well] = (len(labels), n_rounds_df)

    if mismatched:
        message = (
            f"Segmentation images and rounds don't match in {len(mismatched)} well(s), so no transforms were found:\n"
        )
        for well, (n_labels, n_rounds) in mismatched.items():
            message += f"\t{well}: {n_labels} label images in segmented, {n_rounds} rounds listed in df_{well}.pkl\n"
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
    anchors = [j[2] for j in jobs]
    moving = [j[3] for j in jobs]
    transforms = {well: np.array([np.identity(3) for _ in range(len(paths))]) for well, paths in paths_of.items()}

    with ProcessPoolExecutor(n_workers, mp_context=get_context("spawn")) as executor:
        register = functools.partial(register_round, downscale_factor=downscale_factor)
        results = executor.map(register, anchors, moving)
        for job, result in tqdm(
            zip(jobs, results, strict=True),
            total=len(jobs),
            desc="Aligning images",
            unit="image",
        ):
            well, r, _, _ = job
            transforms[well][r] = result

    for well, matrix in transforms.items():
        save_transforms(exp, well, matrix)

    return transforms


def alignment_preview(
    exp: Experiment, well: str, downscale_factor: int = 1
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    before = read_labels(exp.segmented, well)
    before = before[:, ::downscale_factor, ::downscale_factor] > 0

    transforms = load_transforms(exp, well)
    transforms = rescale_transform(transforms, 1 / downscale_factor)
    after = apply_transforms(transforms, before)
    return before[exp.anchor_round], before, after


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


def save_transforms(exp: Experiment, well: str, transforms: npt.NDArray[np.float64]) -> Path:
    save_path = tmat_path(exp, well, ".npy")
    np.save(save_path, transforms)
    return save_path


def tmat_path(exp: Experiment, well: str, suffix: str | None = None) -> Path:
    pth = Path(exp.output_df) / f"tmat_{well}"
    if suffix is not None:
        pth = pth.with_suffix(suffix)
    return pth
