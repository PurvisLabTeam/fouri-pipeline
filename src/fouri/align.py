"""Utils for StackReg alignment."""

import functools
import pickle
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from pystackreg import StackReg
from skimage.transform import EuclideanTransform, downscale_local_mean, warp
from tifffile import imread

from fouri.config import Experiment
from fouri.io import label_paths, read_labels
from fouri.parallel import worker_count
from fouri.preprocess import read_round_table
from fouri.report import progress

if TYPE_CHECKING:
    import napari
    from napari.layers import Image


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
    anchors = [j[2] for j in jobs]
    moving = [j[3] for j in jobs]
    transforms = {well: np.array([np.identity(3) for _ in range(len(paths))]) for well, paths in paths_of.items()}

    with ProcessPoolExecutor(n_workers, mp_context=get_context("spawn")) as executor:
        register = functools.partial(register_round, downscale_factor=downscale_factor)
        results = executor.map(register, anchors, moving)
        for job, result in progress(
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


def manual_alignment_images(
    exp: Experiment, well: str, round_name: str
) -> tuple[int, npt.NDArray[Any], npt.NDArray[Any]]:
    table = read_round_table(exp, well)
    matches = table[table["round_name"] == round_name]["round_index"]

    if matches.empty:
        raise ValueError(
            f"Round {round_name} was not found in the round table for well {well}. "
            f"The table has the following rounds: {', '.join(str(name) for name in table['round_name'].unique())}"
        )

    round_index = int(matches.iloc[0])
    if round_index == exp.anchor_round:
        raise ValueError(
            f"{round_name} is the anchor round, which the other rounds are aligned onto, so it has nothing "
            "to correct. Set round_name to the round to correct."
        )

    anchor_labels, round_labels = read_labels(exp.segmented, well, [exp.anchor_round, round_index])

    return round_index, anchor_labels > 0, round_labels > 0


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


def start_hand_alignment(viewer: "napari.Viewer", anchor_layer: "Image", round_layer: "Image") -> None:
    from napari.layers.base._base_constants import ActionType

    clicks = {"size": 40, "canvas_size_limits": (12, 30), "ndim": 2}
    anchor_clicks = viewer.add_points(name="Anchor clicks", face_color="yellow", **clicks)
    round_clicks = viewer.add_points(name="Round clicks", face_color="cyan", affine=round_layer.affine, **clicks)
    viewer.text_overlay.visible = True

    def wait_for(layer: Any, message: str) -> None:
        viewer.layers.selection.active = layer
        layer.mode = "add"
        viewer.text_overlay.text = message

    def after_click(event: Any) -> None:
        if event.action != ActionType.ADDED:
            return
        if event.source is anchor_clicks:
            wait_for(round_clicks, "Now click the same cell in the round to fix (green).")
            return
        fit = fit_hand_alignment(anchor_clicks.data, round_clicks.data)
        if fit is None:
            wait_for(anchor_clicks, "Click a second cell in the anchor round (magenta), far from the first.")
            return
        matrix, mismatch = fit
        round_layer.affine = matrix
        round_clicks.affine = matrix
        wait_for(
            anchor_clicks,
            f"{len(round_clicks.data)} pairs, cells {mismatch:.1f} pixels apart on average. Click another cell "
            "in the anchor round, or run the next cell when the cells line up.",
        )

    anchor_clicks.events.data.connect(after_click)
    round_clicks.events.data.connect(after_click)
    wait_for(anchor_clicks, "Click a cell in the anchor round (magenta).")


def save_manual_transform(
    exp: Experiment, well: str, round_index: int, napari_affine: npt.NDArray[np.float64]
) -> Path:
    manual = np.linalg.inv(_swap_xy(napari_affine))
    transforms = load_transforms(exp, well)
    transforms[round_index] = manual
    return save_transforms(exp, well, transforms)


def _swap_xy(matrix: npt.NDArray[Any]) -> npt.NDArray[Any]:
    swap_order = [1, 0, 2]
    return matrix[np.ix_(swap_order, swap_order)]
