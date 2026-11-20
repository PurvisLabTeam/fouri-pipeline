"""Nucleus segmentation with Cellpose for the Segment nuclei step.

`segment_wells` segments the alignment channel image of each round and writes the label
(segmentation mask) files. `segment_image` segments a 2D image with a progress bar over the image's
tiles. `gpu_name` returns the name of CUDA device 0.

Notes
-----
A label file is a 2D integer image with the shape of the round's alignment channel image. Background
pixels are 0. The pixels of each nucleus share one positive integer label.

The segmentation editor saves over a label file. The first time the editor saves a round, the
label file as Segment nuclei wrote it is kept under the same name in a `cellpose_original`
directory beside the label files (`fouri.io.original_label_path`): the original Cellpose label
file. A round with an original Cellpose label file is hand-edited.
"""

import warnings
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy.typing as npt
from tifffile import imread

from fouri.config import Experiment
from fouri.io import alignment_paths, label_path, original_label_path, write_tiff
from fouri.models import model_path
from fouri.parallel import Failed, check_results, error_in_words
from fouri.report import Stopped, progress, say, say_skipped


def gpu_name() -> str | None:
    """Return the name of CUDA device 0, or None when no CUDA GPU is available to PyTorch."""
    import torch

    gpu = torch.cuda.is_available()
    return torch.cuda.get_device_name(0) if gpu else None


def segment_wells(
    exp: Experiment,
    diameter: float = 30,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    model: str | None = None,
    allow_cpu: bool = False,
    overwrite: bool = False,
    wells: Sequence[str] | None = None,
) -> None:
    """Segment the alignment channel image of each round of the chosen wells with Cellpose.

    Writes the label (segmentation mask) file for each (round, well), as named by `label_path`. A
    round whose label file exists is skipped unless `overwrite` is True. A hand-edited round (a
    round with an original Cellpose label file) is never replaced, even when `overwrite` is True.
    The skipped rounds and the hand-edited rounds are reported separately. Reports and returns
    before loading Cellpose when no round is left to segment.

    The original Cellpose label file is checked again before each label file is written. A round
    hand-edited while the step runs is not written. These rounds are reported after the last well.

    Segmentation runs on a CUDA GPU. With `allow_cpu` True, segmentation runs on the CPU when no
    CUDA GPU is available. A well that raises an error is recorded as failed. Segmentation continues
    with the next well. After the last well, `check_results` raises `StepFailed` naming the failed
    wells. `Stopped` from a progress bar ends the step. The label files already written are kept.

    `Experiment.pick_wells` raises `ValueError` for a well not in `well_list`. `FileNotFoundError`
    is raised before any well is segmented when `model` is not the name of a kept model
    (`model_path`) or a chosen well does not have a round table (`alignment_paths`).

    Parameters
    ----------
    exp : Experiment
        The Experiment that holds the wells.
    diameter : float, default=30
        The expected nucleus diameter in pixels. Cellpose rescales each image by 30 / `diameter`.
    flow_threshold : float, default=0.4
        The flow error threshold. Cellpose keeps masks with a flow error below the threshold.
    cellprob_threshold : float, default=0.0
        The cell probability threshold. Cellpose includes pixels above the threshold in masks.
    min_size : int, default=15
        The minimum mask size in pixels. Cellpose removes smaller masks.
    model : str | None, default=None
        The name of a kept fine-tuned Cellpose model, as listed by `fouri.models.kept_models`. None
        for Cellpose-SAM.
    allow_cpu : bool, default=False
        If True, segment on the CPU when no CUDA GPU is available.
    overwrite : bool, default=False
        If True, segment rounds whose label file exists, hand-edited rounds excepted.
    wells : Sequence[str] | None, default=None
        The wells to segment. None for every well in `well_list`.

    Raises
    ------
    RuntimeError
        If `allow_cpu` is False and no CUDA GPU is available to PyTorch (the message names the
        probable cause, the PyTorch version, and the CUDA version of the PyTorch build) or the
        Cellpose model is not on a CUDA device.
    """
    mod_path = str(model_path(model)) if model is not None else None
    worklist = {}
    skipped = []
    hand_edited = []
    for well in exp.pick_wells(wells):
        paths = alignment_paths(exp, well)
        edited = [index for index in range(len(paths)) if original_label_path(exp, well, index).exists()]
        well_work = tuple(
            (round_index, image_path, label)
            for round_index, image_path in enumerate(paths)
            if round_index not in edited and (not (label := label_path(exp, well, round_index)).exists() or overwrite)
        )
        if well_work:
            worklist[well] = well_work
        if edited:
            rounds = "round" if len(edited) == 1 else "rounds"
            hand_edited.append(f"{well} ({rounds} {', '.join(str(round_index) for round_index in edited)})")

        n_skipped = len(paths) - len(well_work) - len(edited)
        if n_skipped > 0:
            skipped.append(f"{well} ({n_skipped} of {len(paths)} rounds)")

    if hand_edited:
        say(f"Edited by hand, so kept even with Overwrite: {', '.join(hand_edited)}.")

    if not worklist:
        say("Every round already has labels, nothing to segment. To overwrite them from scratch, use overwrite=True")
        return

    if skipped:
        say_skipped(done="segmented", skipped=skipped)

    import torch

    torch.sparse.check_sparse_tensor_invariants.disable()

    from cellpose import models

    gpu = torch.cuda.is_available()
    if not gpu:
        if not allow_cpu:
            message = "No GPU found, so Segment nuclei would run on the CPU. "
            cuda_version = torch.version.cuda
            message += (
                f"PyTorch was installed with no GPU support (torch {torch.__version__}, no CUDA version found)."
                if cuda_version is None
                else (
                    "The NVIDIA GPU drivers may be missing or too old for this PyTorch version "
                    f"(torch {torch.__version__}, CUDA {cuda_version}). Update the NVIDIA drivers."
                )
            )
            raise RuntimeError(message)
        say("GPU not detected, segmenting on the CPU.")

    cp_model = models.CellposeModel(gpu=gpu, pretrained_model=mod_path) if mod_path else models.CellposeModel(gpu=gpu)

    if cp_model.device.type == "cuda":
        say(f"Detected GPU: {torch.cuda.get_device_name(0)}")
    if cp_model.device.type != "cuda" and not allow_cpu:
        raise RuntimeError(f"Cellpose is running on {cp_model.device}, not the GPU.")

    outcomes: list[str | Failed] = []  # each well's name, or its Failed
    not_written: defaultdict[str, list[int]] = defaultdict(list)
    for well, work in progress(worklist.items(), desc="Overall Progress", unit="well", total=len(worklist)):
        try:
            segment_path = Path(exp.segmented) / well
            segment_path.mkdir(parents=True, exist_ok=True)

            # the bar closes even when the well fails, so the next well's bar starts clean
            with progress(work, desc=f"Segmenting {well}", unit="round", leave=False) as round_bar:
                for round_index, path_image, path_label in round_bar:
                    image = imread(path_image)
                    labels, peak_gb = segment_image(
                        cp_model,
                        image,
                        diameter,
                        desc=f"Segmenting round {round_index}",
                        min_size=min_size,
                        cellprob_threshold=cellprob_threshold,
                        flow_threshold=flow_threshold,
                    )
                    if original_label_path(exp, well, round_index).exists():
                        not_written[well].append(round_index)
                        torch.cuda.empty_cache()
                        continue

                    write_tiff(path_label, labels)

                    if peak_gb is not None:
                        round_bar.set_postfix(peak_VRAM=f"{peak_gb:.2f}GB")

                    torch.cuda.empty_cache()
        except Stopped:
            raise  # a stop ends the step; the rounds already written are kept
        except Exception as error:  # one well failing: note it and go on to the next well
            outcomes.append(Failed(well, error_in_words(error)))
        else:
            outcomes.append(well)

    del cp_model
    torch.cuda.empty_cache()

    if not_written:
        say(
            "The following labels were edited by hand while this step ran, so they were not overwritten: "
            f"{', '.join(f'{w} (round(s): {", ".join(map(str, rounds))})' for w, rounds in not_written.items())}"
        )
    check_results(outcomes, unit="well")


def segment_image(
    model: Any,
    im: npt.NDArray[Any],
    diameter: float,
    *,
    flow_threshold: float,
    cellprob_threshold: float,
    min_size: int,
    desc: str | None = None,
) -> tuple[npt.NDArray[Any], float | None]:
    """Segment a 2D image with a Cellpose model, with a progress bar over the image's tiles.

    During segmentation, `cellpose.transforms.make_tiles` is replaced by a wrapper that sets the
    bar's total to the number of tiles. A forward hook on `model.net` advances the bar by the number
    of tiles in each batch. The original `make_tiles` is restored and the hook removed when
    segmentation ends, including when `model.eval` raises. When `cellpose.transforms.make_tiles`
    does not exist, the image is segmented without the bar, with a `UserWarning`.

    Parameters
    ----------
    model : Any
        A Cellpose `CellposeModel`.
    im : npt.NDArray[Any]
        The 2D image to segment.
    diameter : float
        The expected nucleus diameter in pixels. Cellpose rescales the image by 30 / `diameter`.
    flow_threshold : float
        The flow error threshold. Cellpose keeps masks with a flow error below the threshold.
    cellprob_threshold : float
        The cell probability threshold. Cellpose includes pixels above the threshold in masks.
    min_size : int
        The minimum mask size in pixels. Cellpose removes smaller masks.
    desc : str | None, default=None
        The label of the progress bar.

    Returns
    -------
    labels : npt.NDArray[Any]
        The label (segmentation mask) image. Background pixels are 0. The pixels of each nucleus
        share one positive integer label.
    peak_gb : float | None
        The peak GPU memory reserved by PyTorch during segmentation, in GiB. None when the model is
        not on a CUDA device or the image is segmented without the bar.
    """
    import torch
    from cellpose import transforms

    original_make_tiles = getattr(transforms, "make_tiles", None)
    if original_make_tiles is None:
        warnings.warn(
            "Cellpose changed, so the tile progress bar is turned off. Segmentation is not "
            "affected. Please tell whoever maintains the pipeline so it can be updated "
            "(cellpose.transforms.make_tiles is missing).",
            stacklevel=1,
        )
        labels, _, _ = model.eval(
            im,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
        )
        return labels, None

    on_gpu = torch.device(model.device).type == "cuda"
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(model.device)

    bar = progress(desc=desc, unit="tile", leave=False)

    def count_tiles(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        bar.update(inputs[0].shape[0])  # must return nothing

    def make_tiles_and_count(*args: Any, **kwargs: Any) -> Any:
        result = original_make_tiles(*args, **kwargs)
        bar.reset(len(result[1]))
        return result

    hook = model.net.register_forward_hook(count_tiles)

    # lets the bar show total tiles, put back in `finally` below.
    transforms.make_tiles = make_tiles_and_count
    try:
        labels, _, _ = model.eval(
            im,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
        )
    finally:
        hook.remove()
        transforms.make_tiles = original_make_tiles
        bar.close()

    peak_gb = torch.cuda.max_memory_reserved(model.device) / (1024**3) if on_gpu else None

    return labels, peak_gb
