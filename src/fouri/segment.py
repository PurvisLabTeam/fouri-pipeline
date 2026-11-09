"""Cellpose segmentation with progress bar over an image's tiles."""

import warnings
from collections.abc import Sequence
from pathlib import Path

from tifffile import imread

from fouri.config import Experiment
from fouri.io import alignment_paths, label_path, write_tiff
from fouri.report import progress, say, say_skipped


def gpu_name() -> str | None:
    import torch

    gpu = torch.cuda.is_available()
    return torch.cuda.get_device_name(0) if gpu else None


def segment_wells(
    exp: Experiment,
    diameter: float = 30,
    allow_cpu: bool = False,
    overwrite: bool = False,
    wells: Sequence[str] | None = None,
) -> None:
    """Segment all the round images for all wells in the experiment."""
    worklist = {}
    skipped = []
    for well in exp.pick_wells(wells):
        paths = alignment_paths(exp, well)
        well_work = tuple(
            (round_index, image_path, label)
            for round_index, image_path in enumerate(paths)
            if not (label := label_path(exp, well, round_index)).exists() or overwrite
        )
        if well_work:
            worklist[well] = well_work

        n_skipped = len(paths) - len(well_work)
        if n_skipped > 0:
            skipped.append(f"{well} ({n_skipped} of {len(paths)} rounds)")

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
            raise RuntimeError(
                f"No GPU available to PyTorch (torch {torch.__version__}, CUDA build "
                f"{torch.version.cuda}). Segmenting on CPU takes significantly longer. To do it "
                "anyway choose 'Run on the CPU' in the app or set allow_cpu=True. If this PC has "
                "a GPU and CUDA build is None, PyTorch was installed without GPU support. Run "
                "'uv sync'."
            )
        say("GPU not detected, segmenting on the CPU.")

    model = models.CellposeModel(gpu=gpu)

    if model.device.type == "cuda":
        say(f"Detected GPU: {torch.cuda.get_device_name(0)}")
    if model.device.type != "cuda" and not allow_cpu:
        raise RuntimeError(f"Cellpose is running on {model.device}, not the GPU.")

    for well, work in progress(worklist.items(), desc="Overall Progress", unit="well", total=len(worklist)):
        segment_path = Path(exp.segmented) / well
        segment_path.mkdir(parents=True, exist_ok=True)

        round_bar = progress(work, desc=f"Segmenting {well}", unit="round", leave=False)
        for round_index, path_image, path_label in round_bar:
            image = imread(path_image)
            labels, peak_gb = segment_image(model, image, diameter, desc=f"Segmenting round {round_index}")
            write_tiff(path_label, labels)

            if peak_gb is not None:
                round_bar.set_postfix(peak_VRAM=f"{peak_gb:.2f}GB")

            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()


def segment_image(model, im, diameter, desc=None):
    """Segment one 2D image with a Cellpose model, showing a bar over its tiles.

    Returns labels and peak gpu memory in GB, or None when the model isn't on
    the GPU.
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
        labels, _, _ = model.eval(im, diameter=diameter)
        return labels, None

    on_gpu = torch.device(model.device).type == "cuda"
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(model.device)

    bar = progress(desc=desc, unit="tile", leave=False)

    def count_tiles(module, inputs, output):
        bar.update(inputs[0].shape[0])  # must return nothing

    def make_tiles_and_count(*args, **kwargs):
        result = original_make_tiles(*args, **kwargs)
        bar.reset(len(result[1]))
        return result

    hook = model.net.register_forward_hook(count_tiles)

    # lets the bar show total tiles, put back in `finally` below.
    transforms.make_tiles = make_tiles_and_count
    try:
        labels, _, _ = model.eval(im, diameter=diameter)
    finally:
        hook.remove()
        transforms.make_tiles = original_make_tiles
        bar.close()

    peak_gb = torch.cuda.max_memory_reserved(model.device) / (1024**3) if on_gpu else None

    return labels, peak_gb
