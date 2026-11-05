"""Cellpose segmentation with progress bar over an image's tiles."""

import warnings
from pathlib import Path

from tifffile import imread

from fouri.config import Experiment
from fouri.io import write_tiff
from fouri.preprocess import alignment_paths
from fouri.report import progress, say


def segment_wells(exp: Experiment, diameter: float = 30, allow_cpu: bool = False) -> None:
    """Segment all the round images for all wells in the experiment."""
    import torch

    torch.sparse.check_sparse_tensor_invariants.disable()

    from cellpose import models

    gpu = torch.cuda.is_available()
    if not gpu:
        if not allow_cpu:
            raise RuntimeError(
                f"No GPU available to PyTorch (torch {torch.__version__}, CUDA build "
                f"{torch.version.cuda}). If CUDA build is None, run uv sync. To run on the CPU "
                "(slow), set allow_cpu=True"
            )
        say("GPU not detected, Cellpose is running on CPU (allow_cpu is set to True)")

    model = models.CellposeModel(gpu=gpu)

    if model.device.type == "cuda":
        say(f"Detected GPU: {torch.cuda.get_device_name(0)}")
    if model.device.type != "cuda" and not allow_cpu:
        raise RuntimeError(f"Cellpose is running on {model.device}, not the GPU.")

    for well in progress(exp.well_list, desc="Overall Progress", unit="well", total=len(exp.well_list)):
        segment_path = Path(exp.segmented) / well
        segment_path.mkdir(parents=True, exist_ok=True)

        round_bar = progress(alignment_paths(exp, well), desc=f"Segmenting {well}", unit="round", leave=False)
        for round_index, path in enumerate(round_bar):
            image = imread(path)
            labels, peak_gb = segment_image(model, image, diameter, desc=f"Segmenting round {round_index}")
            img_path = segment_path / f"labels_{well}_round_{round_index:03d}.tif"
            write_tiff(img_path, labels)

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
