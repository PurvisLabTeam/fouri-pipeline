"""Cellpose segmentation with progress bar over an image's tiles"""

import warnings

from tqdm.auto import tqdm


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
            "(cellpose.transforms.make_tiles is missing)."
        )
        labels, _, _ = model.eval(im, diameter=diameter)
        return labels, None

    on_gpu = torch.device(model.device).type == "cuda"
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(model.device)

    bar = tqdm(desc=desc, unit="tile", leave=False)

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
