"""Cytoplasmic rings around the nuclei of a label (segmentation mask) image.

Method by Katarzyna Kedziora (2021).
"""

import cv2
import numpy as np
import numpy.typing as npt
from skimage.morphology import disk


def make_rings(image: npt.NDArray, width: int = 5, gap: int = 0) -> npt.NDArray:
    """Return the cytoplasmic ring around each nucleus, labeled with the nucleus's label.

    A pixel is added to a ring when exactly one nucleus lies within `width` pixels and no nucleus
    lies within `gap` pixels. A pixel within `width` pixels of two or more nuclei joins no ring.
    Distances are Euclidean, measured with a disk-shaped structuring element
    (`skimage.morphology.disk`).

    Parameters
    ----------
    image : npt.NDArray
        The label (segmentation mask) image of the nuclei, 0 for background. Must hold a distinct
        positive integer for each nucleus.
    width : int, default=5
        The maximum width of the ring from the nucleus' edge.
    gap : int, default=0
        The width in pixels of the band around every nucleus that is left out of all rings. The ring
        starts `gap + 1` pixels from the nucleus' edge. With 0, the ring touches the nucleus.

    Returns
    -------
    npt.NDArray
        The ring label image, in the shape and dtype of `image`. Each ring pixel holds the label of
        the ring's nucleus. Every other pixel is 0.

    Notes
    -----
    The returned image has rings only, and sets nucleus pixels to 0.
    """
    reach = disk(width)
    image_f32 = image.astype(np.float32)

    dilated = cv2.dilate(image_f32, reach)
    eroded = cv2.erode(np.where(image_f32 > 0, image_f32, np.inf), reach)
    too_close = cv2.dilate((image_f32 > 0).astype(np.uint8), disk(gap)) > 0

    in_ring = (dilated == eroded) & (dilated > 0) & ~too_close
    return np.where(in_ring, dilated, 0).astype(image.dtype)
