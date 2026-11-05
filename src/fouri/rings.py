"""Utils for building cytoplasmic rings around each nucleus.

Method by Katarzyna Kedziora (2021): a pixel is assigned to a nucleus's ring
only if no other nucleus would have an overlapping ring. Reimplemented with
OpenCV in 2026.
"""

import cv2
import numpy as np
import numpy.typing as npt
from skimage.morphology import disk


def make_rings(image: npt.NDArray, width: int = 5, gap: int = 0) -> npt.NDArray:
    """Return a ring of cytoplasm around each nucleus, with its label.

    Parameters
    ----------
    image : ndarray of int
        Label image of the nuclei: 0 for background, one number per nucleus, e.g. the anchor
        round's segmentation after ``clear_border``.
    width : int, optional
        How far a ring reaches from its nucleus's edge, in pixels. A pixel within ``width`` of
        two or more nuclei joins no ring. By default 5.
    gap : int, optional
        Pixels this close to any nucleus are left out, so a ring starts ``gap + 1`` pixels from
        its nucleus's edge. By default 0, a ring touching its nucleus.
    """
    reach = disk(width)
    image_f32 = image.astype(np.float32)

    dilated = cv2.dilate(image_f32, reach)
    eroded = cv2.erode(np.where(image_f32 > 0, image_f32, np.inf), reach)
    too_close = cv2.dilate((image_f32 > 0).astype(np.uint8), disk(gap)) > 0

    in_ring = (dilated == eroded) & (dilated > 0) & ~too_close
    return np.where(in_ring, dilated, 0).astype(image.dtype)
