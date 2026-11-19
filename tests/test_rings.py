import numpy as np

from fouri.rings import make_rings


def test_a_pixel_two_nuclei_could_claim_joins_no_ring() -> None:
    labels = np.zeros((30, 40), dtype=np.uint16)
    labels[12:18, 8:14] = 1
    labels[12:18, 18:24] = 2

    rings = make_rings(labels, width=6, gap=1)

    assert not rings[12:18, 14:18].any()
    assert rings[15, 4] == 1
    assert rings[15, 27] == 2
    assert set(np.unique(rings)) == {0, 1, 2}
