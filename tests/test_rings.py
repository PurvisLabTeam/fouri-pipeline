import numpy as np
import pytest

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


@pytest.mark.parametrize("gap", [1, 2])
def test_a_pixel_within_the_gap_joins_no_ring(gap: int) -> None:
    labels = np.zeros((30, 30), dtype=np.uint16)
    labels[12:18, 12:18] = 1

    rings = make_rings(labels, width=6, gap=gap)

    assert not rings[12:18, 12:18].any()
    assert not rings[15, 18 : 18 + gap].any()
    assert not rings[15, 12 - gap : 12].any()
    assert not rings[18 : 18 + gap, 15].any()
    assert rings[15, 18 + gap] == 1
    assert rings[15, 11 - gap] == 1
    assert rings[18 + gap, 15] == 1
