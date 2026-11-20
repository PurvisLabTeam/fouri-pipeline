import pytest

from fouri.app.embeddings import read_number


@pytest.mark.parametrize("ranking_k", [1, 2, 3])
def test_the_set_up_refuses_a_ranking_k_below_4(ranking_k: int) -> None:
    number, problem = read_number(float(ranking_k), "ranking_k")

    assert number is None
    assert problem is not None
    assert read_number(4.0, "ranking_k") == (4, None)
