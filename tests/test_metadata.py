import pandas as pd
import pytest

from fouri.metadata import check_metadata

CONDITIONS = ["treatment", "treatment_time"]


@pytest.mark.parametrize(
    ("rows", "levels", "named", "unnamed"),
    [
        pytest.param(
            [("B02", 1, "control", "24h"), ("B03", 2, "palbo", "24h"), ("C02", 3, "palbo", "96h")],
            [],
            [],
            [],
            id="no-problem",
        ),
        pytest.param(
            [
                ("B02", 1, "control", "24h"),
                ("B03", 2, "palbo", "24h"),
                ("C02", 3, "palbo", "96h"),
                ("B02", 1, "control", "24h"),
                ("C02", 3, "palbo", "96h"),
            ],
            ["error"],
            ["B02", "C02"],
            ["B03"],
            id="wells-listed-twice",
        ),
        pytest.param(
            [("B02", 1, "control", "24h"), ("B03", 1, "palbo", "24h"), ("C02", 3, "palbo", "96h")],
            ["warning"],
            [],
            [],
            id="sample-id-covering-two-conditions",
        ),
        pytest.param(
            [("B02", 1, "control", "24h"), ("B03", 2, "control", "24h"), ("C02", 3, "palbo", "96h")],
            ["warning"],
            [],
            [],
            id="condition-with-two-sample-ids",
        ),
        pytest.param(
            [
                ("B02", 1, "control", "24h"),
                ("B03", 1, "palbo", "24h"),
                ("C02", 3, "control", "24h"),
                ("B02", 1, "control", "24h"),
            ],
            ["error", "warning", "warning"],
            ["B02"],
            ["B03", "C02"],
            id="every-problem",
        ),
    ],
)
def test_check_metadata_names_duplicate_wells_and_id_problems(
    rows: list[tuple[str, int, str, str]], levels: list[str], named: list[str], unnamed: list[str]
) -> None:
    table = pd.DataFrame(rows, columns=["well", "sample_ID", *CONDITIONS])

    issues = check_metadata(table, CONDITIONS)

    errors = [message for level, message in issues if level == "error"]
    assert sorted(level for level, _ in issues) == levels
    assert all(any(well in message for message in errors) for well in named)
    assert not any(well in message for message in errors for well in unnamed)
