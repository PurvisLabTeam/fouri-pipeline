import pandas as pd
import pytest
from builders import aligned, converted, measured, segmented, varied_cell_data

from fouri import features, io
from fouri.config import Experiment

METADATA = {
    "B02": ("control", "24h", 1),
    "B03": ("palbo", "24h", 2),
    "C02": ("palbo", "96h", 3),
}


def test_metadata_joins_its_own_well(experiment: Experiment) -> None:
    varied_cell_data(experiment)

    table = features.build_fullest_df(experiment)

    columns = ["well", "treatment", "treatment time", "sample_ID"]
    for well, treatment, treatment_time, sample in table[columns].itertuples(index=False):
        assert (treatment, treatment_time, sample) == METADATA[well]


def test_treatment_time_is_renamed(experiment: Experiment) -> None:
    varied_cell_data(experiment)

    table = features.build_fullest_df(experiment)

    assert table.set_index("well")["treatment time"].groupby(level=0).first().to_dict() == {
        well: values[1] for well, values in METADATA.items()
    }


def test_sample_id_is_a_category_of_numbers(experiment: Experiment) -> None:
    varied_cell_data(experiment)

    table = features.build_fullest_df(experiment)

    assert isinstance(table["sample_ID"].dtype, pd.CategoricalDtype)
    assert list(table["sample_ID"].cat.categories) == [1, 2, 3]


def test_a_formula_error_other_than_a_missing_column_stops(experiment: Experiment) -> None:
    varied_cell_data(experiment)
    table = features.build_fullest_df(experiment)

    kept = features.add_derived_features(table, [{"name": "skipped", "formula": "R9_DNA_nuc_mean / nuc_area"}])
    with pytest.raises(SyntaxError):
        features.add_derived_features(table, [{"name": "broken", "formula": "nuc_area +"}])

    assert list(kept.columns) == list(table.columns)


def test_an_old_row_number_column_is_dropped(experiment: Experiment) -> None:
    varied_cell_data(experiment)
    path = io.cell_data_path(experiment, "B02")
    cells = pd.read_csv(path)
    cells.to_csv(path)

    table = features.build_fullest_df(experiment)

    assert len(table.columns) == len(cells.columns) + len(experiment.metadata_columns)


def test_measured_columns_come_from_the_round_tables(experiment: Experiment) -> None:
    unconverted = features.measured_columns(experiment)
    for well in ("B02", "B03"):
        converted(experiment, well)
    segmented(experiment, "B02")
    aligned(experiment, "B02")
    measured(experiment, "B02")
    written = set(pd.read_csv(io.cell_data_path(experiment, "B02")).columns) - {"well"}

    assert unconverted == []
    assert set(features.measured_columns(experiment)) == written
