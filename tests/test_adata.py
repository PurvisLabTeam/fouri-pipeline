import re

import anndata as ad
import numpy as np
import pytest
from builders import measured_values, varied_cell_data

from fouri import io, run
from fouri.config import Experiment


def _built(exp: Experiment) -> ad.AnnData:
    run.build_cell_table(exp)
    return ad.read_h5ad(io.adata_path(exp))


def test_var_kind_and_measure_are_set_per_column(experiment: Experiment) -> None:
    varied_cell_data(experiment)

    var = _built(experiment).var

    assert var.loc["R0_pRB_nuc_mean", ["kind", "round", "signal", "measure"]].tolist() == [
        "channel",
        "R0",
        "pRB",
        "nuc_mean",
    ]
    assert var.loc["R2_EdU_nuc_mean", ["kind", "round", "signal", "measure"]].tolist() == [
        "channel",
        "R2",
        "EdU",
        "nuc_mean",
    ]
    assert var.loc["nuc_area", "kind"] == "shape"
    assert var.loc["pRB_RB_ratio", "kind"] == "derived"
    for name, row in var[var["kind"] == "channel"].iterrows():
        assert name == f"{row['round']}_{row['signal']}_{row['measure']}"


def test_a_column_empty_in_an_included_cell_is_left_out_of_x(experiment: Experiment) -> None:
    varied_cell_data(experiment)
    measured_values(experiment, "B03", "R1_cMyc_nuc_mean", [2010.0, float("nan"), 2030.0, 2040.0])

    var_names = list(_built(experiment).var_names)

    assert "R1_cMyc_nuc_mean" not in var_names
    assert "R0_pRB_nuc_mean" in var_names


def test_a_value_differing_between_wells_is_left_empty_and_named(experiment: Experiment) -> None:
    varied_cell_data(experiment)
    table = io.read_round_table(experiment, "B03")
    table.loc[(table["round_name"] == "R0") & (table["signal"] == "pRB"), "exposure_ms"] = 250.0
    io.save_round_table(experiment, "B03", table)

    var = _built(experiment).var

    assert np.isnan(var.loc["R0_pRB_nuc_mean", "exposure_ms"])
    assert var.loc["R0_RB_nuc_mean", "exposure_ms"] == pytest.approx(300.0)


def test_the_cell_table_id_is_new_each_run(experiment: Experiment) -> None:
    varied_cell_data(experiment)

    first = _built(experiment).uns["cell_table"]["id"]
    second = _built(experiment).uns["cell_table"]["id"]

    assert re.fullmatch(r"[0-9a-f]{32}", first)
    assert re.fullmatch(r"[0-9a-f]{32}", second)
    assert first != second
