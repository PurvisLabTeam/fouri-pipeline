from collections.abc import Sequence

import anndata as ad
import pandas as pd
import pytest
from builders import subsampled, varied_cell_data, with_filters

from fouri import io, run
from fouri.app.cells import load_cells
from fouri.app.exports import export_counts
from fouri.app.rules import kept_cells, switched_on_filters
from fouri.config import Experiment
from fouri.export import export_cells

SMALL_AREAS = {"name": "small areas", "use": True, "rule": "between", "feature": "nuc_area", "low": 0, "high": 40}
SWITCHED_OFF = {**SMALL_AREAS, "name": "very small areas", "use": False, "high": 33}
KEY = "X_phate_3d_top2_knn5"


def _table(exp: Experiment, wells: Sequence[str] | None = None) -> Experiment:
    varied_cell_data(exp, wells)
    exp = with_filters(exp, [SMALL_AREAS, SWITCHED_OFF])
    run.build_cell_table(exp, filters=switched_on_filters(exp))
    return exp


@pytest.mark.parametrize("wells", [None, ["B02", "C02"]], ids=["every-well", "one-well-not-measured"])
def test_kept_cells_match_build_cell_tables_included(experiment: Experiment, wells: Sequence[str] | None) -> None:
    exp = _table(experiment, wells)

    cells = load_cells(exp)
    kept, problem = kept_cells(cells, exp)

    assert problem is None
    assert kept is not None
    shown = pd.Series(kept.to_numpy(), index=pd.MultiIndex.from_frame(cells.table[["well", "label"]].astype(str)))
    built = pd.read_csv(io.fullest_df_path(exp))
    included = pd.Series(
        built["included"].to_numpy(), index=pd.MultiIndex.from_frame(built[["well", "label"]].astype(str))
    )
    assert shown.sort_index().to_dict() == included.sort_index().to_dict()
    assert not included.all()


@pytest.mark.parametrize("embeddings", [(), (KEY,)], ids=["every-cell", "subsampled"])
def test_export_counts_match_the_export(experiment: Experiment, embeddings: Sequence[str]) -> None:
    exp = _table(experiment)
    if embeddings:
        subsampled(exp, [0, 2, 5, 7, 9, 10, 11], KEY)

    counts = export_counts(exp, corrected=False, embeddings=embeddings)
    path = export_cells(exp, "preview check", embeddings=embeddings)

    record = ad.read_h5ad(path).uns["export"]
    assert counts.cells == int(record["cells"])
    assert counts.filtered == int(record["filtered"])
    assert counts.filtered > 0
