from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from builders import with_filters, with_labels

from fouri.app.batches import save_batch_setup
from fouri.app.embeddings import save_setup
from fouri.app.rules import save_filters, save_labels, switched_on_filters, switched_on_labels
from fouri.config import Experiment, load_config, save_config

SMALL_AREAS = {"name": "small areas", "use": True, "rule": "between", "feature": "nuc_area", "low": 0, "high": 40}
SIZE = {
    "column": "size",
    "use": True,
    "rule": "threshold",
    "feature": "nuc_area",
    "cutoffs": [35],
    "classes": ["s", "l"],
}
BATCH_SETUP = {"batch": "plate column", "keep": ["treatment"], "log_intensities": True}
EMBEDDING_SETUP = {
    "picked": ["nuc_area", "R0_pRB_nuc_mean"],
    "samples": [1, 3],
    "ranking_k": 10,
    "top_n": [2],
    "methods": {"UMAP": {"dims": [2], "settings": {"n_neighbors": [5], "min_dist": [0.1], "metric": ["euclidean"]}}},
    "skipped": [],
    "seed": 3,
}

SAVES: dict[str, Callable[[Experiment], Experiment]] = {
    "filters": lambda exp: save_filters(exp, [SMALL_AREAS]),
    "labels": lambda exp: save_labels(exp, [SIZE]),
    "batch set-up": lambda exp: save_batch_setup(exp, BATCH_SETUP),
    "embedding set-up": lambda exp: save_setup(exp, EMBEDDING_SETUP),
}


def _files(folder: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in folder.glob("experiment.json*")}


@pytest.mark.parametrize("app", [["filters"], "filters"], ids=["list", "text"])
@pytest.mark.parametrize("save", list(SAVES), ids=list(SAVES))
def test_an_app_value_not_a_mapping_refuses_every_save(experiment: Experiment, app: Any, save: str) -> None:
    save_config({**experiment.config, "app": app}, experiment.config_path)
    exp = load_config(experiment.config_path, announce=False)
    before = _files(Path(exp.config_path).parent)

    with pytest.raises(ValueError):
        SAVES[save](exp)

    assert _files(Path(exp.config_path).parent) == before


def test_only_switched_on_filters_and_labels_are_sent(experiment: Experiment) -> None:
    exp = with_filters(
        experiment,
        [
            SMALL_AREAS,
            {**SMALL_AREAS, "name": "very small areas", "use": False, "high": 33},
            {**SMALL_AREAS, "name": "large areas", "low": 40, "high": 100},
        ],
    )
    exp = with_labels(
        exp,
        [
            SIZE,
            {**SIZE, "column": "pRB level", "use": False, "feature": "R0_pRB_nuc_mean", "cutoffs": [1200]},
            {**SIZE, "column": "RB level", "feature": "R0_RB_nuc_mean", "cutoffs": [900]},
        ],
    )

    assert switched_on_filters(exp) == ["small areas", "large areas"]
    assert switched_on_labels(exp) == ["size", "RB level"]
