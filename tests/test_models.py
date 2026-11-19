import json
from pathlib import Path
from typing import Any

import pytest

from fouri import models

KEPT = "20261117T204206"
OTHER = "20261118T101500"
MODEL_JSON: dict[str, Any] = {
    "created": "2026-11-17T20:42:06",
    "experiment_key": "test",
    "config": "/data/4i/experiment.json",
    "wells": {"B02": [0, 2], "C02": [1]},
    "squares": 3,
    "n_epochs": 100,
}
WRONG_TYPES: dict[str, Any] = {
    "created": 20261117,
    "experiment_key": 7,
    "config": 7,
    "wells": ["B02"],
    "squares": "3",
    "n_epochs": "100",
}
BROKEN = [
    *(pytest.param({k: v for k, v in MODEL_JSON.items() if k != key}, id=f"{key}-missing") for key in MODEL_JSON),
    *(pytest.param({**MODEL_JSON, key: value}, id=f"{key}-wrong-type") for key, value in WRONG_TYPES.items()),
]


@pytest.fixture
def models_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(models, "app_data_dir", lambda: tmp_path / "app-data")
    return models.models_dir()


def _kept_model(folder: Path, name: str, entries: dict[str, Any]) -> None:
    directory = folder / name
    (directory / "models").mkdir(parents=True)
    (directory / "models" / "model").write_bytes(b"stand-in model")
    (directory / "model.json").write_text(json.dumps(entries), encoding="utf-8")


@pytest.mark.parametrize("entries", BROKEN)
def test_a_model_json_missing_any_key_is_left_out(models_folder: Path, entries: dict[str, Any]) -> None:
    _kept_model(models_folder, KEPT, MODEL_JSON)
    _kept_model(models_folder, OTHER, entries)

    kept, skipped = models.kept_models()

    assert [model.name for model in kept] == [KEPT]
    assert skipped == [OTHER]
    assert models.model_path(KEPT) == models_folder / KEPT / "models" / "model"
    with pytest.raises(FileNotFoundError):
        models.model_path(OTHER)


def test_a_partial_model_is_neither_listed_nor_accepted(models_folder: Path) -> None:
    _kept_model(models_folder, KEPT, MODEL_JSON)
    _kept_model(models_folder, f"{OTHER}.partial", {**MODEL_JSON, "created": "2026-11-18T10:15:00"})

    kept, skipped = models.kept_models()

    assert [model.name for model in kept] == [KEPT]
    assert skipped == []
    for name in (f"{OTHER}.partial", OTHER):
        with pytest.raises(FileNotFoundError):
            models.model_path(name)
