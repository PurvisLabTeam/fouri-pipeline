import json
from pathlib import Path, PureWindowsPath

import pytest

from fouri import models
from fouri.app.finetuning import delete_model

KEPT = "20261117T204206"
PARTIAL = "20261118T101500.partial"
NO_MODEL_JSON = "20261120T000000"
NOT_KEPT = "20261201T000000"
MODEL_JSON = {
    "created": "2026-11-17T20:42:06",
    "experiment_key": "test",
    "config": "/data/4i/experiment.json",
    "wells": {"B02": [0]},
    "squares": 1,
    "n_epochs": 100,
}


@pytest.fixture
def models_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(models, "app_data_dir", lambda: tmp_path / "app-data")
    return models.models_dir()


def _model_directory(directory: Path, *, model_json: bool = True) -> None:
    (directory / "models").mkdir(parents=True)
    (directory / "models" / "model").write_bytes(b"stand-in model")
    if model_json:
        (directory / "model.json").write_text(json.dumps(MODEL_JSON), encoding="utf-8")


def _tree(root: Path) -> dict[Path, bytes | None]:
    return {path.relative_to(root): path.read_bytes() if path.is_file() else None for path in root.rglob("*")}


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "{drive}model", f"../cellpose/{KEPT}", f"..\\cellpose\\{KEPT}", PARTIAL, NO_MODEL_JSON, NOT_KEPT],
    ids=["empty", "dot", "dot-dot", "drive", "slash", "backslash", "partial", "no-model-json", "not-kept"],
)
def test_delete_model_refuses_names_outside_the_models_directory(models_folder: Path, name: str) -> None:
    name = name.format(drive=PureWindowsPath(models_folder).drive or "C:")
    for directory in (KEPT, "model", PARTIAL):
        _model_directory(models_folder / directory)
    _model_directory(models_folder / NO_MODEL_JSON, model_json=False)
    (models_folder / "model.json").write_text(json.dumps(MODEL_JSON), encoding="utf-8")
    (models_folder.parent / "model.json").write_text(json.dumps(MODEL_JSON), encoding="utf-8")
    root = models_folder.parent.parent
    before = _tree(root)

    with pytest.raises(FileNotFoundError):
        delete_model(name)

    assert _tree(root) == before
