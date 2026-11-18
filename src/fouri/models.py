"""@TODO."""

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from fouri.io import app_data_dir


@dataclass(frozen=True)
class KeptModel:
    name: str
    created: dt.datetime
    experiment_key: str
    wells: dict[str, list[int]]
    path: Path


def models_dir() -> Path:
    """@TODO."""
    return app_data_dir() / "models" / "cellpose"


def kept_models() -> tuple[list[KeptModel], list[str]]:
    """@TODO."""
    if not models_dir().exists():
        return [], []
    subdirs = [d for d in models_dir().iterdir() if not d.name.endswith(".partial") and (d / "model.json").exists()]

    models = []
    skipped = []
    for directory in subdirs:
        try:
            model = _read_kept_model(directory)
        except (KeyError, ValueError, TypeError):
            skipped.append(directory.name)
        else:
            models.append(model)

    models.sort(key=lambda model: (model.created, model.name), reverse=True)
    skipped.sort()
    return models, skipped


def model_path(name: str) -> Path:
    """@TODO."""
    directory = models_dir() / name
    if directory.name.endswith(".partial") or not (directory / "model.json").exists():
        raise FileNotFoundError(f"Model {name} is not on this PC.")

    try:
        model = _read_kept_model(directory)
    except (KeyError, ValueError, TypeError) as err:
        raise FileNotFoundError(f"Model {name} cannot be used. Its model.json file is corrupted.") from err

    return model.path


def shown_name(model: KeptModel) -> str:
    """@TODO."""
    return f"Fine-tuned {model.created:%Y-%m-%d %H:%M} [{model.experiment_key}: {', '.join(model.wells)}]"


def _read_kept_model(directory: Path) -> KeptModel:
    with open(directory / "model.json", encoding="utf-8") as fh:
        model = json.load(fh)

    created = dt.datetime.fromisoformat(model["created"])
    experiment_key = model["experiment_key"]
    wells = model["wells"]

    if not isinstance(experiment_key, str):
        raise TypeError(
            f"{directory / 'model.json'} contains an invalid type for key 'experiment_key': "
            f"Expected str, got {type(experiment_key)}"
        )

    wells_checks = (
        isinstance(wells, dict)
        and all(isinstance(key, str) for key in wells)
        and all(isinstance(val, list) for val in wells.values())
        and all(isinstance(v, int) and not isinstance(v, bool) for val in wells.values() for v in val)
    )
    if not wells_checks:
        raise TypeError(
            f"{directory / 'model.json'} contains invalid types for the 'wells' dict. Must be dict[str, list[int]]."
        )

    return KeptModel(
        name=directory.name,
        path=directory / "models" / "model",
        experiment_key=experiment_key,
        wells=wells,
        created=created,
    )
