import json
import platform
from pathlib import Path
from typing import Any

import pytest
from builders import CONFIG

from fouri.config import Experiment, load_config, save_config


def _written(folder: Path, config: dict[str, Any]) -> Experiment:
    path = folder / "experiment.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return load_config(path, announce=False)


def test_pick_wells_none_gives_plate_order_without_repeats(tmp_path: Path) -> None:
    exp = _written(tmp_path, {**CONFIG, "well_list": ["C02", "B02", "B03", "B02"]})

    assert exp.pick_wells(None) == ["B02", "B03", "C02"]


def test_an_interrupted_save_leaves_experiment_json_whole(experiment: Experiment) -> None:
    path = Path(experiment.config_path)
    before = path.read_bytes()

    with pytest.raises(TypeError):
        save_config({**experiment.config, "unwritable": object()}, path)

    assert path.read_bytes() == before
    assert load_config(path, announce=False).config == experiment.config
    assert sorted(path.parent.glob("*.partial")) == []


@pytest.mark.parametrize(
    ("keys", "system", "base"),
    [
        pytest.param({}, "Linux", "config", id="config-folder"),
        pytest.param({"base_dir": "base"}, "Linux", "base", id="base-dir"),
        pytest.param({"base_dir": "base", "mac_base_dir": "mac"}, "Darwin", "mac", id="mac-base-dir-on-macos"),
        pytest.param({"base_dir": "base", "mac_base_dir": "mac"}, "Linux", "mac", id="mac-base-dir-on-linux"),
        pytest.param({"base_dir": "base", "mac_base_dir": "mac"}, "Windows", "base", id="base-dir-on-windows"),
    ],
)
def test_paths_resolve_against_base_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keys: dict[str, str], system: str, base: str
) -> None:
    monkeypatch.setattr(platform, "system", lambda: system)
    folders = {name: tmp_path / name for name in ("config", "base", "mac")}
    folders["config"].mkdir()
    elsewhere = tmp_path / "elsewhere" / "masks"
    config = {
        **CONFIG,
        **{key: str(folders[name]) for key, name in keys.items()},
        "paths": {"segmented": "labels/nuclei", "masks": str(elsewhere)},
    }

    exp = _written(folders["config"], config)

    assert Path(exp.base_dir) == folders[base]
    assert Path(exp.segmented) == folders[base] / "labels" / "nuclei"
    assert Path(exp.masks) == elsewhere
    assert Path(exp.cell_data) == folders[base] / "cell_data"
