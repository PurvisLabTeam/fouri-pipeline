"""Fixtures for every test: the made-up experiment, and messages printed again after each test."""

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from builders import cellpose_modules, in_process_map, make_experiment, umap_module

from fouri import embedding, report
from fouri.config import Experiment

NICEGUI_STORAGE = Path(tempfile.mkdtemp(prefix="fouri-tests-nicegui-"))
os.environ["NICEGUI_STORAGE_PATH"] = str(NICEGUI_STORAGE)
NAPARI_SETTINGS = Path(tempfile.mkdtemp(prefix="fouri-tests-napari-"))
os.environ["NAPARI_CONFIG"] = str(NAPARI_SETTINGS / "settings.yaml")


def pytest_xdist_auto_num_workers(config):
    """Control the number of xdist workers when `-n auto` is passed.

    Calculates a percentage of the total system cores.
    """
    total_cores = os.cpu_count() or 1
    workers = max(1, int(total_cores * 0.75))
    return workers


def pytest_unconfigure(config: pytest.Config) -> None:
    shutil.rmtree(NICEGUI_STORAGE, ignore_errors=True)
    shutil.rmtree(NAPARI_SETTINGS, ignore_errors=True)


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    """Return the made-up experiment, written in the test's temporary folder."""
    return make_experiment(tmp_path)


@pytest.fixture(autouse=True)
def _print_again() -> Iterator[None]:
    """Send messages back to printing after the test, since report keeps its progress file."""
    yield
    report.report_to(None)


@pytest.fixture
def stand_in_umap(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = umap_module()
    monkeypatch.setitem(sys.modules, "umap", module)
    monkeypatch.setattr(embedding, "parallel_map", in_process_map)
    return module


@pytest.fixture
def stand_in_cellpose(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModuleType]:
    modules = cellpose_modules()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return modules
