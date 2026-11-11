"""Fixtures for every test: the made-up experiment, and messages printed again after each test."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from builders import make_experiment

from fouri import report
from fouri.config import Experiment


def pytest_xdist_auto_num_workers(config):
    """Control the number of xdist workers when `-n auto` is passed.

    Calculates a percentage of the total system cores.
    """
    total_cores = os.cpu_count() or 1
    workers = max(1, int(total_cores * 0.75))
    return workers


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    """Return the made-up experiment, written in the test's temporary folder."""
    return make_experiment(tmp_path)


@pytest.fixture(autouse=True)
def _print_again() -> Iterator[None]:
    """Send messages back to printing after the test, since report keeps its progress file."""
    yield
    report.report_to(None)
