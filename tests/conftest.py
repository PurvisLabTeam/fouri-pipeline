"""Fixtures for every test: the made-up experiment, and messages printed again after each test."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from builders import make_experiment

from fouri import report
from fouri.config import Experiment


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    """Return the made-up experiment, written in the test's temporary folder."""
    return make_experiment(tmp_path)


@pytest.fixture(autouse=True)
def _print_again() -> Iterator[None]:
    """Send messages back to printing after the test, since report keeps its progress file."""
    yield
    report.report_to(None)
