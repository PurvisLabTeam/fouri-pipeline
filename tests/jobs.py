"""Toy jobs for parallel_map, in a module of their own so spawned workers can import them."""

import os
import time
from pathlib import Path


def square(item: int) -> int:
    return item * item


def square_unless(item: int, bad: int) -> int:
    """Square ``item``, or raise when it is ``bad``."""
    if item == bad:
        raise ValueError(f"made-up failure on item {item}")
    return item * item


def square_or_exit(item: int, bad: int) -> int:
    """Square ``item``, or end the worker outright on ``bad``, as running out of memory does."""
    if item == bad:
        os._exit(1)
    return item * item


def wait_then_report(seconds: float) -> tuple[float, float]:
    """Wait ``seconds``, then return them with the time this job finished."""
    time.sleep(seconds)
    return seconds, time.time()


def create_stop_file(item: int, stop_file: Path) -> int:
    """Create ``stop_file`` and return ``item``, after a short wait.

    The wait keeps the one worker from getting through the queue before the step notices the stop.
    """
    Path(stop_file).touch()
    time.sleep(0.3)
    return item
