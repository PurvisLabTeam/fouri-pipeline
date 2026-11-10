"""Helpers for parallel processing in the notebooks."""

import math
import os
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing import get_context

import psutil

from fouri.report import StepFailed, Stopped, progress, say

MEM_PER_WORKER = 8e9  # In bytes, equals 8 GB per worker
MEM_MIN_FREE = 4e9  # In bytes, equals 4 GB
MIN_CPU_FREE = 2  # Minimum number of cpus to keep available to the OS


@dataclass(frozen=True)
class Failed:
    """An item whose job raised an error."""

    item: str
    error: str


@dataclass(frozen=True)
class Dropped:
    """An item that did not start because a run was manually stopped."""

    item: str


def check_results(results: Sequence[object], *, unit: str) -> None:
    """Say which items failed.

    Raise Stopped if any were dropped, otherwise StepFailed if any failed.
    """
    failed = []
    dropped = []
    for result in results:
        match result:
            case Failed(item=item, error=error):
                say(f"{item}: {error}")
                failed.append(item)
            case Dropped(item=item):
                dropped.append(item)
    if dropped:
        raise Stopped

    failed = list(dict.fromkeys(failed).keys())
    if failed:
        message = f"{len(failed)} {unit}(s) failed: {', '.join(failed)}. The other {unit}s finished and were saved."
        raise StepFailed(message, failed)


def parallel_map[T, R](
    function: Callable[[T], R],
    items: Sequence[T],
    *,
    desc: str,
    unit: str,
    n_workers: int | None = None,
    mem_per_worker: float = MEM_PER_WORKER,
    names: Sequence[str] | None = None,
) -> list[R | Failed | Dropped]:
    """Run ``function`` over ``items`` in spawned processes; return each outcome in item order.

    An item whose job raised comes back as a ``Failed``. When a bar raises ``Stopped``, the items
    no worker has taken yet come back as ``Dropped``, and the ones running finish first. ``names``
    names each item in these; without it an item's name is ``str(item)``.
    """
    if not items:
        return []
    names = [str(item) for item in items] if names is None else names
    n_workers = (
        min(n_workers, len(items))
        if n_workers is not None
        else worker_count(max_count=len(items), mem_per_worker=mem_per_worker)
    )
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn")) as executor:
        jobs = [executor.submit(function, item) for item in items]
        try:
            for _ in progress(as_completed(jobs), desc=desc, unit=unit, total=len(jobs)):
                pass  # results are read below, in item order
        except Stopped:
            for job in jobs:
                job.cancel()  # works only on jobs no worker has taken yet; the running ones finish
    # leaving the pool waited for the jobs still running

    outcomes: list[R | Failed | Dropped] = []
    for name, job in zip(names, jobs, strict=True):
        if job.cancelled():  # checked first: asking a cancelled job for its error raises
            outcomes.append(Dropped(name))
            continue
        error = job.exception()
        if isinstance(error, BrokenProcessPool):  # a RuntimeError, so checked before error_in_words
            broken = f"A worker process stopped unexpectedly, possibly out of memory; this {unit} did not finish."
            outcomes.append(Failed(name, broken))
        elif error is not None:
            outcomes.append(Failed(name, error_in_words(error)))
        else:
            outcomes.append(job.result())
    return outcomes


def error_in_words(error: BaseException) -> str:
    """Return an error as a message: fouri's own as written, anything else with its class name."""
    if isinstance(error, (ValueError, RuntimeError, FileNotFoundError)):
        return str(error)
    return f"{type(error).__name__}: {error}"


def worker_count(min_count=1, max_count=None, mem_per_worker: float = MEM_PER_WORKER):
    mem_available = psutil.virtual_memory().available
    cores = physical if (physical := psutil.cpu_count(logical=False)) is not None else os.cpu_count() or 1

    cpu_bound = cores - MIN_CPU_FREE
    mem_bound = math.floor((mem_available - MEM_MIN_FREE) / mem_per_worker)

    raw_min = min(cpu_bound, mem_bound)
    if raw_min < min_count:
        workers = min_count
        bottleneck = "forced minimum"
        warnings.warn(
            f"Severe resource constraints detected (CPU bound: {cpu_bound}. Mem bound: "
            f"{_to_gb(mem_available):.1f}). Forcing worker count to {min_count}. Expect "
            "degraded performance.",
            stacklevel=2,
        )
    else:
        workers = raw_min
        bottleneck = "cpu" if workers == cpu_bound else "memory"

    # no more processes than jobs to run (max_count of 0 or less is ignored)
    if max_count is not None and 0 < max_count < workers:
        workers = max_count

    say(
        f"Running {workers} job(s) in parallel (limited by {bottleneck}: {_to_gb(mem_available):.1f}GB "
        f"available memory, {cores} total cores)"
    )
    return workers


def _to_gb(num_bytes):
    return num_bytes / 1e9
