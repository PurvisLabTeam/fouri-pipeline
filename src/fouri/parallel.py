"""Helpers for parallel processing in the notebooks."""

import math
import os
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import psutil

from fouri.report import progress

MEM_PER_WORKER = 8e9  # In bytes, equals 8 GB per worker
MEM_MIN_FREE = 4e9  # In bytes, equals 4 GB
MIN_CPU_FREE = 2  # Minimum number of cpus to keep available to the OS


def parallel_map[T, R](
    function: Callable[[T], R],
    items: Sequence[T],
    *,
    desc: str,
    unit: str,
    n_workers: int | None = None,
    mem_per_worker: float = MEM_PER_WORKER,
) -> list[R]:
    if not items:
        return []
    n_workers = (
        min(n_workers, len(items))
        if n_workers is not None
        else worker_count(max_count=len(items), mem_per_worker=mem_per_worker)
    )
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn")) as executor:
        results = executor.map(function, items)
        return list(progress(results, desc=desc, unit=unit, total=len(items)))


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

    print(
        f"Running {workers} job(s) in parallel (limited by {bottleneck}: {_to_gb(mem_available):.1f}GB "
        f"available memory, {cores} total cores)"
    )
    return workers


def _to_gb(num_bytes):
    return num_bytes / 1e9
