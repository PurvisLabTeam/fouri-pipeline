"""Helpers for parallel processing in the notebooks."""

import math
import os
import re
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from enum import StrEnum
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


class _Bottleneck(StrEnum):
    ITEMS = "items"
    CORES = "CPU"
    MEM = "memory"
    FORCED_MIN = "forced_minimum"


def check_results(results: Sequence[object], *, unit: str) -> None:
    """Say which items failed.

    Raise Stopped if any were dropped, otherwise StepFailed if any failed.
    """
    reasons: dict[str, list[str]] = {}  # each failed item's distinct reasons, in order
    dropped = []
    finished = 0
    for result in results:
        match result:
            case Failed(item=item, error=error):
                say(f"{item}: {error}")
                item_reasons = reasons.setdefault(item, [])
                if (reason := _first_sentence(error)) not in item_reasons:
                    item_reasons.append(reason)
            case Dropped(item=item):
                dropped.append(item)
            case _:
                finished += 1
    if dropped:
        raise Stopped

    if reasons:
        listed = "".join(f"\n{item} ({'; '.join(why)})" for item, why in reasons.items())
        rerun = "run the step again to redo the failed ones."
        closing = f"The other {unit}s finished and were saved; {rerun}" if finished else rerun.capitalize()
        raise StepFailed(f"{len(reasons)} {unit}(s) failed:{listed}\n\n{closing}", list(reasons))


def _first_sentence(error: str) -> str:
    return re.split(r"\.\s|\n", error, maxsplit=1)[0].strip().removesuffix(".")


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
    names each item in these; without it an item's name is ``str(item)``. @TODO.
    """
    if not items:
        return []
    names = [str(item) for item in items] if names is None else names
    n_workers = (
        min(n_workers, len(items))
        if n_workers is not None
        else worker_count(max_count=len(items), mem_per_worker=mem_per_worker, unit=unit)
    )
    n_items = len(items)
    slots: list[Future[R] | None] = [None] * n_items
    next_item = 0
    running: set[Future[R]] = set()
    pool_broke = False
    bar = progress(desc=desc, unit=unit, total=n_items)
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn")) as executor, bar:
            while next_item < n_items or running != set():
                while len(running) < n_workers and next_item < n_items:
                    job = executor.submit(function, items[next_item])
                    slots[next_item] = job
                    running.add(job)
                    next_item += 1

                done, not_done = wait(running, return_when=FIRST_COMPLETED)
                running = not_done
                bar.update(n=len(done))
    except Stopped:
        pass
    except BrokenProcessPool:
        pool_broke = True

    broken = f"A worker process stopped unexpectedly, possibly out of memory. This {unit} did not finish."
    outcomes: list[R | Failed | Dropped] = []
    for name, slot in zip(names, slots, strict=True):
        if slot is None:
            outcomes.append(Failed(item=name, error=broken) if pool_broke else Dropped(item=name))
            continue
        match slot.exception():
            case None:
                outcomes.append(slot.result())
            case BrokenProcessPool():
                outcomes.append(Failed(item=name, error=broken))
            case _ as exc:
                outcomes.append(Failed(item=name, error=error_in_words(exc)))
    return outcomes


def error_in_words(error: BaseException) -> str:
    """Return an error as a message: fouri's own as written, anything else with its class name."""
    if isinstance(error, (ValueError, RuntimeError, FileNotFoundError)):
        return str(error)
    return f"{type(error).__name__}: {error}"


def worker_count(min_count=1, max_count=None, mem_per_worker: float = MEM_PER_WORKER, unit: str = "well"):
    mem_available = psutil.virtual_memory().available
    cores = physical if (physical := psutil.cpu_count(logical=False)) is not None else os.cpu_count() or 1

    cpu_bound = cores - MIN_CPU_FREE
    mem_bound = math.floor((mem_available - MEM_MIN_FREE) / mem_per_worker)
    available_gb = round(_to_gb(mem_available), 1)

    raw_min = min(cpu_bound, mem_bound)
    if raw_min < min_count:
        workers = min_count
        bottleneck = _Bottleneck.FORCED_MIN
    else:
        workers = raw_min
        bottleneck = _Bottleneck.CORES if workers == cpu_bound else _Bottleneck.MEM

    resource_limit = workers

    # no more processes than jobs to run (max_count of 0 or less is ignored)
    if max_count is not None and 0 < max_count < workers:
        workers = max_count
        bottleneck = _Bottleneck.ITEMS

    match bottleneck:
        case _Bottleneck.ITEMS:
            say(
                f"Running all {workers} {unit}(s) in parallel. This PC can run up to {resource_limit} "
                f"{unit}(s) at the same time ({available_gb} GB of memory available, {cores} cores)."
            )
        case _Bottleneck.MEM | _Bottleneck.CORES:
            say(
                f"Running {workers} {unit}(s) in parallel, limited by {bottleneck}: {available_gb} GB "
                f"of memory available, {cores} cores."
            )
        case _Bottleneck.FORCED_MIN:
            say(
                f"Running {workers} {unit}(s) in parallel: this PC is low on free memory or cores "
                f"({available_gb} GB of memory available, {cores} cores). This step will be slower "
                "than usual."
            )
    return workers


def _to_gb(num_bytes: int | float) -> float:
    return num_bytes / 1e9
