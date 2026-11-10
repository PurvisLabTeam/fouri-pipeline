"""Messages and progress bars, printed as usual or written to the app's progress file.

By default ``say`` prints and ``progress`` draws a tqdm bar. When the app starts a step it first
calls ``report_to(path)``: from then on every message, and every bar opened, moved and closed,
is written as one JSON line to that file, which the app's page reads as the step runs. The step
program ends with ``finished``. Only the main process reports; worker processes print nothing.

To stop a step, the app creates the file ``stop_path(progress file)``; the next time a bar moves it
raises ``Stopped``. Outputs are written whole or not at all, so a stopped step leaves no half files.

The lines look like this::

    {"time": 1731189000.12, "event": "message", "text": "B02: 90 cell(s) dropped."}
    {"time": ..., "event": "bar", "action": "open", "bar": 2, "parent": 1, "desc": "Segmenting B02",
     "unit": "round", "total": 6, "done": 0, "note": ""}
    {"time": ..., "event": "bar", "action": "move", "bar": 2, "parent": 1, ..., "done": 3, ...}
    {"time": ..., "event": "bar", "action": "close", "bar": 2, "parent": 1, ..., "done": 6, ...}
    {"time": ..., "event": "finished", "outcome": "failed", "failed": ["C02"], "error": "..."}
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Never, overload

from tqdm.auto import tqdm

_target: Path | None = None  # the progress file, or None to print
_stop: Path | None = None  # the file whose appearance stops the step
_lock = threading.Lock()
_open_bars: list[int] = []  # bars open now, outermost first: a new bar's parent is the last one
_bar_ids = itertools.count(1)
MOVE_EVERY = 0.25  # seconds between two "move" lines of one bar


class Stopped(Exception):  # noqa: N818 (it reads as what happened)
    """The app asked the step to stop."""


class StepFailed(RuntimeError):  # noqa: N818
    """Some items of a step failed."""

    failed: list[str]

    def __init__(self, message: str, failed: Sequence[str]) -> None:
        self.failed = list(failed)
        super().__init__(message)


def stop_path(progress_file: str | Path) -> Path:
    """Return the file the app creates to stop a step that reports to ``progress_file``."""
    progress_file = Path(progress_file)
    return progress_file.with_name(progress_file.name + ".stop")


def report_to(path: str | Path | None) -> None:
    """Write messages and bars to ``path`` as JSON lines from now on; None goes back to printing."""
    global _target, _stop
    _target = None if path is None else Path(path)
    _stop = None if path is None else stop_path(path)
    _open_bars.clear()


def _write(event: dict[str, Any]) -> None:
    if _target is None:
        return
    line = json.dumps({"time": round(time.time(), 3), **event}, default=str)
    with _lock, open(_target, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def say(text: str) -> None:
    if _target is None:
        tqdm.write(text)
    else:
        _write({"event": "message", "text": text})


def say_skipped(done: str, skipped: list[str]) -> None:
    if not skipped:
        return
    say(f"Skipped, already {done}: {', '.join(skipped)}. To redo them, run with overwrite=True")


def finished(outcome: Literal["done", "failed", "stopped"], *, failed: Sequence[str] = (), error: str = "") -> None:
    """Write the step's last line: how it ended, the items that failed, the error if any."""
    _write({"event": "finished", "outcome": outcome, "failed": list(failed), "error": error})


class _FileBar(tqdm):
    """A tqdm bar that draws nothing and writes its progress to the progress file instead."""

    def __init__(self, items: Any, *, desc: str | None, total: float | None, leave: bool, unit: str) -> None:
        self._id = next(_bar_ids)
        self._parent = _open_bars[-1] if _open_bars else None
        self._last_move = 0.0
        self._reported_close = False
        devnull = open(os.devnull, "w")  # noqa: SIM115 (closed with the bar)
        super().__init__(items, desc=desc, total=total, leave=leave, unit=unit, file=devnull)
        _open_bars.append(self._id)
        self._event("open")

    def _event(self, action: str) -> None:
        _write(
            {
                "event": "bar",
                "action": action,
                "bar": self._id,
                "parent": self._parent,
                "desc": self.desc,
                "unit": self.unit,
                "total": self.total,
                "done": self.n,
                "note": self.postfix or "",
            }
        )

    def update(self, n: float | None = 1) -> bool | None:
        displayed = super().update(n)
        now = time.monotonic()
        if now - self._last_move >= MOVE_EVERY or (self.total is not None and self.n >= self.total):
            self._last_move = now
            self._event("move")
            if _stop is not None and _stop.exists():
                raise Stopped("Stopped from the app.")
        return displayed

    def reset(self, total: float | None = None) -> None:
        super().reset(total)
        self._event("move")

    def close(self) -> None:
        if not self._reported_close and hasattr(self, "_id"):
            self._reported_close = True
            if self._id in _open_bars:
                _open_bars.remove(self._id)
            self._event("close")
        fp = getattr(self, "fp", None)
        super().close()
        if fp is not None and not fp.closed:
            fp.close()


@overload
def progress[T](
    items: Iterable[T],
    *,
    desc: str | None = None,
    unit: str = "it",
    total: float | None = None,
    leave: bool = True,
) -> tqdm[T]: ...
@overload
def progress(
    items: None = None,
    *,
    desc: str | None = None,
    unit: str = "it",
    total: float | None = None,
    leave: bool = True,
) -> tqdm[Never]: ...
def progress(
    items: Iterable[Any] | None = None,
    *,
    desc: str | None = None,
    unit: str = "it",
    total: float | None = None,
    leave: bool = True,
) -> tqdm[Any]:
    if _target is None:
        return tqdm(items, desc=desc, total=total, leave=leave, unit=unit)
    return _FileBar(items, desc=desc, total=total, leave=leave, unit=unit)
