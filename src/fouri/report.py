"""Messages and progress bars, printed to the terminal or written to the app's progress file.

`say` prints a message and `progress` draws a tqdm bar. After `report_to(path)`, each message and
each bar's opening, progress, and closing are written to `path` as JSON lines instead. A bar
writes a `move` line at most every `MOVE_EVERY` seconds, and when the bar reaches its total. A
step run by the app ends the file with `finished`.

The app stops a step by creating the file `stop_path(path)`. The next `move` line a bar writes
after that raises `Stopped`.

Notes
-----
The lines of the progress file, `bar` being the bar's number and `parent` the number of the bar
open around it:

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


class Stopped(Exception):  # noqa: N818
    """Exception raised when the app stops a step."""


class StepFailed(RuntimeError):  # noqa: N818
    """Exception raised when items of a step failed, with the names of the failed items.

    Parameters
    ----------
    message : str
        The message stated for the step, naming each failed item.
    failed : Sequence[str]
        The names of the failed items.

    Attributes
    ----------
    failed : list[str]
        The names of the failed items.
    """

    failed: list[str]

    def __init__(self, message: str, failed: Sequence[str]) -> None:
        self.failed = list(failed)
        super().__init__(message)


def stop_path(progress_file: str | Path) -> Path:
    """Return the stop file for a progress file: the same path with `.stop` appended to the name."""
    progress_file = Path(progress_file)
    return progress_file.with_name(progress_file.name + ".stop")


def report_to(path: str | Path | None) -> None:
    """Write messages and bars to `path` as JSON lines from now on. None falls back to printing."""
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
    """Print a message, or write the message to the progress file after `report_to`."""
    if _target is None:
        tqdm.write(text)
    else:
        _write({"event": "message", "text": text})


def say_skipped(done: str, skipped: list[str]) -> None:
    """Report the items skipped as already done, and how to redo them, unless `skipped` is empty."""
    if not skipped:
        return
    say(f"Skipped, already {done}: {', '.join(skipped)}. To redo them, run with overwrite=True")


def finished(outcome: Literal["done", "failed", "stopped"], *, failed: Sequence[str] = (), error: str = "") -> None:
    """Write the last line of the progress file: the outcome, the failed items, and the error."""
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
    """Open a progress bar, drawn by tqdm or written to the progress file after `report_to`.

    Iterating the bar iterates `items` and moves the bar. After `report_to`, the bar writes `open`,
    `move`, and `close` lines instead of drawing, nested under the bar open when it is created.

    Parameters
    ----------
    items : Iterable[Any] | None, default=None
        The items to iterate with the bar. None for a bar moved by `update`.
    desc : str | None, default=None
        The label of the bar.
    unit : str, default="it"
        The noun for one step of the bar (e.g., `well`).
    total : float | None, default=None
        The number of steps at completion. None for the length of `items`, when `items` has one.
    leave : bool, default=True
        Whether a drawn bar stays on screen when closed.

    Returns
    -------
    tqdm[Any]
        The bar, closed at the end of a `with` block or of the iteration.
    """
    if _target is None:
        return tqdm(items, desc=desc, total=total, leave=leave, unit=unit)
    return _FileBar(items, desc=desc, total=total, leave=leave, unit=unit)
