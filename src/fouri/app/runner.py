"""A run started from the app: its steps one after another, each as its own program.

Each step is started through ``python -m fouri.run`` and reports to a progress file of JSON lines
(see ``fouri.report``). A run's files sit in one folder under the system's temporary folder,
``fouri-app/<run id>/``: ``<step>.jsonl`` per step, its stop file beside it, and everything the
step program printed in ``<step>.log``. Nothing here builds a page: the Run page and the header
read a ``Run``'s progress.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal, cast

from fouri import report
from fouri.run import STEP_NAMES
from fouri.run import command as step_command
from fouri.status import Step

Outcome = Literal["done", "failed", "stopped"]
OUTCOMES = ("done", "failed", "stopped")
RUNS_FOLDER = "fouri-app"


@dataclass(frozen=True)
class StepRequest:
    """One step of a run: the step, its wells and its settings.

    ``images`` is how many images Segment nuclei has to segment, counted when the run was asked
    for, so its time left can go by image; None for every other step, or when it was not counted.
    """

    step: Step
    wells: list[str] | None
    settings: dict[str, Any]
    images: int | None = None


@dataclass
class Bar:
    """One progress bar a step reported, as of its latest line."""

    bar: int
    parent: int | None
    desc: str
    unit: str
    total: float | None
    done: float
    note: str
    closed: bool
    started: float | None  # the time on its open line, None when the line carries no time
    updated: float | None  # the time on its latest line


def time_left(bar: Bar, now: float) -> float | None:
    """Return a bar's seconds left at its rate so far, or None before a rate exists.

    The items left times the seconds per item done so far, less the seconds since the latest line,
    never below 0. A bar with no total, nothing done yet, or closed has no time left.
    """
    if bar.closed or bar.total is None or bar.done <= 0 or bar.started is None or bar.updated is None:
        return None
    per_item = (bar.updated - bar.started) / bar.done
    return max((bar.total - bar.done) * per_item - (now - bar.updated), 0.0)


@dataclass
class StepProgress:
    """What one step of a run has reported so far."""

    step: Step
    bars: dict[int, Bar] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    outcome: Outcome | None = None
    failed: list[str] = field(default_factory=list)
    error: str = ""


def read_progress(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Return the whole lines written after ``offset``, parsed, and the offset past the last."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            written = f.read()
    except FileNotFoundError:
        return [], offset
    end = written.rfind(b"\n") + 1  # a line still being written is read next time
    lines = []
    for raw in written[:end].splitlines():
        try:
            line = json.loads(raw)
        except ValueError:  # not a progress line
            continue
        if isinstance(line, dict):
            lines.append(line)
    return lines, offset + end


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _take(progress: StepProgress, line: dict[str, Any]) -> None:
    """Add one progress line to what the step has reported; a finished line gives its outcome."""
    event = line.get("event")
    if event == "message":
        progress.messages.append(str(line.get("text", "")))
    elif event == "bar" and isinstance(line.get("bar"), int):
        parent = line.get("parent")
        at = _number(line.get("time"))
        earlier = progress.bars.get(line["bar"])
        progress.bars[line["bar"]] = Bar(
            bar=line["bar"],
            parent=parent if isinstance(parent, int) else None,
            desc=str(line.get("desc") or ""),
            unit=str(line.get("unit") or ""),
            total=_number(line.get("total")),
            done=_number(line.get("done")) or 0.0,
            note=str(line.get("note") or ""),
            closed=line.get("action") == "close",
            # a bar's first line is its open line; were it missed, the first line read stands in
            started=at if earlier is None or line.get("action") == "open" else earlier.started,
            updated=at,
        )
    elif event == "finished":
        outcome = line.get("outcome")
        progress.outcome = cast(Outcome, outcome) if outcome in OUTCOMES else "failed"
        progress.failed = [str(item) for item in line.get("failed") or []]
        progress.error = str(line.get("error") or "")


def _quiet() -> dict[str, Any]:
    """Return how to start a step program: no console window on Windows, its own session elsewhere.

    The app has no console, so on Windows each program would otherwise open one. Elsewhere, its own
    session keeps a Ctrl+C typed in the app's terminal from reaching the step, which the app stops
    through its stop file instead.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


class Run:
    """A run: its steps started one after another, each as its own program."""

    run_id: str
    progress: list[StepProgress]

    def __init__(
        self,
        config: str | Path,
        requests: Sequence[StepRequest],
        *,
        folder: Path | None = None,
        command: Callable[..., list[str]] | None = None,
    ) -> None:
        if not requests:
            raise ValueError("A run needs at least one step.")
        self.config = str(config)
        self.requests = list(requests)
        self.run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")  # ISO's compact form, allowed in folder names
        self.folder = folder if folder is not None else Path(tempfile.gettempdir()) / RUNS_FOLDER / self.run_id
        self.progress = [StepProgress(request.step) for request in self.requests]
        self._command = command if command is not None else step_command
        self._current = -1  # index of the step running, or of the last one started
        self._process: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None
        self._offset = 0  # how far the running step's progress file has been read
        self._finished_line: dict[str, Any] | None = None  # kept until the program has exited
        self._stopped = False

    def _progress_file(self, index: int) -> Path:
        return self.folder / f"{self.requests[index].step.value}.jsonl"

    def _log_file(self, index: int) -> Path:
        return self.folder / f"{self.requests[index].step.value}.log"

    def _start_step(self, index: int) -> None:
        request = self.requests[index]
        progress_file, log_file = self._progress_file(index), self._log_file(index)
        for leftover in (progress_file, report.stop_path(progress_file), log_file):
            leftover.unlink(missing_ok=True)
        self._current, self._offset, self._finished_line = index, 0, None
        argv = self._command(
            request.step,
            self.config,
            wells=request.wells,
            settings=request.settings,
            progress=progress_file,
            run=self.run_id,
        )
        environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # the log reads the same on every PC
        try:
            self._log = open(log_file, "wb")  # noqa: SIM115 (closed when the step ends)
            self._process = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT, env=environment, **_quiet()
            )
        except OSError as error:
            self._close_log()
            self.progress[index].outcome = "failed"
            self.progress[index].error = f"{STEP_NAMES[request.step]} could not start: {error}"

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def start(self) -> None:
        """Start the first step's program."""
        if self._current >= 0:
            raise RuntimeError("This run has already started.")
        self.folder.mkdir(parents=True, exist_ok=True)
        self._start_step(0)

    def poll(self) -> None:
        """Read new progress lines, and start the next step when the current one ended done."""
        if self._process is None:
            return
        progress = self.progress[self._current]
        exit_code = self._process.poll()  # asked first, so every line it wrote is on disk when it has ended
        lines, self._offset = read_progress(self._progress_file(self._current), self._offset)
        for line in lines:
            if line.get("event") == "finished":
                self._finished_line = line  # the step still writes its run record line after this one
            else:
                _take(progress, line)
        if exit_code is None:
            return
        self._process = None
        self._close_log()
        if self._finished_line is not None:  # a step has an outcome only once its program has ended
            _take(progress, self._finished_line)
        else:  # it ended without saying how: killed, or crashed before reporting
            progress.outcome = "failed"
            progress.error = (
                f"{STEP_NAMES[progress.step]} ended unexpectedly (exit code {exit_code}). "
                f"What it printed is in {self._log_file(self._current)}"
            )
        # a failed or stopped step ends the run, and so does Stop
        if progress.outcome == "done" and not self._stopped and self._current + 1 < len(self.requests):
            self._start_step(self._current + 1)

    def stop(self) -> None:
        """Create the running step's stop file."""
        self._stopped = True
        if self._process is not None:
            report.stop_path(self._progress_file(self._current)).touch()

    @property
    def finished(self) -> bool:
        """Return whether no step is running and none is left to start."""
        return self._current >= 0 and self._process is None
