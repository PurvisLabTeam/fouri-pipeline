from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any, Literal

from fouri.config import Experiment
from fouri.io import run_record_path
from fouri.metadata import plate_sort_key
from fouri.run import STEP_NAMES
from fouri.status import Step

Result = Literal["done", "failed", "stopped", "running", "did not finish"]
EVENTS = ("started", "ended")
STEPS = {step.value for step in Step}
CPU_NOTE = "segmentation on the CPU"
INDENT = "    "


@dataclass(frozen=True)
class RunRow:
    run: str | None
    started: str
    steps: list[str]
    wells: list[str]
    result: Result
    seconds: float
    notes: list[str]


@dataclass
class _StepLines:
    step: str
    pc: Any
    user: Any
    started: dict[str, Any] | None = None
    ended: dict[str, Any] | None = None


def _parse(text: str) -> dict[str, Any] | None:
    try:
        line = json.loads(text)
    except ValueError:
        return None
    if not isinstance(line, dict) or line.get("event") not in EVENTS or line.get("step") not in STEPS:
        return None
    try:
        dt.datetime.fromisoformat(str(line.get("time")))
    except ValueError:
        return None
    run = line.get("run")
    if run is not None and not isinstance(run, str):
        return None
    return line


def _read_lines(exp: Experiment) -> list[dict[str, Any]]:
    path = run_record_path(exp)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        parsed = (_parse(text) for text in f if text.strip())
        return [line for line in parsed if line is not None]


def _open_match(candidates: list[_StepLines], line: dict[str, Any]) -> _StepLines | None:
    for lines in reversed(candidates):
        if lines.ended is None and (lines.step, lines.pc, lines.user) == (
            line["step"],
            line.get("pc"),
            line.get("user"),
        ):
            return lines
    return None


def _group(record: list[dict[str, Any]]) -> list[tuple[str | None, list[_StepLines]]]:
    runs: list[tuple[str | None, list[_StepLines]]] = []
    by_id: dict[str, list[_StepLines]] = {}
    unnamed: list[_StepLines] = []
    for line in record:
        run = line.get("run")
        candidates = by_id.get(run, []) if run is not None else unnamed
        match = _open_match(candidates, line) if line["event"] == "ended" else None
        if match is not None:
            match.ended = line
            continue
        lines = _StepLines(line["step"], line.get("pc"), line.get("user"))
        if line["event"] == "started":
            lines.started = line
        else:
            lines.ended = line
        if run is None:
            unnamed.append(lines)
            runs.append((None, [lines]))
        elif run in by_id:
            by_id[run].append(lines)
        else:
            by_id[run] = [lines]
            runs.append((run, by_id[run]))
    return runs


def _start(steps: list[_StepLines]) -> str:
    starts = [str(lines.started["time"]) for lines in steps if lines.started is not None]
    if starts:
        return min(starts)
    ends = []
    for lines in steps:
        assert lines.ended is not None
        seconds = _seconds(lines.ended)
        moment = dt.datetime.fromisoformat(str(lines.ended["time"])) - dt.timedelta(seconds=seconds)
        ends.append(moment.isoformat(timespec="seconds"))
    return min(ends)


def _seconds(line: dict[str, Any]) -> float:
    value = line.get("seconds")
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _first(lines: _StepLines) -> dict[str, Any]:
    line = lines.started if lines.started is not None else lines.ended
    assert line is not None
    return line


def _wells(steps: list[_StepLines]) -> list[str]:
    wells: set[str] = set()
    for lines in steps:
        listed = _first(lines).get("wells")
        if isinstance(listed, list):
            wells.update(str(well) for well in listed)
    return sorted(wells, key=plate_sort_key)


def _result(run: str | None, steps: list[_StepLines], running: str | None) -> Result:
    outcomes = [lines.ended.get("outcome") for lines in steps if lines.ended is not None]
    if "failed" in outcomes:
        return "failed"
    if "stopped" in outcomes:
        return "stopped"
    if any(lines.ended is None for lines in steps):
        return "running" if run is not None and run == running else "did not finish"
    return "done"


def _on_cpu(lines: _StepLines) -> bool:
    settings = _first(lines).get("settings")
    return lines.step == Step.SEGMENT.value and isinstance(settings, dict) and settings.get("allow_cpu") is True


def _table_setting(lines: _StepLines, key: str) -> list[str]:
    settings = _first(lines).get("settings")
    if lines.step != Step.TABLE.value or not isinstance(settings, dict):
        return []
    names = settings.get(key)
    return [str(name) for name in names] if isinstance(names, list) else []


def _applied(steps: list[_StepLines], key: str) -> list[str]:
    applied: list[str] = []
    for lines in steps:
        for name in _table_setting(lines, key):
            if name not in applied:
                applied.append(name)
    return applied


def _notes(steps: list[_StepLines]) -> list[str]:
    notes = [CPU_NOTE] if any(_on_cpu(lines) for lines in steps) else []
    filters = _applied(steps, "filters")
    if filters:
        notes.append(f"filters: {', '.join(filters)}")
    labels = _applied(steps, "labels")
    if labels:
        notes.append("\n".join(["labels:", *(f"{INDENT}{name}" for name in labels)]))
    failed: list[str] = []
    for lines in steps:
        listed = lines.ended.get("failed") if lines.ended is not None else None
        for item in listed if isinstance(listed, list) else []:
            if str(item) not in failed:
                failed.append(str(item))
    if failed:
        notes.append(f"failed: {', '.join(failed)}")
    return notes


def clock_text(moment: dt.datetime) -> str:
    return f"{moment:%Y-%m-%d} {moment.hour % 12 or 12}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'}"


def read_runs(exp: Experiment, *, running: str | None = None) -> list[RunRow]:
    rows = [
        RunRow(
            run=run,
            started=_start(steps),
            steps=[STEP_NAMES[Step(lines.step)] for lines in steps],
            wells=_wells(steps),
            result=_result(run, steps, running),
            seconds=round(sum((_seconds(lines.ended) for lines in steps if lines.ended is not None), 0.0), 1),
            notes=_notes(steps),
        )
        for run, steps in _group(_read_lines(exp))
    ]
    order = sorted(range(len(rows)), key=lambda index: (rows[index].started, index), reverse=True)
    return [rows[index] for index in order]
