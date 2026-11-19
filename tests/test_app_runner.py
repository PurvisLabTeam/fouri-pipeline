import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fouri.app.runner import Run, StepRequest, read_progress
from fouri.status import Step

STEP_PROGRAM = """\
import json
import sys

progress, outcome, ending, code = sys.argv[1:]
finished = {"time": 0.0, "event": "finished", "outcome": outcome, "failed": [], "error": ""}
with open(progress, "a", encoding="utf-8") as f:
    f.write(json.dumps({"time": 0.0, "event": "message", "text": "Measured B02."}) + "\\n")
    f.write(json.dumps(finished) + ("\\n" if ending == "whole" else ""))
sys.exit(int(code))
"""
REQUESTS = [StepRequest(Step.MEASURE, ["B02"], {}), StepRequest(Step.TABLE, None, {})]


def _step_command(
    folder: Path, endings: dict[Step, tuple[str, str, int]]
) -> tuple[Callable[..., list[str]], list[Step]]:
    program = folder / "step_program.py"
    program.write_text(STEP_PROGRAM, encoding="utf-8")
    started: list[Step] = []

    def command(step: Step, config: Any, *, progress: Path, **options: Any) -> list[str]:
        started.append(step)
        outcome, ending, code = endings[step]
        return [sys.executable, str(program), str(progress), outcome, ending, str(code)]

    return command, started


def _wait(run: Run) -> None:
    deadline = time.monotonic() + 60
    while not run.finished:
        assert time.monotonic() < deadline
        run.poll()
        time.sleep(0.02)


@pytest.mark.parametrize(("outcome", "code"), [("failed", 1), ("stopped", 3)])
def test_a_failed_or_stopped_step_ends_the_run(tmp_path: Path, outcome: str, code: int) -> None:
    command, started = _step_command(
        tmp_path, {Step.MEASURE: (outcome, "whole", code), Step.TABLE: ("done", "whole", 0)}
    )
    run = Run(tmp_path / "experiment.json", REQUESTS, folder=tmp_path / "run", command=command)

    run.start()
    _wait(run)

    assert started == [Step.MEASURE]
    assert [progress.outcome for progress in run.progress] == [outcome, None]


def test_a_half_written_finished_line_is_not_read(tmp_path: Path) -> None:
    command, started = _step_command(tmp_path, {Step.MEASURE: ("done", "half", 0), Step.TABLE: ("done", "whole", 0)})
    run = Run(tmp_path / "experiment.json", REQUESTS, folder=tmp_path / "run", command=command)

    run.start()
    _wait(run)

    assert run.progress[0].messages == ["Measured B02."]
    assert run.progress[0].outcome == "failed"
    assert started == [Step.MEASURE]
    progress_file = tmp_path / "run" / "measure.jsonl"
    lines, offset = read_progress(progress_file, 0)
    assert [line["event"] for line in lines] == ["message"]
    assert offset == progress_file.read_bytes().index(b"\n") + 1
