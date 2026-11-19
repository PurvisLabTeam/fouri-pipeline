import json
from typing import Any

from fouri.app.records import read_runs
from fouri.config import Experiment
from fouri.io import run_record_path


def _line(run: str, step: str, event: str, time: str, wells: list[str] | None, **ended: Any) -> str:
    entry = {"run": run, "step": step, "settings": {}, "wells": wells, "event": event, "time": time}
    return json.dumps({**entry, "pc": "lab-pc", "user": "lab", **ended})


def test_a_bad_run_record_line_is_passed_over(experiment: Experiment) -> None:
    done = {"seconds": 60.0, "outcome": "done", "failed": [], "error": ""}
    lines = [
        _line("20261118T100000", "measure", "started", "2026-11-18T10:00:00", ["B02", "C02"]),
        _line("20261118T100000", "measure", "ended", "2026-11-18T10:01:00", ["B02", "C02"], **done),
        _line("20261118T110000", "measure", "started", "2026-11-18T11:00:00", ["Z99"]),
        _line("20261118T120000", "table", "started", "2026-11-18T12:00:00", None),
        _line("20261118T120000", "table", "ended", "2026-11-18T12:01:00", None, **done),
    ]
    run_record_path(experiment).write_text("\n".join(lines) + "\n", encoding="utf-8")

    runs = read_runs(experiment)

    assert [(run.run, run.wells, run.result) for run in runs] == [
        ("20261118T120000", [], "done"),
        ("20261118T100000", ["B02", "C02"], "done"),
    ]
