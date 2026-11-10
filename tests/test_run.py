"""A step run as the app runs it records the wells that failed."""

import json
from pathlib import Path

from builders import aligned, converted, segmented

from fouri import io, run
from fouri.config import Experiment


def test_a_real_step_records_its_failed_wells(experiment: Experiment) -> None:
    for well in ("B02", "B03"):
        converted(experiment, well)
        segmented(experiment, well)
    aligned(experiment, "B02")
    aligned(experiment, "B03", rounds=[])  # every round but the anchor left unaligned

    outcome = run.run_step("measure", experiment.config_path, wells=["B02", "B03"])

    lines = Path(io.run_record_path(experiment)).read_text(encoding="utf-8").splitlines()
    ended = json.loads(lines[-1])
    assert outcome == "failed"
    assert ended["event"] == "ended"
    assert ended["outcome"] == "failed"
    assert ended["failed"] == ["B03"]
