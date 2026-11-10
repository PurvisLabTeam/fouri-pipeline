"""Whether this PC has a GPU for Segment nuclei, and how long segmenting takes without one.

The check runs in a program of its own, so the app never loads PyTorch.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

from fouri.config import Experiment
from fouri.status import Step, step_status

CPU_MINUTES_PER_IMAGE = 20
# prints the GPU's name, or an empty line when PyTorch finds none
GPU_PROGRAM = "from fouri.segment import gpu_name; print(gpu_name() or '')"


def _no_window() -> dict[str, Any]:
    """Return how to start the check so that, on Windows, it opens no console window."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


async def check_gpu(command: Sequence[str] | None = None) -> str | None:
    """Run ``fouri.segment.gpu_name`` in its own program; return the GPU's name, or None."""
    argv = list(command) if command is not None else [sys.executable, "-c", GPU_PROGRAM]
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            **_no_window(),
        )
    except (OSError, NotImplementedError):  # no such program, or an event loop that can't start one
        return None
    try:
        output, _ = await process.communicate()
    except asyncio.CancelledError:  # the app is closing: don't leave the check running
        process.kill()
        raise
    if process.returncode != 0:
        return None
    # the name is the last line printed; anything PyTorch printed before it is ignored
    lines = output.decode("utf-8", errors="replace").splitlines()
    name = lines[-1].strip() if lines else ""
    return name or None


def images_to_segment(exp: Experiment, wells: Sequence[str], *, overwrite: bool) -> int:
    """Count the images Segment nuclei would segment in these wells."""
    if not wells:
        return 0
    status = step_status(exp, Step.SEGMENT, wells)
    return sum(well.total if overwrite else well.total - well.done for well in status.values())


def cpu_minutes(images: int) -> int:
    """Return the minutes segmenting ``images`` images takes on the CPU."""
    return images * CPU_MINUTES_PER_IMAGE
