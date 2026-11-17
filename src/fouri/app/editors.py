from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fouri.app.runner import RUNS_FOLDER

EDITORS: dict[str, tuple[str, str]] = {
    "segmentation": ("Segmentation", "Check and correct each round's segmentation masks."),
    "alignment": ("Alignment", "Check how each round lines up with the anchor round, and fix it by hand."),
    "artifacts": ("Artifacts", "Paint and grow masks over artifacts, channel by channel."),
}

READY: tuple[str, ...] = ("segmentation", "alignment", "artifacts")

OUTPUT_FOLDER = "editors"
POLL_SECONDS = 0.5
TAIL_BYTES = 65536


def editor_command(editor: str, config: str | Path, well: str) -> list[str]:
    return [sys.executable, "-m", "fouri.viewers", editor, "--config", str(config), "--well", well]


def _output_file() -> Path:
    folder = Path(tempfile.gettempdir()) / RUNS_FOLDER / OUTPUT_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix="editor-", suffix=".log", dir=folder)
    os.close(handle)
    return Path(name)


def _start(command: Sequence[str], output: Path) -> subprocess.Popen[bytes]:
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    options: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT, "env": environment}
    with open(output, "wb") as log:
        if sys.platform != "win32":
            return subprocess.Popen(list(command), stdout=log, start_new_session=True, **options)
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            return subprocess.Popen(
                list(command), stdout=log, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **options
            )
        except OSError:
            return subprocess.Popen(list(command), stdout=log, creationflags=flags, **options)


def _last_line(output: Path) -> str:
    try:
        with open(output, "rb") as log:
            log.seek(max(0, output.stat().st_size - TAIL_BYTES))
            tail = log.read()
    except OSError:
        return ""
    lines = tail.decode("utf-8", errors="replace").splitlines()
    return next((line.strip() for line in reversed(lines) if line.strip()), "")


async def run_editor(command: Sequence[str]) -> tuple[int, str]:
    output = _output_file()
    process = _start(command, output)
    while (code := process.poll()) is None:
        await asyncio.sleep(POLL_SECONDS)
    return code, _last_line(output)
