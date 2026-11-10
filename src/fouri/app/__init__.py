"""The 4i app: the pipeline's pages in their own window.

``main`` is what the ``fouri-pipeline`` launcher calls. Nothing else in fouri imports this
package, and this module imports nicegui only inside ``main``, after pointing its storage at the
user's app-data.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

APP_FOLDER = "4i Pipeline"


def storage_folder() -> Path:
    """Return the folder NiceGUI keeps its storage in, inside the user's app-data."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
    else:
        base = Path.home() / ".local" / "share"
    return base / APP_FOLDER / "nicegui"


def main(argv: Sequence[str] | None = None) -> None:
    """Start the app in its own window, or in a browser tab with ``--browser``."""
    parser = argparse.ArgumentParser(prog="fouri-pipeline", description="The Purvis lab 4i pipeline app.")
    parser.add_argument("--browser", action="store_true", help="open in a browser tab instead of a window")
    args = parser.parse_args(argv)

    # NiceGUI reads the storage folder when it is imported, and creates only the path's last folder
    folder = storage_folder()
    folder.mkdir(parents=True, exist_ok=True)
    os.environ["NICEGUI_STORAGE_PATH"] = str(folder)

    from nicegui import app, ui

    from fouri.app import pages  # noqa: F401  (registers every page's address)
    from fouri.app.state import POLL_SECONDS, find_gpu, poll_run, reopen_last, stop_run

    app.on_startup(reopen_last)
    app.on_startup(find_gpu)  # NiceGUI runs it in the background, so pages open while the check goes on
    # the app follows the run whichever page is open; quitting stops it and keeps finished wells
    app.timer(POLL_SECONDS, poll_run)
    app.on_shutdown(stop_run)

    # only this PC can reach the app, not the rest of the network
    if args.browser:
        ui.run(title="4i pipeline", dark=True, reload=False, host="127.0.0.1", port=8080)
    else:
        app.native.window_args["confirm_close"] = True  # the window asks before it closes
        ui.run(title="4i pipeline", dark=True, reload=False, host="127.0.0.1", native=True, window_size=(1400, 900))
