"""The 4i app: the pipeline's pages in their own window.

For now one placeholder page, so the launcher and the desktop shortcut can be built and tried
before the pages exist. ``main`` is what the ``fouri-pipeline`` launcher calls.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _home() -> None:
    from nicegui import ui

    ui.label("4i pipeline").classes("text-2xl font-semibold")
    ui.label("Placeholder: the app's pages are not built yet.")


def main(argv: Sequence[str] | None = None) -> None:
    """Start the app in its own window, or in a browser tab with ``--browser``."""
    parser = argparse.ArgumentParser(prog="fouri-pipeline", description="The Purvis lab 4i pipeline app.")
    parser.add_argument("--browser", action="store_true", help="open in a browser tab instead of a window")
    args = parser.parse_args(argv)

    from nicegui import ui

    # only this PC can reach the app, not the rest of the network
    if args.browser:
        ui.run(_home, title="4i pipeline", reload=False, host="127.0.0.1", port=8080)
    else:
        ui.run(_home, title="4i pipeline", reload=False, host="127.0.0.1", native=True, window_size=(1400, 900))
