"""@TODO."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import napari

from fouri.config import load_config
from fouri.viewers.alignment import AlignmentViewer
from fouri.viewers.base import BaseViewer

EDITORS: dict[str, type[BaseViewer]] = {"alignment": AlignmentViewer}


def main(argv: Sequence[str] | None = None) -> int:
    """Open a napari editor on one well of the experiment and wait until the window closes."""
    parser = argparse.ArgumentParser(
        "python -m fouri.viewers", description="Open the napari viewers for the fouri steps"
    )
    parser.add_argument("editor", choices=EDITORS.keys())
    parser.add_argument("--config", help="The experiment's experiment.json file", required=True)
    parser.add_argument("--well", required=True)
    parser.add_argument("--round", required=False)

    args = parser.parse_args(argv)
    exp = load_config(Path(args.config), announce=False)
    try:
        wells = exp.pick_wells(args.well)
    except ValueError as e:
        parser.error(str(e))
    well = wells[0]
    EDITORS[args.editor](exp, well, args.round)
    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
