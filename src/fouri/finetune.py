"""@TODO."""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy.typing as npt
import scipy.ndimage as ndi
import tifffile

from fouri.config import Experiment
from fouri.io import alignment_paths, label_path, original_label_path
from fouri.models import KeptModel, models_dir, shown_name
from fouri.report import progress, say

if TYPE_CHECKING:
    from tqdm.auto import tqdm

SQUARE_SIZE: Final[int] = 300


@dataclass(frozen=True)
class Square:
    well: str
    round_index: int
    rows: slice
    columns: slice


class _TrainingLog(logging.Handler):
    def __init__(self, bar: tqdm[Any], squares: int) -> None:
        super().__init__()
        self.bar = bar
        self.squares = squares
        self.epoch_line = False

    def emit(self, record: logging.LogRecord) -> None:
        text = record.getMessage()
        epoch, _, rest = text.partition(",")
        if epoch.isdigit() and "train_loss=" in rest:
            self.bar.update(int(epoch) + 1 - self.bar.n)
            self.epoch_line = True

        elif "removing from train set" in text:
            removed = int(text.split()[0])
            fewest = text.partition("(")[2].partition(")")[0]
            if removed == self.squares:
                raise ValueError(f"Every square has fewer than {fewest} nuclei, which is the minimum CellPose allows.")
            say(f"Cellpose excluded {removed} of {self.squares} squares with less than {fewest} nuclei.")


def training_squares(exp: Experiment, wells: Sequence[str]) -> list[Square]:
    """@TODO."""
    squares = []
    for well in wells:
        for round_index, _ in enumerate(alignment_paths(exp, well)):
            orig_path = original_label_path(exp, well, round_index)
            if not orig_path.exists():
                continue

            saved = tifffile.imread(label_path(exp, well, round_index))
            original = tifffile.imread(orig_path)

            differences, _ = ndi.label(saved != original)
            bboxes = ndi.find_objects(differences)
            squares.extend(
                [
                    Square(
                        well=well,
                        round_index=round_index,
                        rows=_square_span(box[0], saved.shape[0]),
                        columns=_square_span(box[1], saved.shape[1]),
                    )
                    for box in bboxes
                    if isinstance(box, tuple)
                ]
            )

    return squares


def finetune_cellpose(exp: Experiment, n_epochs: int = 100, wells: Sequence[str] | None = None) -> Path:
    """@TODO."""
    if n_epochs < 1:
        raise ValueError(f"n_epochs must be >= 1; got {n_epochs}")

    wells = exp.pick_wells(wells)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU not detected, which is required for fine-tuning.")

    squares = training_squares(exp, wells)
    if squares == []:
        raise ValueError(f"No hand-edited segmentation masks exist for: {', '.join(wells)}.")

    wells_rounds_to_train: defaultdict[str, set[int]] = defaultdict(set)
    grouped: defaultdict[tuple[str, int], list[Square]] = defaultdict(list)
    for square in squares:
        wells_rounds_to_train[square.well].add(square.round_index)
        grouped[(square.well, square.round_index)].append(square)

    trained = {well: sorted(rounds) for well, rounds in wells_rounds_to_train.items()}
    missing = [well for well in wells if well not in trained]
    if missing:
        say(f"No corrections found for wells: {', '.join(missing)}, skipping...")

    say(
        f"Fine-tuning on {len(squares)} squares across {sum(len(rounds) for rounds in trained.values())} "
        f"rounds and {len(trained)} wells ({', '.join(trained.keys())})"
    )

    train_data: list[npt.NDArray[Any]] = []
    train_labels: list[npt.NDArray[Any]] = []
    for square_group, grouped_squares in grouped.items():
        well, round_index = square_group
        image = tifffile.imread(alignment_paths(exp, well)[round_index])
        label = tifffile.imread(label_path(exp, well, round_index))

        train_data.extend(image[square.rows, square.columns].copy() for square in grouped_squares)
        train_labels.extend(label[square.rows, square.columns].copy() for square in grouped_squares)

    created = dt.datetime.now().replace(microsecond=0)
    created_name = f"{created:%Y%m%dT%H%M%S}"
    final_dir = models_dir() / created_name
    if final_dir.exists():
        raise FileExistsError(f"A model with the name {created_name} already exists or is being made on this PC.")
    partial = final_dir.with_suffix(".partial")
    partial.mkdir(parents=True, exist_ok=False)
    try:
        from cellpose import models, train

        model = models.CellposeModel(gpu=True)
        with progress(desc="Fine-tuning", unit="epoch", total=n_epochs) as bar:
            handler = _TrainingLog(bar, len(squares))
            logger = logging.getLogger("cellpose.train")
            old_level = logger.level
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            try:
                train.train_seg(
                    model.net, train_data, train_labels, save_path=partial, model_name="model", n_epochs=n_epochs
                )
                bar.update(n_epochs - bar.n)
                if handler.epoch_line is False:
                    say(
                        "Cellpose's training log was not recognized, so the bar did not move and Stop does not work. "
                        f"The model was still trained and kept (cellpose v{metadata.version('cellpose')})."
                    )
            finally:
                logger.removeHandler(handler)
                logger.setLevel(old_level)

        model_json = {
            "created": created.isoformat(),
            "experiment_key": exp.key,
            "config": exp.config_path,
            "wells": trained,
            "squares": len(squares),
            "n_epochs": n_epochs,
        }
        with open(partial / "model.json", "w", encoding="utf-8") as fh:
            json.dump(model_json, fh, indent=2)

        partial.rename(final_dir)

    except BaseException:
        shutil.rmtree(partial)
        raise

    kept = KeptModel(
        name=created_name, created=created, experiment_key=exp.key, wells=trained, path=final_dir / "models" / "model"
    )

    say(f"Kept the model as {shown_name(kept)}.")

    return final_dir


def _square_span(box: slice, size: int) -> slice:
    center = (box.stop + box.start) // 2
    start = max(0, min(center - SQUARE_SIZE // 2, size - SQUARE_SIZE))
    stop = start + SQUARE_SIZE

    return slice(start, stop)
