"""@TODO."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from magicgui.widgets import PushButton, RadioButtons, Widget
from napari.layers import Image, Points
from napari.layers.base._base_constants import ActionType
from scipy.spatial.distance import pdist
from skimage.measure import regionprops_table

from fouri.align import fit_hand_alignment, round_index_of, save_manual_transform, transform_to_napari
from fouri.io import load_transforms, read_labels
from fouri.viewers.base import BaseViewer

MIN_PAIRS = 3  # The minimum number of cell alignment pairs to click before fitting is allowed
MIN_SPREAD = 1 / 3  # How far apart the farthest clicks must be as a fraction of the image's shortest side

# Messages for when users click on the anchor or round layer
CLICK_ROUND = "Now click the same cell in the round to fix (green)."
CLICK_ANCHOR = "Click another cell in the anchor round (magenta)."

SAVED_ALIGNMENT = "Saved alignment"
NO_ALIGNMENT = "No alignment"

KEY_FIT = "Shift+F"


class AlignmentViewer(BaseViewer):
    title = "Hand alignment"

    anchor_layer: Image
    anchor_clicks: Points
    anchor_labels: npt.NDArray[Any]
    anchor_centers: npt.NDArray[np.float64]
    round_index: int
    round_layer: Image
    round_clicks: Points
    round_labels: npt.NDArray[Any]
    round_centers: npt.NDArray[np.float64]
    fitted_pairs: int
    min_spread: float
    start_from: RadioButtons

    def fit(self) -> None:
        next_layer, next_click = self._next_turn()
        complete_pairs = min(len(self.anchor_clicks.data), len(self.round_clicks.data))
        if complete_pairs < MIN_PAIRS:
            self._wait_for(
                next_layer, f"Fit needs at least {MIN_PAIRS} pairs, you have {complete_pairs}. {next_click}"
            )
            return

        spread = pdist(self.anchor_clicks.data[:complete_pairs]).max()
        if spread < self.min_spread:
            message = (
                f"Spread your clicks out: the cells you clicked are at most {spread:,.0f} pixels apart, "
                f"and at least two need to be {self.min_spread:,.0f} pixels apart. {next_click}"
            )
            self._wait_for(next_layer, message)
            return

        result = fit_hand_alignment(self.anchor_clicks.data, self.round_clicks.data)
        if result is None:
            message = (
                f"These {complete_pairs} pairs could not be fitted. Check that each pair is "
                f"the same cell in both rounds. {next_click}"
            )
            self._wait_for(next_layer, message)
            return

        matrix, mismatch = result
        self.round_layer.affine = matrix
        self.round_clicks.affine = matrix
        self.fitted_pairs = complete_pairs

        message = (
            f"{complete_pairs} pairs fitted, cells are {mismatch:.1f} pixels apart on average. "
            f"Add pairs and press Fit again, or press Save when the cells line up. {next_click}"
        )
        self._wait_for(next_layer, message)

    def undo(self) -> None:
        anchor_counts = len(self.anchor_clicks.data)
        round_counts = len(self.round_clicks.data)
        success_prefix = "Removed the last click."
        if anchor_counts > round_counts:
            self.anchor_clicks.data = self.anchor_clicks.data[:-1]
            self._wait_for(self.anchor_clicks, f"{success_prefix} {CLICK_ANCHOR}")
            return

        if anchor_counts == round_counts and anchor_counts > 0:
            self.round_clicks.data = self.round_clicks.data[:-1]
            self._wait_for(self.round_clicks, f"{success_prefix} Press Fit again to refit without it. {CLICK_ROUND}")
            return

        message = f"There are no clicks to undo. {CLICK_ANCHOR}"
        self._wait_for(self.anchor_clicks, message)

    def save(self) -> None:
        next_layer, next_click = self._next_turn()
        if self.fitted_pairs == 0:
            self._wait_for(next_layer, f"Press Fit before saving; nothing has been fitted yet. {next_click}")
            return
        save_manual_transform(
            self.exp, self.well, self.round_index, np.asarray(self.round_layer.affine, dtype=np.float64)
        )
        self._wait_for(next_layer, f"Saved the fit from {self.fitted_pairs} pairs of cells. {next_click}")

    def _build_layers(self) -> None:
        clicks = {"size": 40, "canvas_size_limits": (12, 30), "ndim": 2}
        # 1 x 1 placeholders until _load_well and _load_round put a well's images in
        placeholder = np.zeros((1, 1), dtype=bool)

        self.anchor_layer = self.viewer.add_image(
            placeholder, blending="additive", colormap="magenta", name="Anchor Round"
        )
        self.round_layer = self.viewer.add_image(placeholder, blending="additive", colormap="green", name="Round")
        self.anchor_clicks = self.viewer.add_points(name="Anchor clicks", face_color="yellow", **clicks)
        self.round_clicks = self.viewer.add_points(name="Round clicks", face_color="cyan", **clicks)
        self.anchor_clicks.events.data.connect(self._after_click)
        self.round_clicks.events.data.connect(self._after_click)

    def _tool_widgets(self) -> list[Widget]:
        self.start_from = RadioButtons(choices=[SAVED_ALIGNMENT, NO_ALIGNMENT], label="Start from")
        self.start_from.value = SAVED_ALIGNMENT
        self.start_from.changed.connect(self._restart_round)

        fit_button = PushButton(text=f"Fit ({KEY_FIT})")
        fit_button.changed.connect(self.fit)

        return [self.start_from, fit_button]

    def _extra_keys(self) -> dict[str, Callable[[], None]]:
        return {KEY_FIT: self.fit}

    def _round_choices(self, round_table: pd.DataFrame) -> list[str]:
        choices = [
            str(row.round_name)
            for row in round_table[round_table["is_align_channel"]].sort_values(by="round_index").itertuples()
            if row.round_index != self.exp.anchor_round
        ]
        if not choices:
            raise ValueError("There are no rounds to align besides the anchor round.")
        return choices

    def _load_well(self, well: str, round_table: pd.DataFrame) -> None:
        anchor_labels = read_labels(self.exp.segmented, well, [self.exp.anchor_round])[0]
        anchor_centers = _cell_centers(anchor_labels)

        same_size = self.anchor_layer.data.shape == anchor_labels.shape
        self.anchor_labels = anchor_labels
        self.anchor_centers = anchor_centers
        self.min_spread = MIN_SPREAD * min(anchor_labels.shape)
        self.anchor_layer.data = anchor_labels > 0
        if not same_size:
            # first well, or a montage of another size: fit it in the window, else keep the zoom
            self.viewer.reset_view()

    def _load_round(self, round_name: str) -> None:
        first_message = "Click a cell in the anchor round (magenta)."
        round_index = round_index_of(self.round_table, round_name, self.exp.anchor_round, self.well)
        round_labels = read_labels(self.exp.segmented, self.well, [round_index])[0]
        round_centers = _cell_centers(round_labels)
        try:
            transforms = load_transforms(self.exp, self.well)
        except FileNotFoundError:
            transforms = None

        saved = None
        if transforms is not None and not np.isnan(transforms[round_index]).any():
            saved = transform_to_napari(transforms[round_index])
        start_position = saved if self.start_from.value == SAVED_ALIGNMENT and saved is not None else np.eye(3)

        if self.start_from.value == SAVED_ALIGNMENT and saved is None:
            first_message = f"{round_name} has no saved alignment yet, so it starts unaligned. " + first_message

        self.fitted_pairs = 0
        self.round_index = round_index
        self.round_labels = round_labels
        self.round_centers = round_centers

        self.anchor_clicks.data = np.empty((0, 2))
        self.round_clicks.data = np.empty((0, 2))
        self.round_layer.data = round_labels > 0
        self.round_layer.name = round_name
        self.round_layer.affine = start_position
        self.round_clicks.affine = start_position

        self._wait_for(self.anchor_clicks, first_message)

    def _restart_round(self) -> None:
        self.show_round(self.round_name)

    def _after_click(self, event: Any) -> None:
        if event.action != ActionType.ADDED:
            return
        labels, centers = (
            (self.anchor_labels, self.anchor_centers)
            if event.source is self.anchor_clicks
            else (self.round_labels, self.round_centers)
        )
        is_cell = _snap_to_cell(event.source, labels, centers)
        if not is_cell:
            message = (
                "That wasn't inside a cell. Click inside a cell in the anchor round (magenta)."
                if event.source is self.anchor_clicks
                else "That wasn't inside a cell. Click the same cell in the round to fix (green)."
            )
            self._wait_for(event.source, message)
            return
        if event.source is self.anchor_clicks:
            self._wait_for(self.round_clicks, CLICK_ROUND)
            return

        self._wait_for(
            self.anchor_clicks,
            f"{len(self.round_clicks.data)} pairs selected. {CLICK_ANCHOR[:-1]} or press Fit.",
        )

    def _wait_for(self, layer: Any, message: str) -> None:
        self.viewer.layers.selection.active = layer
        layer.mode = "add"
        self.prompt(message)

    def _next_turn(self) -> tuple[Points, str]:
        return (
            (self.round_clicks, CLICK_ROUND)
            if len(self.anchor_clicks.data) > len(self.round_clicks.data)
            else (self.anchor_clicks, CLICK_ANCHOR)
        )


def _cell_centers(labels: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    centroids = regionprops_table(labels, properties=("label", "centroid"))
    centers = np.full((labels.max() + 1, 2), np.nan)
    centers[centroids["label"]] = np.stack([centroids["centroid-0"], centroids["centroid-1"]], axis=1)

    return centers


def _snap_to_cell(points: Points, labels: npt.NDArray[Any], centers: npt.NDArray[np.float64]) -> bool:
    row, col = np.round(points.data[-1, :], decimals=0).astype(int)
    if row < 0 or col < 0 or row >= labels.shape[0] or col >= labels.shape[1]:
        points.data = points.data[:-1]
        return False

    label = labels[row, col]
    if label == 0:
        points.data = points.data[:-1]
        return False

    new = points.data.copy()
    new[-1] = centers[label]
    points.data = new
    return True
