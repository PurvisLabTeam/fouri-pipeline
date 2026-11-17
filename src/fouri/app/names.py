"""Column names as a lab member says them: a marker and a measure, "cMyc · nucleus mean".

The cell pages offer features by these names, and the plots use them as axis titles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from fouri.adata import POSITION_COLUMNS
from fouri.metadata import feature_prefix

# the measures Measure cells makes for every channel, by the end of the column name
MEASURES = {
    "nuc_mean": "nucleus mean",
    "ring_mean": "ring mean",
    "total_nuc_protein": "nucleus total",
    "nuc_cyto_ratio": "nucleus / ring",
}
VALUE = "value"  # the one measure of a shape or derived feature

# columns that number or place a cell rather than measure it
NOT_FEATURES = frozenset({"label", *POSITION_COLUMNS})


@dataclass(frozen=True)
class Feature:
    """A number column named for the pages: a marker and a measure."""

    column: str
    marker: str
    measure: str

    @property
    def title(self) -> str:
        """Return the axis title, ``cMyc · nucleus mean``, or the marker alone for a value."""
        if self.measure == VALUE:
            return self.marker
        return f"{self.marker} · {self.measure}"


def _signals(row: pd.Series, channels: Sequence[str]) -> list[str]:
    """Return the signals a round of info_exp.csv names, leaving out its empty channels."""
    return [str(row[c]).strip() for c in channels if pd.notna(row[c]) and str(row[c]).strip()]


def _markers(info: pd.DataFrame) -> dict[str, str]:
    """Return each channel's column prefix and marker: the signal, with its round if in several."""
    channels = list(info.columns[1:])
    rounds: dict[str, list[str]] = {}
    for _, row in info.iterrows():
        for signal in _signals(row, channels):
            rounds.setdefault(signal, [])
            if str(row["round_name"]) not in rounds[signal]:
                rounds[signal].append(str(row["round_name"]))
    markers = {}
    for signal, names in rounds.items():
        for round_name in names:
            markers[feature_prefix(round_name, signal)] = signal if len(names) == 1 else f"{signal} ({round_name})"
    return markers


def describe(columns: Sequence[str], info: pd.DataFrame) -> dict[str, Feature]:
    """Name every number column from the rounds and signals in info_exp.csv.

    ``info`` is info_exp.csv as ``fouri.io.read_info_file`` reads it. A channel column, a round and
    signal's prefix followed by one of ``MEASURES``, is named by its marker and measure; a marker
    stained in several rounds keeps its round, ``DNA (R0)``. Any other column is a shape column or a
    derived feature, named by itself with the measure ``value``. ``label`` and the position columns
    are left out, since they do not measure the cell.
    """
    markers = _markers(info)
    features = {}
    for column in columns:
        if column in NOT_FEATURES:
            continue
        features[column] = Feature(column, column, VALUE)
        for measure, words in MEASURES.items():
            prefix = column.removesuffix(f"_{measure}")
            if prefix != column and prefix in markers:
                features[column] = Feature(column, markers[prefix], words)
                break
    return features


def _panel(info: pd.DataFrame) -> list[str]:
    markers = _markers(info)
    channels = list(info.columns[1:])
    order: list[str] = []
    for _, row in info.iterrows():
        for signal in _signals(row, channels):
            marker = markers[feature_prefix(str(row["round_name"]), signal)]
            if marker not in order:
                order.append(marker)
    return order


def marker_menu(features: Mapping[str, Feature], info: pd.DataFrame) -> dict[str, dict[str, str]]:
    columns = {(feature.marker, feature.measure): column for column, feature in features.items()}
    menu: dict[str, dict[str, str]] = {}
    for marker in _panel(info):
        measures = {words: columns[marker, words] for words in MEASURES.values() if (marker, words) in columns}
        if measures:
            menu[marker] = measures
    for column, feature in features.items():
        if feature.measure == VALUE and feature.marker not in menu:
            menu[feature.marker] = {VALUE: column}
    return menu


def menu_order(features: dict[str, Feature], info: pd.DataFrame) -> dict[str, str]:
    menu = marker_menu(features, info)
    columns = [column for measures in menu.values() for column in measures.values()]
    columns += [column for column in features if column not in columns]
    return {column: features[column].title for column in columns}
