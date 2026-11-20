from __future__ import annotations

import contextlib
import copy
import io
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import anndata as ad
import h5py
import pandas as pd
from anndata.io import read_elem

from fouri.adata import is_current_subsample
from fouri.app.cells import CONDITION_JOIN
from fouri.app.state import save_app_value
from fouri.config import Experiment
from fouri.embedding import EMBED_MEM_PER_WORKER, MIN_RANKING_K, Method, Run
from fouri.embedding_setup import _whole
from fouri.io import adata_path
from fouri.metadata import natural_key, plate_sort_key
from fouri.parallel import worker_count

RECORDS = "embeddings"
SUBSAMPLE = "subsample"
ORIGINAL, COMBAT = "original", "combat"
VALUES_NAMES = {ORIGINAL: "Original", COMBAT: "ComBat-corrected"}

OLD_08 = [
    "R0_DNA_total_nuc_protein",
    "pRB_RB_ratio",
    "R1_CyclinA2_nuc_mean",
    "R0_CDK2_nuc_mean",
    "R3_CyclinE1_nuc_mean",
    "R2_CyclinD1_nuc_mean",
    "R1_CDK4_nuc_mean",
    "R3_p21_nuc_mean",
    "R1_E2F1_nuc_mean",
    "R2_CyclinB1_nuc_mean",
    "R4_CDT1_nuc_mean",
    "R5_cMyc_nuc_mean",
]
DEFAULT_SETUP: dict[str, Any] = {
    "picked": list(OLD_08),
    "samples": [],
    "ranking_k": 100,
    "top_n": [10],
    "methods": {
        "PHATE": {"dims": [3], "settings": {"knn": [80], "t": ["auto"], "gamma": [1], "decay": [40], "n_pca": [100]}},
        "UMAP": {"dims": [], "settings": {"n_neighbors": [15], "min_dist": [0.1], "metric": ["euclidean"]}},
    },
    "skipped": [],
    "seed": 42,
    "values": "original",
}
SECONDS_PER_RUN = {(Method.PHATE, 2): 9.0, (Method.PHATE, 3): 9.0, (Method.UMAP, 2): 15.0, (Method.UMAP, 3): 20.0}
CELLS_TIMED = 8000
METRICS = ("euclidean", "manhattan", "cosine", "correlation")


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class Rule:
    one: str
    many: str
    accepts: Callable[[Any], bool]


_WHOLE_1 = Rule("a whole number, 1 or more", "whole numbers, 1 or more", lambda v: _whole(v) and v >= 1)
RULES: dict[str, Rule] = {
    "top_n": _WHOLE_1,
    "ranking_k": Rule(
        f"a whole number, {MIN_RANKING_K} or more",
        f"whole numbers, {MIN_RANKING_K} or more",
        lambda v: _whole(v) and v >= MIN_RANKING_K,
    ),
    "seed": Rule("a whole number, 0 or more", "whole numbers, 0 or more", lambda v: _whole(v) and v >= 0),
    "knn": _WHOLE_1,
    "n_pca": _WHOLE_1,
    "n_neighbors": Rule("a whole number, 2 or more", "whole numbers, 2 or more", lambda v: _whole(v) and v >= 2),
    "t": Rule(
        "a whole number, 1 or more, or auto",
        "whole numbers, 1 or more, or auto",
        lambda v: v == "auto" or (_whole(v) and v >= 1),
    ),
    "gamma": Rule("a number from -1 to 1", "numbers from -1 to 1", lambda v: _number(v) and -1 <= v <= 1),
    "decay": Rule("a number above 0", "numbers above 0", lambda v: _number(v) and v > 0),
    "min_dist": Rule("a number, 0 or more", "numbers, 0 or more", lambda v: _number(v) and v >= 0),
    "metric": Rule(
        "euclidean, manhattan, cosine or correlation",
        "euclidean, manhattan, cosine or correlation",
        lambda v: v in METRICS,
    ),
}


def _value(text: str) -> int | float | str:
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError:
        return text.lower()
    if not math.isfinite(number):
        return text.lower()
    return int(number) if number.is_integer() else number


def read_values(text: str, name: str) -> tuple[list[Any], str | None]:
    values: list[Any] = []
    wrong = False
    for item in str(text).split(","):
        if not item.strip():
            continue
        value = _value(item)
        if not RULES[name].accepts(value):
            wrong = True
        elif value not in values:
            values.append(value)
    if wrong:
        return values, f"enter {RULES[name].many}."
    if not values:
        return values, "enter at least one value."
    return values, None


def read_number(value: float | None, name: str) -> tuple[int | None, str | None]:
    if value is None:
        return None, "enter a value."
    number = int(value) if float(value).is_integer() else value
    if not RULES[name].accepts(number):
        return None, f"enter {RULES[name].one}."
    return int(number), None


def run_seconds(run: Run, cells: int) -> float:
    return SECONDS_PER_RUN[run.method, run.dims] * cells / CELLS_TIMED


def total_seconds(seconds: Sequence[float], at_once: int) -> float:
    lanes = [0.0] * max(at_once, 1)
    for one in sorted(seconds, reverse=True):
        lanes[lanes.index(min(lanes))] += one
    return max(lanes)


def runs_at_once(count: int) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return int(worker_count(max_count=count, mem_per_worker=EMBED_MEM_PER_WORKER, unit="run"))


def time_text(seconds: float) -> str:
    whole = max(round(seconds), 1)
    hours, rest = divmod(whole, 3600)
    minutes, rest = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min" if minutes else f"{hours} h"
    if minutes:
        return f"{minutes} min {rest} s" if rest else f"{minutes} min"
    return f"{rest} s"


def save_setup(exp: Experiment, setup: Mapping[str, Any]) -> Experiment:
    return save_app_value(exp, "embedding", copy.deepcopy(dict(setup)))


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip()


def sample_number(text: str) -> int | None:
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else None


def sample_names(cells: pd.DataFrame, exp: Experiment) -> dict[int, str]:
    if "sample_ID" not in cells or "well" not in cells:
        return {}
    pairs = cells[["sample_ID", "well"]].astype(str).drop_duplicates()
    first: dict[int, str] = {}
    for text, well in sorted(zip(pairs["sample_ID"], pairs["well"], strict=True), key=lambda p: plate_sort_key(p[1])):
        sample = sample_number(text)
        if sample is not None:
            first.setdefault(sample, well)
    metadata = exp.well_metadata
    wells = set(pairs["well"])
    columns = [c for c in exp.condition_columns if any(not _blank(metadata.get(w, {}).get(c)) for w in wells)]
    names: dict[int, str] = {}
    for sample in sorted(first):
        meta = metadata.get(first[sample], {})
        condition = CONDITION_JOIN.join(str(meta[c]).strip() for c in columns if not _blank(meta.get(c)))
        names[sample] = f"{sample}: {condition}" if condition else str(sample)
    return names


def subsample_is_current(exp: Experiment, subsample: ad.AnnData) -> bool:
    table_uns: dict[str, Any] = {}
    path = adata_path(exp)
    if path.is_file():
        with h5py.File(path, "r") as file:
            if "uns/cell_table" in file:
                table_uns["cell_table"] = read_elem(file["uns/cell_table"])
    return is_current_subsample(table_uns, subsample.uns)


def embedding_keys(adata: ad.AnnData) -> list[str]:
    records = adata.uns.get(RECORDS, {})
    return sorted((key for key in adata.obsm if key in records), key=natural_key)


def _name(record: Mapping[str, Any]) -> str:
    parts = [f"{record['method']} {int(record['dims'])}D", f"top {int(record['top_n'])} features"]
    parts += [f"{name} {value}" for name, value in sorted(dict(record.get("settings", {})).items())]
    return " · ".join(parts)


def embedding_names(adata: ad.AnnData) -> dict[str, str]:
    names: dict[str, str] = {}
    seen: dict[str, int] = {}
    for key in embedding_keys(adata):
        name = _name(adata.uns[RECORDS][key])
        seen[name] = seen.get(name, 0) + 1
        names[key] = name if seen[name] == 1 else f"{name} ({seen[name]})"
    return names


def _joined(values: Iterable[Any]) -> str:
    return "\n".join(str(value) for value in values)


def record_rows(adata: ad.AnnData, key: str) -> list[tuple[str, str]]:
    record = adata.uns[RECORDS][key]
    rows = [
        ("Method", str(record["method"])),
        ("Dimensions", str(int(record["dims"]))),
        ("Top N", str(int(record["top_n"]))),
    ]
    rows += [(str(name), str(value)) for name, value in sorted(dict(record.get("settings", {})).items())]
    used = zip(record["features"], record["scores"], strict=True)
    rows += [
        ("Picked features", _joined(record["picked"])),
        ("Samples", _joined(int(sample) for sample in record["samples"])),
        ("Values", VALUES_NAMES.get(str(record.get("values", ORIGINAL)), str(record.get("values")))),
        ("Ranking k", str(int(record["ranking_k"]))),
        ("Seed", str(int(record["seed"]))),
        ("Subsample seed", str(int(adata.uns[SUBSAMPLE]["seed"]))),
        ("Features used", _joined(f"{feature} {float(score):.3f}" for feature, score in used)),
    ]
    return rows
