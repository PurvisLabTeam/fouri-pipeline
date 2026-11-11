from __future__ import annotations

import contextlib
import copy
import io
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fouri.app.cells import CONDITION_JOIN
from fouri.app.names import natural_key
from fouri.app.state import open_experiment
from fouri.config import Experiment, save_config
from fouri.embedding import EMBED_MEM_PER_WORKER, Method, Run, make_runs, run_key
from fouri.metadata import plate_sort_key
from fouri.parallel import worker_count

RECORDS = "embeddings"
SUBSAMPLE = "subsample"
SHORTEST, LONGEST = 0.2, 3.0

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
SETTING_LABELS: dict[Method, dict[str, str]] = {
    Method.PHATE: {
        "knn": "neighbors",
        "t": "diffusion time t",
        "gamma": "gamma",
        "decay": "decay",
        "n_pca": "PCA components",
    },
    Method.UMAP: {"n_neighbors": "neighbors", "min_dist": "smallest distance", "metric": "distance metric"},
}
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


def _whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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
    "ranking_k": _WHOLE_1,
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


def _whole_list(value: object) -> bool:
    return isinstance(value, list) and all(_whole(item) for item in value)


def _text_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _value_list(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, (int, float, str)) and not isinstance(item, bool) for item in value
    )


def _methods_problem(methods: object) -> str | None:
    if not isinstance(methods, dict):
        return '"methods" should be a set of methods.'
    for method, labels in SETTING_LABELS.items():
        if method.value not in methods:
            return f'"{method}" is missing from "methods".'
        spec = methods[method.value]
        if not isinstance(spec, dict):
            return f'"{method}" should be a set of settings.'
        if "dims" not in spec:
            return f'"dims" is missing from {method}.'
        if not _whole_list(spec["dims"]):
            return f'{method}\'s "dims" should be a list of whole numbers.'
        if "settings" not in spec:
            return f'"settings" is missing from {method}.'
        settings = spec["settings"]
        if not isinstance(settings, dict):
            return f'{method}\'s "settings" should be a set of settings.'
        for name in labels:
            if name not in settings:
                return f'{method}\'s setting "{name}" is missing.'
            if not _value_list(settings[name]):
                return f'{method}\'s "{name}" should be a list of values.'
    return None


_KEYS: dict[str, tuple[Callable[[Any], bool], str]] = {
    "picked": (_text_list, "a list of feature names"),
    "samples": (_whole_list, "a list of whole numbers"),
    "ranking_k": (_whole, "a whole number"),
    "top_n": (_whole_list, "a list of whole numbers"),
    "methods": (lambda value: True, ""),
    "skipped": (_text_list, "a list of names"),
    "seed": (_whole, "a whole number"),
    "values": (lambda value: isinstance(value, str), "text"),
}


def _setup_problem(setup: object) -> str | None:
    if not isinstance(setup, dict):
        return "it is not a set of settings."
    for key, (fits, kind) in _KEYS.items():
        if key not in setup:
            return f'"{key}" is missing.'
        if key == "methods":
            problem = _methods_problem(setup[key])
            if problem is not None:
                return problem
        elif not fits(setup[key]):
            return f'"{key}" should be {kind}.'
    return None


def saved_setup(exp: Experiment) -> dict[str, Any] | None:
    app = exp.config.get("app")
    if app is None:
        return None
    if not isinstance(app, dict):
        raise ValueError('"app" in experiment.json is not a set of settings.')
    if app.get("embedding") is None:
        return None
    setup = app["embedding"]
    problem = _setup_problem(setup)
    if problem is not None:
        raise ValueError(problem)
    return copy.deepcopy(setup)


def setup_runs(setup: Mapping[str, Any]) -> list[Run]:
    runs: list[Run] = []
    for method in Method:
        spec = setup["methods"][method.value]
        runs += make_runs(method, spec["dims"], setup["top_n"], spec["settings"])
    return runs


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
    with open(exp.config_path, encoding="utf-8") as file:
        config = json.load(file)
    app = config.get("app")
    config["app"] = {**(app if isinstance(app, dict) else {}), "embedding": copy.deepcopy(dict(setup))}
    save_config(config, exp.config_path)
    return open_experiment(exp.config_path)


def embedding_settings(exp: Experiment) -> dict[str, Any] | None:
    setup = saved_setup(exp)
    if setup is None:
        return None
    skipped = set(setup["skipped"])
    runs = [run for run in setup_runs(setup) if run_key(run) not in skipped]
    return {
        "runs": [
            {"method": str(run.method), "dims": run.dims, "top_n": run.top_n, "settings": dict(run.settings)}
            for run in runs
        ],
        "picked": list(setup["picked"]),
        "samples": list(setup["samples"]),
        "ranking_k": setup["ranking_k"],
        "seed": setup["seed"],
    }


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
        ("Ranking k", str(int(record["ranking_k"]))),
        ("Seed", str(int(record["seed"]))),
        ("Subsample seed", str(int(adata.uns[SUBSAMPLE]["seed"]))),
        ("Features used", _joined(f"{feature} {float(score):.3f}" for feature, score in used)),
    ]
    return rows


def _kept(value: float) -> float:
    return min(max(float(value), SHORTEST), LONGEST)


def box_shape(fig: go.Figure, x: float, y: float, z: float) -> go.Figure:
    fig.update_layout(scene_aspectmode="manual", scene_aspectratio={"x": _kept(x), "y": _kept(y), "z": _kept(z)})
    return fig


def data_proportions(adata: ad.AnnData, key: str) -> tuple[float, float, float]:
    coords = np.asarray(adata.obsm[key], dtype=np.float64)
    spread = np.nanmax(coords, axis=0) - np.nanmin(coords, axis=0)
    longest = float(spread.max())
    if not longest > 0:
        return (1.0, 1.0, 1.0)
    x, y, z = (_kept(length / longest) for length in spread[:3])
    return (x, y, z)
