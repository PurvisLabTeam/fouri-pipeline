from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from fouri.config import Experiment
from fouri.embedding import Method, Run, make_runs, run_key

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


def _whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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
        "values": setup["values"],
    }
