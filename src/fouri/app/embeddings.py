from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import anndata as ad
import numpy as np
import plotly.graph_objects as go

from fouri.app.names import natural_key

RECORDS = "embeddings"
SUBSAMPLE = "subsample"
SHORTEST, LONGEST = 0.2, 3.0


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
    return ", ".join(str(value) for value in values)


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
        ("Features used", ", ".join(f"{feature} {float(score):.3f}" for feature, score in used)),
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
