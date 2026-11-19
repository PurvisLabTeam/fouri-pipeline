from __future__ import annotations

import anndata as ad
import numpy as np
import plotly.graph_objects as go

SHORTEST, LONGEST = 0.2, 3.0


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
