"""Plotly figures for setting feature thresholds and viewing embeddings.

`plot_threshold_histogram` plots the histogram of a feature with the suggested and the chosen
cut-offs. `plot_embedding` plots the cells of an embedding, colored by a measurement or a column of
`.obs`. The font and color constants are shared with the app's other figures.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go

from fouri.features import DEFAULT_SDS, Threshold, curve_range, middle_values, suggest_cutoffs

FONT = "Calibri, Carlito, sans-serif"  # Calibri on the lab PCs, its stand-in Carlito elsewhere
BAR_COLOR = "#0F9E8E"  # teal
# color and dash pattern of each suggested cut-off
SUGGESTION_STYLES: dict[Threshold, tuple[str, str]] = {
    Threshold.OTSU: ("#7C3AED", "dash"),  # purple
    Threshold.OTSU_2CUT: ("#1D4ED8", "dashdot"),  # blue
    Threshold.VALLEY: ("#E11D48", "dot"),  # rose
}
CHOSEN_COLOR = "#DC2626"  # the cut-offs actually used: red, solid
BAND_COLOR = "rgba(29, 27, 44, 0.07)"  # between Otsu and the valley, a faint tint over the bars
# one color per group (treatment, sample, label ...) in an embedding, in turn: the app's colors
GROUP_COLORS = ["#7C3AED", "#0F9E8E", "#DB2777", "#2563EB", "#D97706", "#059669", "#9333EA", "#0891B2"]
VALUE_SCALE = "Viridis"  # cells colored by a number
NO_VALUE_COLOR = "#C9C6D6"


def plot_threshold_histogram(
    values: npt.ArrayLike,
    cutoffs: Sequence[float],
    *,
    bins: int = 256,
    title: str | None = None,
    x_title: str = "pRB / RB",
    sds: float = DEFAULT_SDS,
) -> go.Figure:
    """Plot the histogram of a feature with the suggested and the chosen cut-offs.

    NaN values are excluded. The bars show the values within `sds` standard deviations of the mean,
    the values from which the suggested cut-offs are computed (`fouri.features.suggest_cutoffs`).
    The initial `x` range runs from the lowest to the highest value at which the kernel density
    estimate of the drawn values is at least 1% of the estimate's maximum
    (`fouri.features.curve_range`), widened to include every cut-off. `bins` bars of equal width
    span the initial range. Double-clicking the plot shows every bar.

    The subtitle states the distance between the Otsu and valley cut-offs, the number of drawn cells
    outside the initial range, and the number of cells beyond `sds` standard deviations. A part that
    does not apply is omitted.

    Parameters
    ----------
    values : npt.ArrayLike
        The feature's value for each cell (e.g., pRB/RB ratios).
    cutoffs : Sequence[float]
        The chosen cut-offs, in the feature's units. Empty for no chosen cut-off.
    bins : int, default=256
        The number of bars across the initial `x` range.
    title : str | None, default=None
        The figure title. None or an empty string falls back to the number of cells with a value
        (e.g., `12,345 cells`).
    x_title : str, default="pRB / RB"
        The `x` axis title.
    sds : float, default=DEFAULT_SDS
        The number of standard deviations from the mean within which a value is drawn and used for
        the suggested cut-offs.

    Returns
    -------
    go.Figure
        The histogram. Each suggestion method is a line trace with a single legend entry. When both
        the Otsu and valley cut-offs exist, the band between the two cut-offs is a filled trace
        without a legend entry. Each chosen cut-off is an editable line shape with a legend entry.
        The chosen cut-offs are the figure's only shapes.

    Raises
    ------
    ValueError
        If `values` does not contain any non-NaN value.

    Notes
    -----
    Infinite values are not excluded. An infinite value raises `IndexError`.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("There are no cells with a value to draw.")

    shown = middle_values(values, sds)
    suggestions = suggest_cutoffs(values, sds)
    view_start, view_end = curve_range(values, sds)
    lines = [cut for cuts in suggestions.values() for cut in cuts] + [float(cut) for cut in cutoffs]
    low, high = min(view_start, *lines), max(view_end, *lines)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=shown,
            xbins={"start": float(shown.min()), "end": float(shown.max()), "size": (high - low) / bins},
            marker_color=BAR_COLOR,  # simple_white outlines each bar in white
            showlegend=False,
            name="Cells",
        )
    )

    if Threshold.OTSU in suggestions and Threshold.VALLEY in suggestions:
        otsu, valley = suggestions[Threshold.OTSU][0], suggestions[Threshold.VALLEY][0]
        low_cut, high_cut = min(otsu, valley), max(otsu, valley)
        # a filled box on the hidden 0-to-1 axis, not a shape, so it never drags with the cut-offs
        fig.add_trace(
            go.Scatter(
                x=[low_cut, high_cut, high_cut, low_cut, low_cut],
                y=[0, 0, 1, 1, 0],
                yaxis="y2",
                mode="lines",
                fill="toself",
                fillcolor=BAND_COLOR,
                line_width=0,
                hoverinfo="skip",
                showlegend=False,
                name="Otsu-valley gap",
            )
        )

    for thresh, cuts in suggestions.items():
        color, dash = SUGGESTION_STYLES[thresh]
        fig.add_trace(_suggestion_lines(cuts, f"{thresh} ({_as_text(cuts)})", color=color, dash=dash))

    # one shape per chosen cut-off, so each can be dragged on its own where the figure is a web page
    for cut in cutoffs:
        fig.add_shape(
            type="line",
            x0=cut,
            x1=cut,
            y0=0,
            y1=1,
            yref="paper",
            line={"color": CHOSEN_COLOR, "dash": "solid", "width": 3},
            opacity=1,  # simple_white draws every shape at 0.3 unless told otherwise
            layer="above",
            editable=True,
            name=f"Your cut-off ({cut:.3f})",
            showlegend=True,
        )

    fig.update_layout(
        template="simple_white",
        font_family=FONT,
        font_size=14,
        title_text=title or f"{values.size:,} cells",
        title_subtitle_text=_subtitle(values, shown, suggestions, low, high, sds),
        # hidden axis from 0 at the bottom of the plot to 1 at the top, for the suggestion lines
        yaxis2={"overlaying": "y", "range": [0, 1], "visible": False, "fixedrange": True},
    )
    fig.update_xaxes(range=[low, high], title_text=x_title)
    fig.update_yaxes(title_text="Cells")
    return fig


def plot_embedding(
    adata: ad.AnnData, key: str, color_by: str, *, title: str | None = None, layer: str | None = None
) -> go.Figure:
    """Plot the cells of an embedding, colored by a measurement or a column of `adata.obs`.

    A 3D embedding is plotted as a 3D scatter plot, with axes of equal length on screen. A 2D
    embedding is plotted as a WebGL scatter plot. Cells with a NaN coordinate are not drawn.
    Hovering over a cell shows the cell's value of `color_by`. The figure's `uirevision` is `key`:
    the zoom and camera position persist across redraws of the same embedding.

    A measurement or a numeric column colors the cells on the `VALUE_SCALE` color scale, from the
    1st to the 99th percentile of the values. The color bar title is `z-score` for a measurement
    read from `adata.X`, `corrected z-score` for a measurement read from `layer`, and the column's
    name for a column. With `layer`, the cells without a value in the layer are drawn first, as a
    trace named `No corrected value` in `NO_VALUE_COLOR`. The percentiles exclude those cells. A
    `layer` not in `adata.layers` raises `KeyError`.

    A text or categorical column gives each group a trace with a legend entry. Group colors are
    assigned from `GROUP_COLORS` in order, cycling when the groups outnumber the colors. The groups
    of a categorical column are in category order. Other groups are sorted as text.

    The method is read from the embedding's record in `.uns["embeddings"]`. When the record does not
    contain a method, the first part of `key` after `X_`, uppercased, is used (e.g., `PHATE`). The
    method names the axes of a 2D embedding.

    Parameters
    ----------
    adata : ad.AnnData
        The cells, with the embedding in `.obsm` and the embedding's record in `.uns["embeddings"]`
        (e.g., the `standard_adata_sub.h5ad` file).
    key : str
        The embedding's key in `.obsm` (e.g., `X_phate_3d_top10_knn80`).
    color_by : str
        A measurement (a name in `adata.var_names`) or a column of `adata.obs`. A name in both is
        read as a measurement.
    title : str | None, default=None
        The figure title. None falls back to the method, the dimensions, `top_n`, and the settings
        from the embedding's record. An empty string draws no title.
    layer : str | None, default=None
        The layer to read the measurement from (e.g., `combat`). None reads `adata.X`.

    Returns
    -------
    go.Figure
        The scatter plot of the embedding's cells.

    Raises
    ------
    ValueError
        If `key` is not in `adata.obsm`. The message lists the embeddings in `adata.obsm`. If
        `color_by` is neither a measurement nor a column of `adata.obs`. If `layer` is not None and
        `color_by` is not a measurement.
    """
    if key not in adata.obsm:
        names = "".join(f"{name}\n" for name in adata.obsm)
        listed = f"\n{names}" if names else " none."
        raise ValueError(f"There is no embedding called {key} in this file. Its embeddings are:{listed}")
    is_measurement = color_by in adata.var_names
    if not is_measurement and color_by not in adata.obs.columns:
        raise ValueError(
            f"Cells can't be colored by {color_by}: it is neither a measurement nor a column describing the cells."
        )
    if layer is not None and not is_measurement:
        raise ValueError(f"Only a measurement has corrected values; {color_by} is a column describing the cells.")

    coords = np.asarray(adata.obsm[key], dtype=np.float64)
    drawn = ~np.isnan(coords).any(axis=1)
    coords = coords[drawn]
    dims = coords.shape[1]
    record: Mapping[str, Any] = adata.uns.get("embeddings", {}).get(key, {})
    method = str(record.get("method", key.removeprefix("X_").split("_")[0].upper()))
    marker_size = 2 if dims == 3 else 3

    def points(rows: npt.NDArray[np.bool_] | slice, **style: Any) -> go.Scatter3d | go.Scattergl:
        xyz = dict(zip("xyz", coords[rows].T, strict=False))
        if dims == 3:
            return go.Scatter3d(mode="markers", **xyz, **style)
        return go.Scattergl(mode="markers", **xyz, **style)

    fig = go.Figure()
    column = None if is_measurement else adata.obs[color_by][drawn]
    if column is None or (pd.api.types.is_numeric_dtype(column) and not isinstance(column.dtype, pd.CategoricalDtype)):
        rows: npt.NDArray[np.bool_] | slice = slice(None)
        if column is None and layer is not None:
            values = np.asarray(adata[drawn, color_by].layers[layer], dtype=np.float64).ravel()
            scale_title = "corrected z-score"
            has_value = ~np.isnan(values)
            if not has_value.all():
                fig.add_trace(
                    points(
                        ~has_value,
                        name="No corrected value",
                        marker={"size": marker_size, "color": NO_VALUE_COLOR, "opacity": 0.8},
                        hovertemplate=f"{color_by}: no corrected value<extra></extra>",
                    )
                )
                fig.update_layout(
                    legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1, "xanchor": "right"}
                )
                rows, values = has_value, values[has_value]
        elif column is None:
            values = np.asarray(adata[drawn, color_by].X, dtype=np.float64).ravel()
            scale_title = "z-score"  # measurements are standardised in the file
        else:
            values = column.to_numpy(dtype=np.float64)
            scale_title = color_by
        if layer is None or values.size:
            low, high = np.nanpercentile(values, [1, 99])
            fig.add_trace(
                points(
                    rows,
                    name=color_by,
                    marker={
                        "size": marker_size,
                        "color": values,
                        "colorscale": VALUE_SCALE,
                        "cmin": float(low),
                        "cmax": float(high),
                        "colorbar": {"title": {"text": scale_title}},
                        "opacity": 0.8,
                    },
                    hovertemplate=f"{color_by}: %{{marker.color:.2f}}<extra></extra>",
                )
            )
    else:
        labels = column.astype(str).to_numpy()
        if isinstance(column.dtype, pd.CategoricalDtype):  # keep the column's own order, e.g. sample 2 before 10
            groups = [str(group) for group in column.cat.categories if str(group) in set(labels)]
        else:
            groups = sorted(set(labels))
        for i, group in enumerate(groups):
            fig.add_trace(
                points(
                    labels == group,
                    name=group,
                    marker={"size": marker_size, "color": GROUP_COLORS[i % len(GROUP_COLORS)], "opacity": 0.7},
                    hovertemplate=f"{color_by}: {group}<extra></extra>",
                )
            )

    fig.update_layout(
        template="simple_white",
        font_family=FONT,
        font_size=14,
        title_text=_embedding_title(method, dims, record) if title is None else title,
        legend_itemsizing="constant",  # legend dots readable although the points are small
        uirevision=key,  # recolouring keeps the camera where the viewer turned it
    )
    if dims == 3:
        fig.update_layout(
            scene={
                "xaxis_title_text": "1",
                "yaxis_title_text": "2",
                "zaxis_title_text": "3",
                "aspectmode": "cube",
            }
        )
    else:
        fig.update_xaxes(title_text=f"{method} 1")
        fig.update_yaxes(title_text=f"{method} 2")
    return fig


def _embedding_title(method: str, dims: int, record: Mapping[str, Any]) -> str:
    parts = [f"{method} {dims}D"]
    if "top_n" in record:
        parts.append(f"top {int(record['top_n'])} features")
    parts += [f"{name} {value}" for name, value in sorted(record.get("settings", {}).items())]
    return " · ".join(parts)


def _suggestion_lines(xs: Sequence[float], name: str, *, color: str, dash: str) -> go.Scatter:
    """One trace drawing a full-height line at each x, on the hidden 0-to-1 axis.

    A trace rather than shapes: its legend entry shows the line's dash pattern, and hides
    or shows all of its lines at once.
    """
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    for x in xs:
        line_x += [x, x, None]  # None breaks the line between cut-offs
        line_y += [0, 1, None]
    return go.Scatter(
        x=line_x,
        y=line_y,
        yaxis="y2",
        mode="lines",
        line={"color": color, "dash": dash, "width": 2},
        name=name,
        hoverinfo="skip",
    )


def _as_text(cuts: Sequence[float]) -> str:
    return ", ".join(f"{cut:.3f}" for cut in cuts)


def _subtitle(
    values: npt.NDArray[np.float64],
    shown: npt.NDArray[np.float64],
    suggestions: dict[Threshold, list[float]],
    low: float,
    high: float,
    sds: float,
) -> str:
    parts = []
    if Threshold.OTSU in suggestions and Threshold.VALLEY in suggestions:
        gap = abs(suggestions[Threshold.OTSU][0] - suggestions[Threshold.VALLEY][0])
        parts.append(f"Otsu and valley {gap:.3f} apart")
    off_screen = int(np.sum((shown < low) | (shown > high)))
    if off_screen:
        parts.append(f"{off_screen:,} cells off screen (double-click to show)")
    far_out = values.size - shown.size
    if far_out:
        parts.append(f"{far_out:,} cells more than {sds:g} SD from the mean not drawn")
    return " · ".join(parts)
