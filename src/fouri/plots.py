"""Figures for the pipeline's interactive steps."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from fouri.features import Threshold, curve_range, middle_values, suggest_cutoffs

FONT = "Calibri, Carlito, sans-serif"  # Calibri on the lab PCs, its stand-in Carlito elsewhere
BAR_COLOR = "#0F9E8E"  # teal
# colour and dash pattern of each suggested cut-off
SUGGESTION_STYLES: dict[Threshold, tuple[str, str]] = {
    Threshold.OTSU: ("#7C3AED", "dash"),  # purple
    Threshold.OTSU_2CUT: ("#1D4ED8", "dashdot"),  # blue
    Threshold.VALLEY: ("#E11D48", "dot"),  # rose
}
CHOSEN_COLOR = "#DC2626"  # the cut-offs actually used: red, solid
BAND_COLOR = "rgba(29, 27, 44, 0.07)"  # between Otsu and the valley, a faint tint over the bars


def plot_threshold_histogram(
    values: npt.ArrayLike,
    cutoffs: Sequence[float],
    *,
    bins: int = 256,
    title: str | None = None,
    x_title: str = "pRB / RB",
) -> go.Figure:
    """Histogram of one feature with the suggested cut-offs and the chosen ones.

    Bars show the cells within 3 standard deviations of the mean, the same cells the
    suggestions are worked out from. The figure opens on the stretch where the smoothed
    curve stays above 1 % of its tallest peak, widened to keep every line on screen;
    `bins` bars fill that stretch. Double-clicking the plot zooms out to every bar.

    Parameters
    ----------
    values : array_like
        The feature's value in every cell of the sample, e.g. its pRB/RB ratios.
    cutoffs : sequence of float
        The cut-offs in use, in the feature's units: one or two values, or none.
    bins : int
        Number of bars across the opening view.
    title : str, optional
        Figure title; by default the number of cells.
    x_title : str
        Label of the x axis.

    Returns
    -------
    plotly.graph_objects.Figure
        Each chosen cut-off is its own editable shape, so it can be dragged where the
        figure is shown in a web page. Each suggestion method is one line trace, so its
        legend entry shows or hides all of its lines. The band between Otsu and the valley
        is a trace too, so the cut-offs are the only shapes: a page may make every shape
        draggable without the band moving.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("There are no cells with a value to draw.")

    shown = middle_values(values)
    suggestions = suggest_cutoffs(values)
    view_start, view_end = curve_range(values)
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
        title_subtitle_text=_subtitle(values, shown, suggestions, low, high),
        # hidden axis from 0 at the bottom of the plot to 1 at the top, for the suggestion lines
        yaxis2={"overlaying": "y", "range": [0, 1], "visible": False, "fixedrange": True},
    )
    fig.update_xaxes(range=[low, high], title_text=x_title)
    fig.update_yaxes(title_text="Cells")
    return fig


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
        parts.append(f"{far_out:,} cells more than 3 SD from the mean not drawn")
    return " · ".join(parts)
