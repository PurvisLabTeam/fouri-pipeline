"""The app's look: a violet palette in light and dark, one font, flat cards, and figures to match.

The palette lives in CSS variables, so the class Quasar puts on the page body in dark mode
switches every color at once. Plotly figures cannot read CSS, so ``restyle`` sets their colors
for the mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from nicegui import ui

LIGHT = {
    "bg": "#F6F5FB",
    "surface": "#FFFFFF",
    "line": "#E6E3F0",
    "text": "#1E1B2E",
    "muted": "#6C6885",
    "primary": "#6D28D9",
    "primary-soft": "#EFE9FD",
    "grid": "#ECEAF3",
}
DARK = {
    "bg": "#0F0D17",
    "surface": "#18151F",
    "line": "#2B2738",
    "text": "#ECEAF4",
    "muted": "#A29DB8",
    "primary": "#A78BFA",
    "primary-soft": "#2A2142",
    "grid": "#26222F",
}


def _variables(palette: dict[str, str]) -> str:
    return "".join(f"--c-{name}: {value};" for name, value in palette.items())


# Quasar's own primary color is set on the body by ui.colors; the !important lets dark mode
# override it, so the switch changes buttons and links without rebuilding the page.
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{ {_variables(LIGHT)} }}
body.body--dark {{ {_variables(DARK)} --q-primary: {DARK["primary"]} !important; }}
body {{ font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--c-bg);
        color: var(--c-text); }}
.q-header {{ background: var(--c-surface) !important; color: var(--c-text) !important;
            border-bottom: 1px solid var(--c-line); }}
.q-drawer {{ background: var(--c-surface) !important; border-right: 1px solid var(--c-line); }}
.q-card {{ background: var(--c-surface); border: 1px solid var(--c-line); border-radius: 14px; box-shadow: none; }}
.muted {{ color: var(--c-muted); }}
.nav-item {{ color: var(--c-muted); border-radius: 10px; }}
.nav-item:hover {{ background: var(--c-primary-soft); color: var(--c-text); }}
.nav-item.active {{ background: var(--c-primary-soft); color: var(--c-primary); font-weight: 600; }}
.brand {{ background: var(--c-primary); color: white; border-radius: 9px; width: 30px; height: 30px;
          display: grid; place-items: center; font-weight: 700; font-size: 14px; }}
.page-title {{ font-size: 1.4rem; font-weight: 600; }}
.q-btn {{ border-radius: 10px; }}
.js-plotly-plot .main-svg {{ border-radius: 12px; }}
"""


def apply_theme() -> None:
    """Add the palette, font and card styles to the page being built."""
    ui.add_css(CSS)
    ui.colors(
        primary=LIGHT["primary"],
        secondary="#0F9E8E",
        accent="#DB2777",
        positive="#0F9E8E",
        warning="#D97706",
        dark=DARK["surface"],
        dark_page=DARK["bg"],
    )
    ui.button.default_props("no-caps unelevated")
    ui.toggle.default_props("no-caps unelevated")
    ui.card.default_props("flat")
    ui.input.default_props("outlined dense")
    ui.number.default_props("outlined dense")
    ui.select.default_props("outlined dense options-dense")
    ui.linear_progress.default_props("rounded")


# fouri.plots' colors, chosen for a white background, and their lighter versions for a dark one
DARK_SWAP = {
    "#7c3aed": "#a78bfa",
    "#1d4ed8": "#60a5fa",
    "#e11d48": "#fb7185",
    "#dc2626": "#f87171",
    "#0f9e8e": "#2dd4bf",
    "rgba(29, 27, 44, 0.07)": "rgba(255, 255, 255, 0.07)",
}
WHITE = "#FFFFFF"
POINTS = {"scatter", "scattergl", "scatter3d"}


def _swap(color: Any) -> Any:
    return DARK_SWAP.get(color.lower(), color) if isinstance(color, str) else color


def _numbers(color: Any) -> bool:
    """Whether a marker color is a number per point rather than one color."""
    if color is None or isinstance(color, str):
        return False
    try:
        first = color[0]
    except (TypeError, IndexError, KeyError):
        return False
    return isinstance(first, int | float | np.number) and not isinstance(first, bool | np.bool_)


def colored_by_value(fig: go.Figure) -> bool:
    """Return whether the figure colors its points by a number on a color scale."""
    return any(
        trace.type in POINTS and (trace.marker.coloraxis is not None or _numbers(trace.marker.color))
        for trace in fig.data
    )


def on_white_panel(fig: go.Figure, *, dark: bool) -> bool:
    """Return whether the figure belongs on a white panel: dark mode, points colored by value."""
    return dark and colored_by_value(fig)


def restyle(fig: go.Figure, *, dark: bool) -> go.Figure:
    """Return the figure styled for the mode, with its color scales and ranges untouched.

    The figure is changed in place: a transparent background with the mode's text, grid and
    axis colors. Points colored by a value keep the light style on a white background even in
    dark mode, since a scale's dark end would sink into a dark one. Otherwise, in dark mode,
    fouri.plots' colors turn lighter and bars get dark outlines.
    """
    white = on_white_panel(fig, dark=dark)
    dark = dark and not white
    palette = DARK if dark else LIGHT
    background = WHITE if white else "rgba(0,0,0,0)"
    fig.update_layout(
        paper_bgcolor=background,
        plot_bgcolor=background,
        font_color=palette["text"],
        legend_font_color=palette["text"],
        title_font_color=palette["text"],
    )
    axes = {
        "gridcolor": palette["grid"],
        "linecolor": palette["muted"],
        "zerolinecolor": palette["line"],
        "tickcolor": palette["muted"],
    }
    fig.update_xaxes(**axes)
    fig.update_yaxes(**axes)
    if any(trace.type == "scatter3d" for trace in fig.data):
        # 3D axes lose the white walls simple_white gives them
        axis_3d = {"showbackground": False, "showgrid": True, "color": palette["text"], **axes}
        fig.update_scenes(xaxis=axis_3d, yaxis=axis_3d, zaxis=axis_3d)
    if not dark:
        return fig
    for trace in fig.data:
        line = getattr(trace, "line", None)
        if line is not None and line.color is not None:
            line.color = _swap(line.color)
        if getattr(trace, "fillcolor", None) is not None:
            trace.fillcolor = _swap(trace.fillcolor)
        marker = getattr(trace, "marker", None)
        if marker is not None:
            if isinstance(marker.color, str):
                marker.color = _swap(marker.color)
            if trace.type in {"histogram", "bar"}:
                marker.line.color = palette["surface"]  # simple_white outlines bars in white
                marker.line.width = 0.6
    for shape in fig.layout.shapes:
        if shape.line is not None and shape.line.color is not None:
            shape.line.color = _swap(shape.line.color)
        if shape.fillcolor is not None:
            shape.fillcolor = _swap(shape.fillcolor)
    return fig
