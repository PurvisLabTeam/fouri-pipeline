from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from nicegui import ui
from nicegui.events import ValueChangeEventArguments

from fouri.app.cells import Cells, load_cells
from fouri.app.names import VALUE, marker_menu
from fouri.app.rules import save_filters
from fouri.app.shell import empty_state, page_frame
from fouri.app.state import STATE
from fouri.app.theme import restyle
from fouri.config import Experiment
from fouri.filters import FilterRule, apply_filters, keep, saved_filters
from fouri.io import read_info_file
from fouri.plots import FONT

LINE = "Excluded cells stay in the table, marked included = false; later steps use only included cells."
NO_RULE = "Add a filter to see which cells it keeps."
FIX_RED = "Fix the filters marked in red, then save."
KINDS = {
    FilterRule.MIDDLE_PERCENTILES: "Keep the middle percentiles",
    FilterRule.ROBUST_SD: "Keep within N robust SDs of the median",
    FilterRule.BETWEEN: "Keep between two values",
}
LABELS = {
    FilterRule.MIDDLE_PERCENTILES: {"low": "Lowest %", "high": "Highest %"},
    FilterRule.ROBUST_SD: {"sds": "SDs"},
    FilterRule.BETWEEN: {"low": "From", "high": "To"},
}
STEPS: dict[FilterRule, float | None] = {
    FilterRule.MIDDLE_PERCENTILES: 0.5,
    FilterRule.ROBUST_SD: 0.5,
    FilterRule.BETWEEN: None,
}
NUMBERS = tuple(dict.fromkeys(name for kind in FilterRule for name in kind.fields))
STARTS: dict[FilterRule, dict[str, float | None]] = {
    FilterRule.MIDDLE_PERCENTILES: {"low": 1.0, "high": 99.0},
    FilterRule.ROBUST_SD: {"sds": 3.0},
}
BETWEEN_PERCENTILES = (1, 99)
SIGNIFICANT_DIGITS = 3
NEW_FEATURE = "nuc_area"
NEW_NAME = "Filter {number}"
HISTOGRAM_PERCENTILES = (0.1, 99.9)
BINS = 160
KEPT_COLOR, LEFT_OUT_COLOR = "#0F9E8E", "#DC2626"
TEMPLATE = "simple_white"
MARGIN = {"l": 60, "r": 20, "t": 60, "b": 50}
LEGEND = {"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1, "xanchor": "right"}
SHOWN = "border-color: var(--c-primary)"
COUNT, PROBLEM = "text-xs muted", "text-xs text-negative"
MODE_SECONDS = 0.25

Menu = dict[str, dict[str, str]]


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip() or None


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _kind(value: str) -> FilterRule | None:
    try:
        return FilterRule(value)
    except ValueError:
        return None


def _fields(value: str) -> list[str]:
    kind = _kind(value)
    return kind.fields if kind is not None else []


def _rounded(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def _as_saved(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _reason(error: Exception) -> str:
    text = error.strerror if isinstance(error, OSError) and error.strerror else str(error)
    return text.rstrip(".")


def _cells(count: int) -> str:
    return f"{count:,} cell" if count == 1 else f"{count:,} cells"


def _covered_line(count: int) -> str:
    verb = "is" if count == 1 else "are"
    return f"{_cells(count)} under artifact masks {verb} left out of these counts."


@dataclass(eq=False)
class _Rule:
    name: str | None
    use: bool
    kind: str
    feature: str | None
    per_sample: bool
    numbers: dict[str, float | None] = field(default_factory=dict)
    saved_at: int | None = None
    kept: pd.Series | None = None
    problem: str = ""

    def saved(self) -> dict[str, Any]:
        rule: dict[str, Any] = {}
        if self.name:
            rule["name"] = self.name
        rule["use"] = self.use
        rule["rule"] = self.kind
        if self.feature:
            rule["feature"] = self.feature
        for name in _fields(self.kind):
            value = self.numbers.get(name)
            if value is not None:
                rule[name] = _as_saved(value)
        rule["per_sample"] = self.per_sample
        return rule


def _loaded(index: int, rule: Mapping[str, Any]) -> _Rule:
    return _Rule(
        name=_text(rule.get("name")),
        use=rule.get("use") is True,
        kind=str(rule.get("rule", "")),
        feature=_text(rule.get("feature")),
        per_sample=bool(rule.get("per_sample", False)),
        numbers={name: _number(rule.get(name)) for name in NUMBERS},
        saved_at=index,
    )


def _saved_list(exp: Experiment) -> list[dict[str, Any]]:
    app = exp.config.get("app")
    rules = app.get("filters") if isinstance(app, dict) else None
    return [dict(rule) for rule in rules] if isinstance(rules, list) else []


def _numbers(table: pd.DataFrame, column: str | None) -> npt.NDArray[np.float64]:
    if column is None or column not in table.columns:
        return np.array([], dtype=np.float64)
    try:
        return table[column].to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError):
        return np.array([], dtype=np.float64)


def _place(menu: Menu, column: str | None) -> tuple[str, str] | None:
    for marker, measures in menu.items():
        for measure, name in measures.items():
            if name == column:
                return marker, measure
    return None


class _FiltersPage:
    def __init__(self, exp: Experiment, cells: Cells, menu: Menu) -> None:
        self.exp = exp
        self.cells = cells
        self.menu = menu
        self.covered = int(cells.covered.sum())
        self.table = cells.table[~cells.covered] if self.covered else cells.table
        try:
            saved = [dict(rule) for rule in saved_filters(exp)]
            self.unreadable = ""
        except ValueError as error:
            saved = _saved_list(exp)
            reason = str(error).strip()
            self.unreadable = reason if reason.endswith(".") else reason + "."
        self.rules = [_loaded(index, rule) for index, rule in enumerate(saved)]
        self.shown: _Rule | None = self.rules[0] if self.rules else None
        self.cards: list[tuple[_Rule, ui.card, ui.label]] = []
        self.figure: go.Figure | None = None
        self.dark = STATE.dark

    def build(self) -> None:
        with ui.row().classes("w-full no-wrap items-start gap-6"):
            with ui.column().classes("w-[420px] shrink-0 gap-2"):
                ui.label("Filters").classes("page-title")
                ui.label(LINE).classes("text-sm muted")
                empty_state(self.cells)
                with ui.row().classes("gap-2"):
                    ui.button("Add a filter", icon="add", on_click=self.add).props("outline")
                    ui.button("Save filters", icon="save", on_click=self.save)
                self.total = ui.label().classes("text-sm font-medium mt-2")
                covered_line = ui.label(_covered_line(self.covered)).classes("text-sm muted")
                covered_line.set_visibility(self.covered > 0)
                self.unreadable_line = ui.label(
                    f"The saved filters can't be read: {self.unreadable} Rename one and save."
                ).classes("text-sm text-negative")
                self.unreadable_line.set_visibility(bool(self.unreadable))
                self.rule_list = ui.column().classes("w-full gap-2")
            with ui.column().classes("grow min-w-0 gap-1"):
                self.plot = ui.plotly({"data": [], "layout": {}}).classes("w-full h-[78vh]")
                self.message = ui.label().classes("text-sm muted")
        self.refresh()
        ui.timer(MODE_SECONDS, self.follow_mode)

    def refresh(self) -> None:
        self.cards.clear()
        self.rule_list.clear()
        with self.rule_list:
            for rule in self.rules:
                self.card(rule)
        self.update()

    def card(self, rule: _Rule) -> None:
        kinds: dict[str, str] = {str(kind): label for kind, label in KINDS.items()}
        if rule.kind not in kinds:
            kinds[rule.kind] = rule.kind
        kind = _kind(rule.kind)
        with ui.card().classes("w-full p-3 gap-1") as card:
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                ui.input("Name", value=rule.name or "", on_change=lambda e: self.set_name(rule, e.value)).props(
                    "dense"
                ).classes("grow")
                ui.switch("Use", value=rule.use, on_change=lambda e: self.switch(rule, bool(e.value)))
                ui.button(icon="bar_chart", on_click=lambda: self.show_rule(rule)).props("flat round dense").tooltip(
                    "Show this rule's histogram"
                )
                ui.button(icon="delete", on_click=lambda: self.delete(rule)).props("flat round dense")
            ui.select(kinds, value=rule.kind, on_change=lambda e: self.set_kind(rule, e.value)).classes("w-full")
            self.feature_picker(rule)
            with ui.row().classes("items-center no-wrap gap-2"):
                if kind is not None:
                    for name in kind.fields:
                        ui.number(
                            LABELS[kind][name],
                            value=rule.numbers.get(name),
                            step=STEPS[kind],
                            on_change=lambda e, name=name: self.set_number(rule, name, e.value),
                        ).classes("w-28")
                ui.switch(
                    "Per sample", value=rule.per_sample, on_change=lambda e: self.set_per_sample(rule, bool(e.value))
                )
            count = ui.label().classes(COUNT)
        self.cards.append((rule, card, count))

    def feature_picker(self, rule: _Rule) -> None:
        menu = dict(self.menu)
        place = _place(menu, rule.feature)
        if place is None and rule.feature is not None:
            menu[rule.feature] = {VALUE: rule.feature}
            place = rule.feature, VALUE
        marker, measure = place if place is not None else (None, None)
        with ui.row().classes("w-full no-wrap gap-1"):
            marker_box = ui.select(list(menu), value=marker, label="Feature", with_input=True).classes("w-7/12")
            measure_box = ui.select(list(menu[marker]) if marker else [], value=measure, label="measure").classes(
                "w-5/12"
            )

        def new_marker(e: ValueChangeEventArguments[Any]) -> None:
            if not e.value:
                return
            options = list(menu[e.value])
            kept = measure_box.value if measure_box.value in options else options[0]
            measure_box.set_options(options, value=kept)
            self.set_feature(rule, menu[e.value][kept])

        def new_measure(e: ValueChangeEventArguments[Any]) -> None:
            if e.value and marker_box.value and e.value in menu[marker_box.value]:
                self.set_feature(rule, menu[marker_box.value][e.value])

        marker_box.on_value_change(new_marker)
        measure_box.on_value_change(new_measure)

    def starting_numbers(self, kind: str, feature: str | None) -> dict[str, float | None]:
        known = _kind(kind)
        if known is None:
            return {}
        if known is not FilterRule.BETWEEN:
            return dict(STARTS[known])
        numbers = _numbers(self.table, feature)
        numbers = numbers[np.isfinite(numbers)]
        if numbers.size == 0:
            return {"low": None, "high": None}
        low, high = np.percentile(numbers, BETWEEN_PERCENTILES)
        return {"low": _rounded(float(low)), "high": _rounded(float(high))}

    def set_name(self, rule: _Rule, value: Any) -> None:
        name = _text(value)
        if name == rule.name:
            return
        rule.name = name
        self.shown = rule
        self.update()

    def set_kind(self, rule: _Rule, value: Any) -> None:
        if not value or value == rule.kind:
            return
        rule.kind = str(value)
        rule.numbers = self.starting_numbers(rule.kind, rule.feature)
        self.shown = rule
        self.refresh()

    def set_feature(self, rule: _Rule, column: str) -> None:
        if column == rule.feature:
            return
        rule.feature = column
        self.shown = rule
        if rule.kind == FilterRule.BETWEEN:
            rule.numbers = self.starting_numbers(FilterRule.BETWEEN, column)
            self.refresh()
            return
        self.update()

    def set_number(self, rule: _Rule, name: str, value: Any) -> None:
        number = _number(value)
        if number == rule.numbers.get(name):
            return
        rule.numbers[name] = number
        self.shown = rule
        self.update()

    def set_per_sample(self, rule: _Rule, value: bool) -> None:
        if value == rule.per_sample:
            return
        rule.per_sample = value
        self.shown = rule
        self.update()

    def switch(self, rule: _Rule, value: bool) -> None:
        if value == rule.use:
            return
        rule.use = value
        self.shown = rule
        if rule.saved_at is not None:
            saved = _saved_list(self.exp)
            if rule.saved_at < len(saved):
                saved[rule.saved_at]["use"] = value
                try:
                    self.exp = save_filters(self.exp, saved)
                except (OSError, ValueError) as error:
                    ui.notify(f"Not saved: {_reason(error)}", type="negative")
        self.update()

    def show_rule(self, rule: _Rule) -> None:
        self.shown = rule
        self.update()

    def add(self) -> None:
        taken = {rule.name for rule in self.rules}
        number = next(n for n in itertools.count(1) if NEW_NAME.format(number=n) not in taken)
        rule = _Rule(
            name=NEW_NAME.format(number=number),
            use=True,
            kind=FilterRule.MIDDLE_PERCENTILES,
            feature=NEW_FEATURE,
            per_sample=True,
            numbers=dict(STARTS[FilterRule.MIDDLE_PERCENTILES]),
        )
        self.rules.append(rule)
        self.shown = rule
        self.refresh()

    def delete(self, rule: _Rule) -> None:
        self.rules.remove(rule)
        if self.shown is rule:
            self.shown = self.rules[0] if self.rules else None
        self.refresh()

    def save(self) -> None:
        names = Counter(rule.name for rule in self.rules if rule.name)
        twice = next((name for name, count in names.items() if count > 1), None)
        if twice is not None:
            ui.notify(f'Two filters are named "{twice}". Rename one, then save.', type="negative")
            return
        if any(rule.kept is None for rule in self.rules):
            ui.notify(FIX_RED, type="negative")
            return
        try:
            self.exp = save_filters(self.exp, [rule.saved() for rule in self.rules])
        except (OSError, ValueError) as error:
            ui.notify(f"Not saved: {_reason(error)}", type="negative")
            return
        for index, rule in enumerate(self.rules):
            rule.saved_at = index
        self.unreadable = ""
        self.unreadable_line.set_visibility(False)
        count = len(self.rules)
        noun = "filter" if count == 1 else "filters"
        ui.notify(f"Saved {count} {noun} to experiment.json.", type="positive")

    def update(self) -> None:
        table = self.table
        for rule in self.rules:
            try:
                rule.kept, rule.problem = keep(table, rule.saved()), ""
            except (ValueError, TypeError) as error:
                rule.kept, rule.problem = None, str(error)
        switched_on = [rule.saved() for rule in self.rules if rule.use and rule.kept is not None]
        included, _ = apply_filters(table, switched_on)
        total, kept = len(table), int(included.sum())
        if any(rule.use for rule in self.rules):
            self.total.text = (
                f"Included: {kept:,} of {total:,} cells. The filters switched on leave out {total - kept:,}."
            )
        else:
            self.total.text = f"Included: {total:,} of {total:,} cells. No filter is switched on."
        for rule, card, count in self.cards:
            card.style(replace=SHOWN if rule is self.shown else "")
            if rule.kept is None:
                count.text = rule.problem
                count.classes(replace=PROBLEM)
            else:
                left_out = int((~rule.kept).sum())
                share = left_out / total * 100 if total else 0.0
                count.text = f"Leaves out {_cells(left_out)} ({share:.1f} %)."
                count.classes(replace=COUNT)
        self.draw()

    def draw(self) -> None:
        rule = self.shown
        if rule is None or rule.kept is None:
            self.figure = None
            self.plot.set_visibility(False)
            self.message.text = NO_RULE if rule is None else rule.problem
            self.message.classes(replace="text-sm muted" if rule is None else "text-sm text-negative")
            self.message.set_visibility(True)
            return
        self.figure = self.histogram(rule, rule.kept)
        self.message.set_visibility(False)
        self.plot.set_visibility(True)
        self.show()

    def histogram(self, rule: _Rule, kept: pd.Series) -> go.Figure:
        numbers = _numbers(self.table, rule.feature)
        finite = np.isfinite(numbers)
        marked = kept.to_numpy(dtype=bool)
        fig = go.Figure()
        if finite.any():
            low, high = np.percentile(numbers[finite], HISTOGRAM_PERCENTILES)
            if not high > low:
                low, high = low - 0.5, high + 0.5
            edges = np.linspace(low, high, BINS + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            width = float(edges[1] - edges[0])
            for name, mask, color in (
                ("kept", marked & finite, KEPT_COLOR),
                ("left out", ~marked & finite, LEFT_OUT_COLOR),
            ):
                counts, _ = np.histogram(numbers[mask], bins=edges)
                fig.add_bar(x=centers, y=counts, width=width, name=name, marker_color=color)
        feature = self.cells.features.get(rule.feature or "")
        fig.update_layout(
            template=TEMPLATE,
            barmode="stack",
            bargap=0,
            font_family=FONT,
            title_text=rule.name or "",
            xaxis_title_text=feature.title if feature is not None else rule.feature,
            yaxis_title_text="Cells",
            legend=LEGEND,
            margin=MARGIN,
        )
        return fig

    def show(self) -> None:
        if self.figure is None:
            return
        fig = go.Figure(self.figure)
        restyle(fig, dark=STATE.dark)
        self.dark = STATE.dark
        self.plot.update_figure({**fig.to_plotly_json(), "config": {"displaylogo": False}})

    def follow_mode(self) -> None:
        if self.figure is not None and self.dark != STATE.dark:
            self.show()


@ui.page("/filters")
def filters_page() -> None:
    with page_frame("Filters"):
        exp = STATE.experiment
        if exp is None:
            with ui.row().classes("items-center gap-1 text-sm"):
                ui.label("No experiment open.")
                ui.link("Open one on the Experiment page.", "/")
            return
        cells = load_cells(exp)
        if not cells.measured:
            empty_state(cells)
            return
        _FiltersPage(exp, cells, marker_menu(cells.features, read_info_file(exp))).build()
