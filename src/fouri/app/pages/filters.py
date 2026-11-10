"""The Filters page."""

from nicegui import ui

from fouri.app.shell import page_frame


@ui.page("/filters")
def filters_page() -> None:
    with page_frame("Filters"):
        ui.label("Filters").classes("page-title")
        ui.label("Not built yet.").classes("muted")
