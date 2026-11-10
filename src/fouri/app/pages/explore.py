"""The Explore page."""

from nicegui import ui

from fouri.app.shell import page_frame


@ui.page("/explore")
def explore_page() -> None:
    with page_frame("Explore"):
        ui.label("Explore").classes("page-title")
        ui.label("Not built yet.").classes("muted")
