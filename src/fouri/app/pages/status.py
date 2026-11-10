"""The Status page."""

from nicegui import ui

from fouri.app.shell import page_frame


@ui.page("/status")
def status_page() -> None:
    with page_frame("Status"):
        ui.label("Status").classes("page-title")
        ui.label("Not built yet.").classes("muted")
