"""The Batch effects page."""

from nicegui import ui

from fouri.app.shell import page_frame


@ui.page("/batch")
def batch_page() -> None:
    with page_frame("Batch effects"):
        ui.label("Batch effects").classes("page-title")
        ui.label("Not built yet.").classes("muted")
