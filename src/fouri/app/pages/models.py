"""The Models page."""

from nicegui import ui

from fouri.app.shell import page_frame


@ui.page("/models")
def models_page() -> None:
    with page_frame("Models"):
        ui.label("Models").classes("page-title")
        ui.label("Not built yet.").classes("muted")
