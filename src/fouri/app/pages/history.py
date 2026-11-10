"""The History page."""

from nicegui import ui

from fouri.app.shell import page_frame


@ui.page("/history")
def history_page() -> None:
    with page_frame("History"):
        ui.label("History").classes("page-title")
        ui.label("Not built yet.").classes("muted")
