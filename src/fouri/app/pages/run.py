"""The Run page."""

from nicegui import ui

from fouri.app.shell import page_frame


@ui.page("/run")
def run_page() -> None:
    with page_frame("Run"):
        ui.label("Run").classes("page-title")
        ui.label("Not built yet.").classes("muted")
