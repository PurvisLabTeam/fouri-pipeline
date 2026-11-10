"""The Experiment page."""

from nicegui import ui

from fouri.app.shell import page_frame


@ui.page("/")
def experiment_page() -> None:
    with page_frame("Experiment"):
        ui.label("Experiment").classes("page-title")
        ui.label("Not built yet.").classes("muted")
