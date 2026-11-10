"""The Labels & gates page."""

from nicegui import ui

from fouri.app.shell import page_frame


@ui.page("/labels")
def labels_page() -> None:
    with page_frame("Labels & gates"):
        ui.label("Labels & gates").classes("page-title")
        ui.label("Not built yet.").classes("muted")
