"""The Review in napari page."""

from nicegui import ui

from fouri.app.shell import page_frame


@ui.page("/review")
def review_page() -> None:
    with page_frame("Review in napari"):
        ui.label("Review in napari").classes("page-title")
        ui.label("Not built yet.").classes("muted")
