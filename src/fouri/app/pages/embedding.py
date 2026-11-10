"""The Embedding page."""

from nicegui import ui

from fouri.app.shell import page_frame


@ui.page("/embedding")
def embedding_page() -> None:
    with page_frame("Embedding"):
        ui.label("Embedding").classes("page-title")
        ui.label("Not built yet.").classes("muted")
