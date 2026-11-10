"""The app's pages, one module each; importing this package registers every page's address."""

from fouri.app.pages import (
    batch,
    embedding,
    experiment,
    explore,
    filters,
    history,
    labels,
    models,
    review,
    run,
    status,
)

__all__ = [
    "batch",
    "embedding",
    "experiment",
    "explore",
    "filters",
    "history",
    "labels",
    "models",
    "review",
    "run",
    "status",
]
