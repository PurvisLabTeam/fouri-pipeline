"""Picking a file or a folder on this PC.

In the app's own window this opens the system's file dialog. In a browser tab a page cannot
learn a file's path, so an in-app dialog lists this PC's folders instead.
"""

from __future__ import annotations

import os
from pathlib import Path

from nicegui import ui


def _start_folder(start: str | Path) -> Path:
    """Return the folder a dialog opens in: ``start`` or the nearest folder above it, else home."""
    path = Path(start).expanduser() if str(start).strip() else Path.home()
    for folder in (path, *path.parents):
        if folder.is_dir():
            return folder
    return Path.home()


def _named_file(folder: Path, name: str | None) -> Path | None:
    if name is None:
        return None
    try:
        files = sorted((entry for entry in folder.iterdir() if entry.is_file()), key=lambda p: p.name)
    except OSError:
        return None
    matches = [entry for entry in files if entry.name.lower() == name.lower()]
    for entry in matches:
        if entry.name == name:
            return entry
    return matches[0] if matches else None


async def _listing_dialog(start: Path, *, suffix: str | None, name: str | None = None) -> Path | None:
    """List folders, and ``suffix`` files when picking a file; return the pick or None.

    The box at the top shows the folder listed. Enter on a path typed there goes to that folder,
    or picks a ``suffix`` file; a path without a drive or root is read against the folder listed.
    """
    here = start
    with ui.dialog() as dialog, ui.card().classes("w-[560px]"):
        where = ui.input().classes("w-full").props("dense")
        listing = ui.column().classes("w-full gap-0 max-h-96 overflow-auto")

        def show() -> None:
            where.value = str(here)
            listing.clear()
            try:
                entries = sorted(here.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                entries = []
            with listing:
                if here.parent != here:
                    ui.item("..", on_click=lambda: go(here.parent)).props("clickable dense")
                for entry in entries:
                    if entry.name.startswith("."):
                        continue
                    if entry.is_dir():
                        with ui.item(on_click=lambda e=entry: go(e)).props("clickable dense"):
                            ui.icon("folder").classes("mr-2 text-amber-500")
                            ui.label(entry.name)
                    elif suffix is not None and entry.name.lower().endswith(suffix.lower()):
                        with ui.item(on_click=lambda e=entry: dialog.submit(e)).props("clickable dense"):
                            ui.icon("description").classes("mr-2 muted")
                            ui.label(entry.name)

        def go(path: Path) -> None:
            nonlocal here
            here = path
            show()

        def go_typed() -> None:
            # normpath, not resolve: resolve would turn a mapped drive (Z:) into its network name
            path = Path(os.path.normpath(here / Path(where.value.strip()).expanduser()))
            if path.is_dir():
                sought = _named_file(path, name)
                if sought is not None:
                    dialog.submit(sought)
                    return
                go(path)
            elif path.is_file():
                if suffix is not None and path.name.lower().endswith(suffix.lower()):
                    dialog.submit(path)
                    return
                kind = "is a file" if suffix is None else f"isn't a {suffix} file"
                ui.notify(f"{path.name} {kind}; showing its folder.", type="warning")
                go(path.parent)
            else:
                ui.notify(f"No such folder: {path}", type="negative")
                where.value = str(here)

        where.on("keydown.enter", go_typed)
        show()
        with ui.row().classes("w-full justify-end mt-2"):
            ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat")
            if suffix is None:
                ui.button("Use this folder", on_click=lambda: dialog.submit(here))
    chosen = await dialog
    dialog.delete()
    return chosen if isinstance(chosen, Path) else None


async def _pick(start: str | Path, *, suffix: str | None, name: str | None = None) -> Path | None:
    return await _listing_dialog(_start_folder(start), suffix=suffix, name=name)


async def pick_file(start: str | Path, *, suffix: str, name: str | None = None) -> Path | None:
    """Return a ``suffix`` file picked in a dialog opened at ``start``, or None if canceled."""
    return await _pick(start, suffix=suffix, name=name)


async def pick_folder(start: str | Path) -> Path | None:
    """Return a folder picked in a dialog opened at ``start``, or None if canceled."""
    return await _pick(start, suffix=None)


async def ask(question: str, yes: str) -> bool:
    with ui.dialog() as dialog, ui.card().classes("p-5 gap-3 max-w-lg"):
        ui.label(question).classes("text-sm")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
            ui.button(yes, on_click=lambda: dialog.submit(True))
    answer = await dialog
    dialog.delete()
    return answer is True
