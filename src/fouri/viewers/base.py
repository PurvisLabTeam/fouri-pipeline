"""@TODO."""

from __future__ import annotations

import abc
import textwrap
from collections.abc import Callable
from enum import StrEnum

import napari
import pandas as pd
from magicgui.widgets import ComboBox, Container, Label, PushButton, Widget
from qtpy.QtCore import QEvent, QObject, QTimer
from qtpy.QtWidgets import QMessageBox

from fouri.config import Experiment
from fouri.io import read_round_table

# Max character width of the prompt overlay panel
PROMPT_WIDTH = 70


class KeyboardShortcuts(StrEnum):
    SAVE = "Shift+S"
    UNDO = "Ctrl+Z"
    ROUND_PREVIOUS = "Up"
    ROUND_NEXT = "Down"
    WELL_PREVIOUS = "Left"
    WELL_NEXT = "Right"
    ROUND_FIRST = "Shift+Up"
    ROUND_LAST = "Shift+Down"
    WELL_FIRST = "Shift+Left"
    WELL_LAST = "Shift+Right"


class BaseViewer(abc.ABC):
    viewer: napari.Viewer
    exp: Experiment
    well: str
    title: str
    round_table: pd.DataFrame
    round_names: list[str]
    round_name: str
    well_picker: ComboBox
    round_picker: ComboBox

    _close_guard: _CloseGuard
    _self_close: bool
    asks_on_round_switch: bool = False

    def __init__(self, exp: Experiment, well: str, round_name: str | None = None) -> None:
        self.exp = exp
        self.viewer = napari.Viewer()
        self._build_layers()

        self.well_picker = ComboBox(choices=exp.well_list, label="Well")
        self.well_picker.value = well
        self.round_picker = ComboBox(choices=[], label="Round")

        self.well_picker.changed.connect(self._on_well_picked)
        self.round_picker.changed.connect(self._on_round_picked)
        self._self_close = False

        button_up = PushButton(text="▲")
        button_down = PushButton(text="▼")
        button_left = PushButton(text="◀")
        button_right = PushButton(text="▶")

        button_up.changed.connect(lambda: self.step_round(-1))
        button_down.changed.connect(lambda: self.step_round(1))
        button_left.changed.connect(lambda: self.step_well(-1))
        button_right.changed.connect(lambda: self.step_well(1))

        pad = Container(
            widgets=[
                Container(widgets=[button_up], layout="horizontal"),
                Container(widgets=[button_left, button_right], layout="horizontal"),
                Container(widgets=[button_down], layout="horizontal"),
            ],
            labels=False,
        )

        button_save = PushButton(text=f"Save ({KeyboardShortcuts.SAVE})")
        button_undo = PushButton(text=f"Undo ({KeyboardShortcuts.UNDO})")

        button_save.changed.connect(self.save)
        button_undo.changed.connect(self.undo)

        save_undo = Container(widgets=[button_undo, button_save], layout="horizontal")

        keys = KeyboardShortcuts
        shortcut_lines = [
            "Shortcuts",
            f"{keys.ROUND_PREVIOUS} / {keys.ROUND_NEXT}: previous / next round",
            f"{keys.ROUND_FIRST} / {keys.ROUND_LAST}: first / last round",
            f"{keys.WELL_PREVIOUS} / {keys.WELL_NEXT}: previous / next well",
            f"{keys.WELL_FIRST} / {keys.WELL_LAST}: first / last well",
            "Scroll: zoom",
            "Space + drag: pan",
            *self._extra_shortcuts(),
        ]
        shortcut_help = Label(value="\n".join(shortcut_lines), label="")

        panel = Container(
            widgets=[self.well_picker, self.round_picker, pad, *self._tool_widgets(), save_undo, shortcut_help],
        )
        panel.min_width = 250
        self.viewer.window.add_dock_widget(panel.native, area="right", name=self.title)

        self._close_guard = _CloseGuard(self)
        self.viewer.window._qt_window.installEventFilter(self._close_guard)

        self.viewer.canvas.overlays.text.visible = True
        self._bind_keys()
        self.show_well(well, round_name if round_name is not None else self._start_round())

    def show_well(self, well: str, round_name: str | None = None) -> None:
        round_table = read_round_table(self.exp, well)
        choices = self._round_choices(round_table)
        refused = None
        if round_name is not None and round_name not in choices:
            refused = round_name
            round_name = choices[0]
        current = getattr(self, "round_name", None)
        target = round_name if round_name is not None else current if current in choices else choices[0]
        previous = (self.well, self.round_table, self.round_names, self.round_name) if hasattr(self, "well") else None
        try:
            self._enter_well(well, round_table, choices, target)
        except Exception:
            if previous is not None:
                self._enter_well(*previous)
            raise
        if refused is not None:
            self.prompt(f"{refused} is not a recognized round for {well}. Opened {round_name} instead.")

    def _enter_well(self, well: str, round_table: pd.DataFrame, choices: list[str], round_name: str) -> None:
        self._load_well(well, round_table)
        self.well = well
        self.round_table = round_table
        self.round_names = choices

        with self.round_picker.changed.blocked():
            self.round_picker.choices = choices

        self._set_quiet(self.well_picker, well)
        self.show_round(round_name)

    def show_round(self, round_name: str) -> None:
        self._load_round(round_name)
        self.round_name = round_name
        self._set_quiet(self.round_picker, round_name)

    def _on_well_picked(self, well: str) -> None:
        may_leave = self._may_leave()
        if not may_leave:
            self._set_quiet(self.well_picker, self.well)
            return
        try:
            self.show_well(well)
        except (FileNotFoundError, ValueError) as error:
            self.prompt(f"Could not open well {well}: {error}")
            self._set_quiet(self.well_picker, self.well)

    @staticmethod
    def _set_quiet(picker: ComboBox, value: str) -> None:
        with picker.changed.blocked():
            picker.value = value

    def _on_round_picked(self, round_name: str) -> None:
        if self.asks_on_round_switch and not self._may_leave():
            self._set_quiet(self.round_picker, self.round_name)
            return
        try:
            self.show_round(round_name)
        except (FileNotFoundError, ValueError) as error:
            self.prompt(f"Could not open round {round_name}: {error}")
            self._set_quiet(self.round_picker, self.round_name)

    def step_round(self, step: int) -> None:
        self._go_to_round(self.current_round_position + step)

    def step_well(self, step: int) -> None:
        self._go_to_well(self.current_well_position + step)

    def prompt(self, message: str) -> None:
        self.viewer.canvas.overlays.text.text = textwrap.fill(message, width=PROMPT_WIDTH)

    def _start_round(self) -> str | None:
        return None

    def _extra_shortcuts(self) -> list[str]:
        return []

    def _unsaved_rounds(self) -> list[str]:
        return []

    def _may_leave(self) -> bool:
        unsaved = self._unsaved_rounds()
        if unsaved == []:
            return True
        msg_box = QMessageBox(self.viewer.window._qt_window)
        msg_box.setText(f"{self.well} has unsaved edits in {', '.join(unsaved)}. Save before leaving?")
        msg_box.setStandardButtons(
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        result = msg_box.exec()

        match result:
            case QMessageBox.StandardButton.Save:
                self.save()
                return not self._unsaved_rounds()
            case QMessageBox.StandardButton.Discard:
                return True
            case _:
                return False

    def _bind_keys(self) -> None:
        keys: dict[str, Callable[[], None]] = {
            KeyboardShortcuts.SAVE: self.save,
            KeyboardShortcuts.UNDO: self.undo,
            KeyboardShortcuts.ROUND_PREVIOUS: lambda: self.step_round(-1),
            KeyboardShortcuts.ROUND_NEXT: lambda: self.step_round(1),
            KeyboardShortcuts.ROUND_FIRST: lambda: self._go_to_round(0),
            KeyboardShortcuts.ROUND_LAST: lambda: self._go_to_round(len(self.round_names) - 1),
            KeyboardShortcuts.WELL_PREVIOUS: lambda: self.step_well(-1),
            KeyboardShortcuts.WELL_NEXT: lambda: self.step_well(1),
            KeyboardShortcuts.WELL_FIRST: lambda: self._go_to_well(0),
            KeyboardShortcuts.WELL_LAST: lambda: self._go_to_well(len(self.exp.well_list) - 1),
        }
        keys |= self._extra_keys()
        for key, action in keys.items():
            self.viewer.bind_key(key, self._on_key(action), overwrite=True)

    @staticmethod
    def _on_key(action: Callable[[], None]) -> Callable[[napari.Viewer], None]:
        return lambda _viewer: action()

    def _go_to_round(self, index: int) -> None:
        if index < 0 or index >= len(self.round_names) or index == self.current_round_position:
            return
        self.round_picker.value = self.round_names[index]

    def _go_to_well(self, index: int) -> None:
        if index < 0 or index >= len(self.exp.well_list) or index == self.current_well_position:
            return
        self.well_picker.value = self.exp.well_list[index]

    @property
    def current_round_position(self) -> int:
        return self.round_names.index(self.round_name)

    @property
    def current_well_position(self) -> int:
        return self.exp.well_list.index(self.well)

    @abc.abstractmethod
    def save(self) -> None: ...

    @abc.abstractmethod
    def undo(self) -> None: ...

    @abc.abstractmethod
    def _build_layers(self) -> None: ...

    @abc.abstractmethod
    def _tool_widgets(self) -> list[Widget]: ...

    @abc.abstractmethod
    def _round_choices(self, round_table: pd.DataFrame) -> list[str]: ...

    @abc.abstractmethod
    def _load_well(self, well: str, round_table: pd.DataFrame) -> None: ...

    @abc.abstractmethod
    def _load_round(self, round_name: str) -> None: ...

    @abc.abstractmethod
    def _extra_keys(self) -> dict[str, Callable[[], None]]: ...


class _CloseGuard(QObject):
    def __init__(self, editor: BaseViewer) -> None:
        super().__init__()
        self.editor = editor

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:  # noqa: N802
        if event is None or event.type() != QEvent.Type.Close:
            return False
        if self.editor._self_close or self.editor._unsaved_rounds() == []:
            return False
        may_leave = self.editor._may_leave()
        if may_leave:
            self.editor._self_close = True
            QTimer.singleShot(0, self.editor.viewer.close)

        event.ignore()
        return True
