"""The Models page."""

from __future__ import annotations

from nicegui import run, ui

from fouri.app.dialogs import ask
from fouri.app.finetuning import Regions, corrected_regions, delete_model
from fouri.app.plate import wells_card
from fouri.app.runner import Run, StepRequest
from fouri.app.shell import page_frame
from fouri.app.state import POLL_SECONDS, STATE, kept_choices, run_going
from fouri.app.steps import BUILT_IN_MODEL, default, models_left_out
from fouri.config import Experiment
from fouri.metadata import normalize_well, plate_sort_key
from fouri.models import KeptModel, kept_models, shown_name
from fouri.parallel import error_in_words
from fouri.status import Step

SECTION_TITLE = "text-[1.05rem] font-semibold"
SUBHEADING = "font-medium mt-3"
DEFAULT_LINE = f"Segment nuclei uses {BUILT_IN_MODEL} unless its Model setting chooses a fine-tuned model."
COUNTING = "Counting corrected regions…"
NO_REGIONS = "No corrected regions saved. Correct nuclei in the segmentation editor first."
REGIONS_FAILED = "The corrected regions could not be counted: {error}"
NO_MODELS = "No fine-tuned models yet."
MODEL_COLUMNS = ("Created", "Experiment", "Wells", "")
MODEL_GRID = "auto auto minmax(0,1fr) auto"
DELETE = "Delete {name}? This can't be undone."
DELETED = "Deleted {name}."
EPOCHS_ERROR = "Enter a whole number of 1 or more."
PICK_WELL = "Pick at least one well."
CHECKING_GPU = "Checking for a GPU…"
NO_GPU = "Fine-tuning needs a CUDA GPU, and this PC has none."


def regions_text(regions: Regions) -> str:
    if not regions.count:
        return NO_REGIONS
    wells = ", ".join(f"{well} ({', '.join(names)})" for well, names in regions.rounds.items())
    return f"Corrected regions saved: {regions.count}, in {wells}."


def epochs(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not float(value).is_integer() or value < 1:
        return None
    return int(value)


class _ModelsPage:
    def __init__(self, exp: Experiment) -> None:
        self.exp = exp
        wells = sorted({normalize_well(well) for well in exp.well_list}, key=plate_sort_key)
        self.kept = kept_choices("models")
        was_picked = self.kept.get("picked", {})
        self.picked = {well: was_picked.get(well, True) for well in wells}
        self.kept["picked"] = self.picked
        self.kept.setdefault("epochs", default(Step.FINETUNE, "n_epochs"))
        self.delete_buttons: list[ui.button] = []

    def build(self) -> None:
        ui.label("Models").classes("page-title")
        with ui.card().classes("w-full max-w-3xl p-5 gap-2"):
            ui.label("Cellpose").classes(SECTION_TITLE)
            ui.label(DEFAULT_LINE).classes("text-sm")
            with ui.row().classes("items-center gap-2"):
                self.spinner = ui.spinner(size="sm")
                self.regions_line = ui.label(COUNTING).classes("text-sm")
            ui.label("Fine-tuned models").classes(SUBHEADING)
            self.models = ui.column().classes("w-full gap-1")
            ui.label("Fine-tune").classes(SUBHEADING)
            wells_card(self.exp, self.picked)
            self.epochs = (
                ui.number(
                    "Epochs",
                    value=self.kept["epochs"],
                    validation={EPOCHS_ERROR: lambda value: epochs(value) is not None},
                    on_change=self.new_epochs,
                )
                .props("step=1")
                .classes("w-56")
            )
            with ui.row().classes("items-center gap-3 mt-2"):
                self.button = ui.button("Fine-tune", icon="model_training", on_click=self.fine_tune)
                self.gpu_line = ui.label().classes("text-sm muted")
        self.show_models()
        self.follow()
        ui.timer(0.1, self.count_regions, once=True)
        ui.timer(POLL_SECONDS, self.follow)

    async def count_regions(self) -> None:
        try:
            regions = await run.io_bound(corrected_regions, self.exp)
        except Exception as error:
            self.regions_line.text = REGIONS_FAILED.format(error=error_in_words(error))
            self.regions_line.classes(replace="text-sm text-negative")
        else:
            self.regions_line.text = regions_text(regions)
        finally:
            self.spinner.visible = False

    def show_models(self) -> None:
        models, _ = kept_models()
        left_out = models_left_out()
        self.delete_buttons = []
        self.models.clear()
        with self.models:
            if not models:
                ui.label(NO_MODELS).classes("text-sm muted")
            else:
                with ui.grid(columns=MODEL_GRID).classes("w-full items-center gap-x-6 gap-y-1"):
                    for title in MODEL_COLUMNS:
                        ui.label(title).classes("text-xs muted font-medium")
                    for model in models:
                        self.model_row(model)
            if left_out:
                ui.label(left_out).classes("text-xs text-warning")
        self.follow()

    def model_row(self, model: KeptModel) -> None:
        ui.label(f"{model.created:%Y-%m-%d %H:%M}").classes("text-sm tabular-nums whitespace-nowrap")
        ui.label(model.experiment_key).classes("text-sm break-all")
        ui.label(", ".join(model.wells)).classes("text-sm")
        button = ui.button("Delete", on_click=lambda: self.delete(model))
        button.props("flat dense no-caps color=negative")
        self.delete_buttons.append(button)

    def new_epochs(self) -> None:
        self.kept["epochs"] = self.epochs.value
        self.follow()

    def follow(self) -> None:
        going = run_going()
        for button in self.delete_buttons:
            if button.enabled == going:
                button.set_enabled(not going)
        allowed = (
            not going and not STATE.checking_gpu and STATE.gpu is not None and epochs(self.epochs.value) is not None
        )
        if self.button.enabled != allowed:
            self.button.set_enabled(allowed)
        if STATE.checking_gpu:
            text, classes = CHECKING_GPU, "text-sm muted"
        elif STATE.gpu is None:
            text, classes = NO_GPU, "text-sm text-warning"
        else:
            text, classes = "", "text-sm muted"
        if self.gpu_line.text != text:
            self.gpu_line.text = text
            self.gpu_line.classes(replace=classes)
        self.gpu_line.set_visibility(bool(text))

    async def delete(self, model: KeptModel) -> None:
        if run_going():
            return
        name = shown_name(model)
        if not await ask(DELETE.format(name=name), "Delete") or run_going():
            return
        try:
            await run.io_bound(delete_model, model.name)
        except OSError as error:
            ui.notify(error_in_words(error), type="negative")
        else:
            ui.notify(DELETED.format(name=name), type="positive")
        self.show_models()

    def fine_tune(self) -> None:
        n_epochs = epochs(self.epochs.value)
        if run_going() or STATE.checking_gpu or STATE.gpu is None or n_epochs is None:
            return
        wells = [well for well, on in self.picked.items() if on]
        if not wells:
            ui.notify(PICK_WELL, type="warning")
            return
        try:
            started = Run(self.exp.config_path, [StepRequest(Step.FINETUNE, wells, {"n_epochs": n_epochs})])
            started.start()
        except OSError as error:
            ui.notify(f"The run could not start: {error}", type="negative")
            return
        STATE.run = started
        ui.navigate.to("/run")


@ui.page("/models")
def models_page() -> None:
    with page_frame("Models"):
        exp = STATE.experiment
        if exp is None:
            with ui.row().classes("items-center gap-1 text-sm"):
                ui.label("No experiment open.")
                ui.link("Open one on the Experiment page.", "/")
            return
        _ModelsPage(exp).build()
