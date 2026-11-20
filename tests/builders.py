"""Write what each pipeline step would have written, for a small made-up experiment.

A test arranges a state in its temporary folder with these, without ND2 files, Cellpose or a GPU.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from tifffile import imread

from fouri import io, measure, preprocess
from fouri.config import Experiment, load_config, save_config
from fouri.parallel import Failed, error_in_words

INFO_EXP = """\
myRound,DAPI,Cy3,Cy5
R0,DNA,pRB,RB
R1,DNA,cMyc,
R2,DNA,CyclinA2,EdU
"""

CONFIG = {
    "experiment_key": "test",
    "experiment_name": "made-up",
    "well_list": ["B02", "B03", "C02"],
    "well_metadata": {
        "B02": {"treatment": "control", "treatment_time": "24h", "sample_ID": 1},
        "B03": {"treatment": "palbo", "treatment_time": "24h", "sample_ID": 2},
        "C02": {"treatment": "palbo", "treatment_time": "96h", "sample_ID": 3},
    },
    "condition_columns": ["treatment", "treatment_time"],
    "derived_features": [{"name": "pRB_RB_ratio", "formula": "R0_pRB_nuc_mean / R0_RB_nuc_mean"}],
}

FOUR_WELLS = {
    **CONFIG,
    "experiment_key": "four",
    "well_list": ["B02", "B03", "C02", "C03"],
    "well_metadata": {
        "B02": {"treatment": "control", "treatment_time": "24h", "batch": "1", "sample_ID": 1},
        "B03": {"treatment": "control", "treatment_time": "24h", "batch": "2", "sample_ID": 2},
        "C02": {"treatment": "palbo", "treatment_time": "24h", "batch": "1", "sample_ID": 3},
        "C03": {"treatment": "palbo", "treatment_time": "24h", "batch": "2", "sample_ID": 4},
    },
}

SWAPPED_INFO = """\
myRound,DAPI,Cy3,Cy5
R0,DNA,pRB,RB
R2,DNA,CyclinA2,EdU
R1,DNA,cMyc,
"""

ND2_FOLDER = "Round_{round_name}_made_up"
ND2_FILE = "Well{well}_ChannelDAPI,Cy3,Cy5_Seq0000.nd2"

REMOVED_NUCLEUS = 4

SIZE = 64  # every image is SIZE x SIZE pixels

# Four 6 x 6 nuclei, one in each quarter of the image. Each sits in a bright square reaching 8 px
# past it: room for its ring (6 px wide after a 1 px gap) and the 1 px margin Measure cells keeps
# from pixels with no image data, so no cell is dropped.
NUCLEUS_STARTS = (13, 45)
NUCLEUS_SIDE = 6
SQUARE_MARGIN = 8

# Made-up acquisition values per optical configuration: excitation nm, emission nm, exposure ms.
ACQUISITION = {"DAPI": (405.0, 450.0, 100.0), "Cy3": (555.0, 590.0, 200.0), "Cy5": (640.0, 690.0, 300.0)}


def _squares(margin: int) -> list[tuple[slice, slice]]:
    """Return each nucleus's rows and columns, widened by ``margin`` on every side."""
    return [
        (slice(row - margin, row + NUCLEUS_SIDE + margin), slice(col - margin, col + NUCLEUS_SIDE + margin))
        for row in NUCLEUS_STARTS
        for col in NUCLEUS_STARTS
    ]


def _round_count(exp: Experiment) -> int:
    return len(io.read_info_file(exp))


def _brightness(channel_in_file: int, round_index: int) -> int:
    return 1000 * (channel_in_file + 1) + 100 * round_index


def _image(value: int, shape: tuple[int, int] = (SIZE, SIZE), per_cell: bool = False) -> npt.NDArray[np.uint16]:
    image = np.zeros(shape, dtype=np.uint16)
    for label, (square, nucleus) in enumerate(zip(_squares(SQUARE_MARGIN), _squares(0), strict=True), start=1):
        if per_cell:
            image[square] = value + 10 * label + 5
            image[nucleus] = value + 10 * label
        else:
            image[square] = value
    return image


def _nuclei(shape: tuple[int, ...] = (SIZE, SIZE)) -> npt.NDArray[np.uint16]:
    labels = np.zeros(shape, dtype=np.uint16)
    for label, square in enumerate(_squares(0), start=1):
        labels[square] = label
    return labels


def _write_experiment(folder: Path, config: Mapping[str, Any]) -> Experiment:
    folder = Path(folder)
    (folder / "info_exp.csv").write_text(INFO_EXP, encoding="utf-8")
    config_path = folder / "experiment.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return load_config(config_path, announce=False)


def make_experiment(folder: Path) -> Experiment:
    """Write info_exp.csv and experiment.json for three wells and three rounds, then load both."""
    return _write_experiment(folder, CONFIG)


def four_wells(folder: Path) -> Experiment:
    return _write_experiment(folder, FOUR_WELLS)


def converted(exp: Experiment, well: str, rounds: int | None = None, *, per_cell: bool = False) -> None:
    """Write the round table and each channel's 64 x 64 image for the first ``rounds`` rounds."""
    info = io.read_info_file(exp)
    rows = []
    brightness = {}  # each image's file name and the value of its bright squares
    for round_index, (_, round_info) in enumerate(info.iloc[:rounds].iterrows()):
        round_name = str(round_info["round_name"])
        for channel_in_file, channel in enumerate(info.columns[1:]):
            cell = round_info[channel]
            if pd.isna(cell) or not str(cell).strip():
                continue
            signal = str(cell).strip()
            raw_file_name = f"Round_{round_name}_well{well}_{signal}.tif"
            brightness[raw_file_name] = _brightness(channel_in_file, round_index)
            excitation, emission, exposure = ACQUISITION[channel]
            rows.append(
                {
                    "dir": str(exp.data_dir),
                    "sub_dir": ND2_FOLDER.format(round_name=round_name),
                    "file": ND2_FILE.format(well=well),
                    "round_name": round_name,
                    "round_index": round_index,
                    "well": well,
                    "width": SIZE,
                    "height": SIZE,
                    "pixel_size_um": 0.65,
                    "objective": "Plan Apo 20x",
                    "bit_depth": 16,
                    "acquired_at": f"2026-11-0{round_index + 1} 10:00:00",
                    "channel_in_file": channel_in_file,
                    "channel_name": channel,
                    "signal": signal,
                    "excitation_nm": excitation,
                    "emission_nm": emission,
                    "exposure_ms": exposure,
                    "raw_file_name": raw_file_name,
                }
            )
    table = pd.DataFrame(rows)
    table["width_min"] = table["width"].min()
    table["height_min"] = table["height"].min()
    table["is_align_channel"] = table["signal"] == "DNA"

    Path(exp.output_df).mkdir(parents=True, exist_ok=True)
    io.save_round_table(exp, well, table)

    folder = Path(exp.output_images) / well
    folder.mkdir(parents=True, exist_ok=True)
    for raw_file_name, value in brightness.items():
        io.write_tiff(folder / raw_file_name, _image(value, per_cell=per_cell))


def segmented(exp: Experiment, well: str, rounds: Sequence[int] | None = None) -> None:
    """Write label images with four square nuclei for the given rounds, every round if None."""
    labels = _nuclei()
    for round_index in range(_round_count(exp)) if rounds is None else rounds:
        path = io.label_path(exp, well, round_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        io.write_tiff(path, labels)


def hand_edited(exp: Experiment, well: str, rounds: Sequence[int]) -> None:
    for round_index in rounds:
        path = io.label_path(exp, well, round_index)
        labels = imread(path)
        original = io.original_label_path(exp, well, round_index)
        original.parent.mkdir(parents=True, exist_ok=True)
        io.write_tiff(original, labels)
        io.write_tiff(path, np.where(labels == REMOVED_NUCLEUS, 0, labels).astype(labels.dtype))


def aligned(exp: Experiment, well: str, rounds: Sequence[int] | None = None) -> None:
    """Write transforms: identity for the anchor and the given rounds (all if None), else NaN."""
    count = _round_count(exp)
    identity = set(range(count) if rounds is None else rounds) | {exp.anchor_round}
    transforms = np.full((count, 3, 3), np.nan)
    for round_index in identity:
        transforms[round_index] = np.eye(3)
    Path(exp.output_df).mkdir(parents=True, exist_ok=True)
    io.save_transforms(exp, well, transforms)


def masked(
    exp: Experiment,
    well: str,
    round_index: int,
    regions: Mapping[str, tuple[slice, slice]],
    *,
    edited: bool = False,
) -> None:
    table = io.read_round_table(exp, well)
    signals = [str(signal) for signal in table.loc[table["round_index"] == round_index, "signal"]]
    unknown = sorted(set(regions) - set(signals))
    if unknown:
        raise ValueError(f"Round {round_index} of {well} has no channel {', '.join(unknown)}")
    arrays = {}
    for signal in signals:
        final = np.zeros((SIZE, SIZE), dtype=bool)
        outlines = np.zeros((SIZE, SIZE), dtype=np.uint16)
        if signal in regions:
            final[regions[signal]] = True
            outlines[regions[signal]] = 1
        arrays[f"final_{signal}"] = final
        arrays[f"outlines_{signal}"] = outlines
    masks: dict[str, dict[str, Any]] = {signal: {"spread": 0, "pad": 0} for signal in signals}
    if edited:
        for signal, entry in masks.items():
            entry["artifacts"] = {"1": {"spread": 0, "pad": 0}} if signal in regions else {}
    record = {"edited": edited, "masks": masks}
    path = io.mask_path(exp, well, round_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    io.mask_record_path(exp, well, round_index).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(path, **arrays)


def measured(exp: Experiment, well: str) -> None:
    """Run ``measure.measure_well`` on a well already converted, segmented and aligned."""
    measure.measure_well(exp, well)


def measured_values(exp: Experiment, well: str, column: str, values: Sequence[float]) -> None:
    path = io.cell_data_path(exp, well)
    table = pd.read_csv(path)
    table[column] = list(values)
    table.to_csv(path, index=False)


def subsampled(exp: Experiment, rows: Sequence[int], key: str, *, cell_table: bool = True) -> None:
    adata = ad.read_h5ad(io.adata_path(exp))[list(rows)].copy()
    dims = 3 if "_3d_" in key else 2
    adata.obsm[key] = np.arange(adata.n_obs * dims, dtype=float).reshape(adata.n_obs, dims) * 1.5 + 0.25
    features = list(adata.var_names[:2])
    adata.uns["subsample"] = {"cells_per_sample": adata.n_obs, "seed": 42}
    if cell_table and "cell_table" in adata.uns:
        adata.uns["subsample"]["cell_table"] = adata.uns["cell_table"]["id"]
    adata.uns["embeddings"] = {
        key: {
            "method": key.split("_")[1].upper(),
            "dims": dims,
            "top_n": len(features),
            "settings": {"knn": 5},
            "picked": features,
            "samples": sorted(int(sample) for sample in adata.obs["sample_ID"].unique()),
            "ranking_k": 10,
            "seed": 42,
            "features": features,
            "scores": np.linspace(0.1, 0.2, len(features)),
        }
    }
    with io.written_whole(io.adata_sub_path(exp)) as partial:
        adata.write_h5ad(partial)


def with_filters(exp: Experiment, filters: Sequence[Mapping[str, Any]]) -> Experiment:
    return _with_app_value(exp, "filters", [dict(rule) for rule in filters])


def with_labels(exp: Experiment, labels: Sequence[Mapping[str, Any]]) -> Experiment:
    return _with_app_value(exp, "labels", [dict(rule) for rule in labels])


def with_embedding_setup(exp: Experiment, setup: Mapping[str, Any]) -> Experiment:
    return _with_app_value(exp, "embedding", dict(setup))


def _with_app_value(exp: Experiment, key: str, value: Any) -> Experiment:
    app = {**exp.config.get("app", {}), key: value}
    save_config({**exp.config, "app": app}, exp.config_path)
    return load_config(exp.config_path, announce=False)


def with_metadata(exp: Experiment, well: str, **values: Any) -> Experiment:
    metadata = {name: dict(entry) for name, entry in (exp.config.get("well_metadata") or {}).items()}
    entry = metadata.setdefault(well, {})
    for name, value in values.items():
        if value is None:
            entry.pop(name, None)
        else:
            entry[name] = value
    save_config({**exp.config, "well_metadata": metadata}, exp.config_path)
    return load_config(exp.config_path, announce=False)


def in_process_map(
    function: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    desc: str,
    unit: str,
    n_workers: int | None = None,
    mem_per_worker: float = 0.0,
    names: Sequence[str] | None = None,
) -> list[Any]:
    names = [str(item) for item in items] if names is None else list(names)
    outcomes: list[Any] = []
    for name, item in zip(names, items, strict=True):
        try:
            outcomes.append(function(item))
        except Exception as error:
            outcomes.append(Failed(name, error_in_words(error)))
    return outcomes


def raw_files(
    exp: Experiment,
    monkeypatch: pytest.MonkeyPatch,
    sizes: Mapping[str, tuple[int, int]] | None = None,
) -> None:
    sizes = {} if sizes is None else sizes
    for round_name in io.read_info_file(exp)["round_name"]:
        folder = Path(exp.data_dir) / ND2_FOLDER.format(round_name=round_name)
        folder.mkdir(parents=True, exist_ok=True)
        for well in exp.well_list:
            (folder / ND2_FILE.format(well=well)).touch()

    def placed(path: str | Path) -> tuple[int, tuple[int, int]]:
        round_name = Path(path).parent.name.removeprefix("Round_").removesuffix("_made_up")
        names = [str(name) for name in io.read_info_file(exp)["round_name"]]
        round_index = names.index(round_name) if round_name in names else 0
        return round_index, sizes.get(round_name, (SIZE, SIZE))

    def nd2_metadata(path: str | Path) -> io.ND2Metadata:
        round_index, (height, width) = placed(path)
        return io.ND2Metadata(
            width=width,
            height=height,
            pixel_size_um=0.65,
            objective="Plan Apo 20x",
            bit_depth=16,
            acquired_at=f"2026-11-0{round_index + 1} 10:00:00",
            channels=[io.ChannelMetadata(channel, *values) for channel, values in ACQUISITION.items()],
        )

    def read_nd2(path: str | Path) -> npt.NDArray[np.uint16]:
        round_index, shape = placed(path)
        return np.stack(
            [_image(_brightness(channel_in_file, round_index), shape) for channel_in_file in range(len(ACQUISITION))]
        )

    monkeypatch.setattr(preprocess, "nd2_metadata", nd2_metadata)
    monkeypatch.setattr(preprocess, "read_nd2", read_nd2)
    monkeypatch.setattr(preprocess, "parallel_map", in_process_map)


class _StandInUMAP:
    def __init__(self, n_components: int = 2, **settings: Any) -> None:
        self.n_components = n_components
        self.settings = settings

    def fit_transform(self, values: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        values = np.asarray(values, dtype=np.float64)
        if np.isnan(values).any():
            raise ValueError("Stand-in UMAP given NaN values.")
        return values[:, [column % values.shape[1] for column in range(self.n_components)]]


def umap_module() -> ModuleType:
    module = ModuleType("umap")
    vars(module).update(UMAP=_StandInUMAP)
    return module


class _StandInNet:
    def register_forward_hook(self, hook: Callable[..., None]) -> SimpleNamespace:
        return SimpleNamespace(remove=lambda: None)


class _StandInCellposeModel:
    def __init__(self, gpu: bool = False, pretrained_model: str | Path | None = None, **settings: Any) -> None:
        self.gpu = gpu
        self.pretrained_model = pretrained_model
        self.device = SimpleNamespace(type="cuda" if gpu else "cpu")
        self.net = _StandInNet()

    def eval(self, image: npt.NDArray[Any], **settings: Any) -> tuple[npt.NDArray[np.uint16], None, None]:
        return _nuclei(image.shape), None, None


def _train_seg(
    net: Any,
    train_data: Sequence[npt.NDArray[Any]],
    train_labels: Sequence[npt.NDArray[Any]],
    *,
    save_path: str | Path,
    model_name: str,
    n_epochs: int = 100,
    **settings: Any,
) -> tuple[Path, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    log = logging.getLogger("cellpose.train")
    for epoch in range(n_epochs):
        if epoch in (0, 5) or epoch % 10 == 0:
            log.info("%d, train_loss=0.5000, test_loss=0.0000, LR=0.000050, time 1.00s", epoch)
    model_file = Path(save_path) / "models" / model_name
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(b"stand-in model")
    return model_file, np.zeros(n_epochs), np.zeros(n_epochs)


def cellpose_modules() -> dict[str, ModuleType]:
    torch = ModuleType("torch")
    vars(torch).update(
        __version__="0.0+stand-in",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index=0: "Stand-in GPU",
            empty_cache=lambda: None,
            reset_peak_memory_stats=lambda device=None: None,
            max_memory_reserved=lambda device=None: 0,
        ),
        version=SimpleNamespace(cuda="13.0"),
        device=lambda device: SimpleNamespace(type=str(getattr(device, "type", device))),
        sparse=SimpleNamespace(check_sparse_tensor_invariants=SimpleNamespace(disable=lambda: None)),
    )
    models = ModuleType("cellpose.models")
    vars(models).update(CellposeModel=_StandInCellposeModel)
    train = ModuleType("cellpose.train")
    vars(train).update(train_seg=_train_seg)
    transforms = ModuleType("cellpose.transforms")
    vars(transforms).update(make_tiles=lambda *args, **kwargs: None)
    cellpose = ModuleType("cellpose")
    vars(cellpose).update(models=models, train=train, transforms=transforms)
    return {
        "torch": torch,
        "cellpose": cellpose,
        "cellpose.models": models,
        "cellpose.train": train,
        "cellpose.transforms": transforms,
    }


def with_info(exp: Experiment, text: str) -> None:
    Path(exp.info_file).write_text(text, encoding="utf-8")


def well_outputs(exp: Experiment, well: str) -> dict[Path, bytes]:
    files = [io.round_table_path(exp, well), io.tmat_path(exp, well, ".npy"), io.cell_data_path(exp, well)]
    for folder in (exp.output_images, exp.segmented, exp.masks):
        if (Path(folder) / well).exists():
            files.extend(path for path in (Path(folder) / well).rglob("*") if path.is_file())
    return {path: path.read_bytes() for path in files if path.exists()}


VARIED_FEATURES = ["nuc_area", "R0_pRB_nuc_mean", "R0_RB_nuc_mean", "R1_cMyc_nuc_mean", "R2_EdU_nuc_mean"]


def varied_cell_data(exp: Experiment, wells: Sequence[str] | None = None) -> None:
    for index, well in enumerate(exp.well_list):
        if wells is not None and well not in wells:
            continue
        converted(exp, well)
        segmented(exp, well)
        aligned(exp, well)
        measured(exp, well)
        cells = [4 * index + cell for cell in range(4)]
        measured_values(exp, well, "nuc_area", [30.0 + k + 0.5 * (k % 3) for k in cells])
        measured_values(exp, well, "R0_pRB_nuc_mean", [1000.0 + 50 * k + 7 * (k % 3) for k in cells])
        measured_values(exp, well, "R0_RB_nuc_mean", [800.0 + 20 * k + 11 * (k % 4) for k in cells])
        measured_values(exp, well, "R1_cMyc_nuc_mean", [2000.0 + 30 * (5 * k % 12) for k in cells])
        measured_values(exp, well, "R2_EdU_nuc_mean", [3000.0 + 40 * (7 * k % 12) for k in cells])


def with_derived(exp: Experiment, derived: Sequence[Mapping[str, str]]) -> Experiment:
    save_config({**exp.config, "derived_features": [dict(spec) for spec in derived]}, exp.config_path)
    return load_config(exp.config_path, announce=False)
