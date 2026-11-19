"""Run one pipeline step as its own program: how the app starts a step.

    python -m fouri.run measure --config Z:/exp/experiment.json --wells B02 C02
        --settings '{"overwrite": true}' --progress C:/runs/measure.jsonl --run 20261109T173005

``--settings`` is a JSON object of the step function's keyword arguments; ``--wells`` (per-well
steps only) defaults to every well. With ``--progress`` every message and bar goes to that file as
JSON lines (see ``fouri.report``), ending with a ``finished`` line; creating
``report.stop_path(progress file)`` stops the step. Each step adds a "started" and an "ended" line
to the experiment's run record (``io.run_record_path``) with the wells, settings, fouri version
and outcome, tagged with ``--run`` so the app can group the steps of one run.

Exit codes: 0 done, 1 failed, 3 stopped (2 is argparse's for bad arguments).
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import importlib.metadata
import inspect
import json
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from omegaconf import DictConfig, OmegaConf

from fouri import report
from fouri.adata import check_subsample
from fouri.config import Experiment, load_config
from fouri.io import adata_path, adata_sub_path, run_record_path
from fouri.parallel import error_in_words
from fouri.preprocess import changed_wells
from fouri.report import say
from fouri.status import WELL_STEPS, Step

STEP_NAMES = {
    Step.CONVERT: "Convert images",
    Step.SEGMENT: "Segment nuclei",
    Step.ALIGN: "Align rounds",
    Step.ARTIFACTS: "Find artifacts",
    Step.MEASURE: "Measure cells",
    Step.TABLE: "Build cell table",
    Step.BATCH: "Batch effects",
    Step.EMBEDDING: "Embedding",
    Step.EXPORT: "Export",
    Step.FINETUNE: "Fine-tune Cellpose",
}
EXIT_CODES = {"done": 0, "failed": 1, "stopped": 3}


def reconvert(config: str | Path, wells: Sequence[str] | None, *, dry_run: bool, force: bool) -> int:
    """@TODO."""
    exp = load_config(Path(config), announce=False)
    changed = changed_wells(exp, wells)
    if changed == []:
        say("No well's rounds have changed in info_exp.csv.")
        return 0
    if force:
        outcome = run_step(
            Step.CONVERT, config, wells=[well.well for well in changed], settings={"reconvert_changed": True}
        )
        return EXIT_CODES[outcome]
    for well in changed:
        say(
            f"{well.well} ({', '.join(well.rounds)} changed): "
            f"{', '.join(f'{o.parent.name}/{o.name}' for o in well.outputs)}"
        )
    if dry_run:
        return 0

    say("Nothing deleted. Run again with --force to delete these and convert the well(s) again.")
    return 2


def build_cell_table(exp: Experiment, filters: Sequence[str] = (), labels: Sequence[str] = ()) -> None:
    """Combine the wells' cell data, add derived features, save fullest_df.csv and the AnnData."""
    given_counts = Counter(labels)
    duplicated_labels = sorted(label for label, count in given_counts.items() if count > 1)
    if duplicated_labels:
        raise ValueError(f"The following given label(s), appear multiple times: {', '.join(duplicated_labels)}.")

    from fouri.adata import build_adata
    from fouri.artifacts import remove_covered
    from fouri.features import add_derived_features, build_fullest_df, measured_columns
    from fouri.filters import apply_filters, named_filters
    from fouri.labels import apply_labels, named_labels

    measured = set(measured_columns(exp))
    clashed_names = [
        feature["name"]
        for feature in exp.derived_features
        if feature["name"].endswith("_covered") or feature["name"] in measured
    ]
    if clashed_names:
        raise ValueError(
            f"Invalid derived feature names found: {', '.join(clashed_names)}. "
            "Please use derived feature names that do not end in '_covered' and that don't clash with existing ones."
        )

    table = add_derived_features(build_fullest_df(exp), exp.derived_features)
    initial_count = len(table)
    table, dropped_count = remove_covered(table)
    filter_rules = named_filters(exp, filters)
    report.say(f"Artifact masks removed {dropped_count:,} of {initial_count:,} cells.")

    included, filter_counts = apply_filters(table, filter_rules)
    table["included"] = included
    total_cells = len(table)
    if filters:
        for rule, count in filter_counts.items():
            report.say(f"{rule}: left out {count:,} cells of {total_cells:,}.")

        total_dropped = (~included).sum()
        report.say(
            f"Kept {(total_cells - total_dropped):,} of {total_cells:,} cells. "
            f"Filters dropped {total_dropped:,} cells."
        )
    if labels:
        label_rules = named_labels(exp, labels)
        applied_labels = apply_labels(table, label_rules)
        for label, col in applied_labels.items():
            table[label] = col
            label_counts = col.value_counts()
            no_group = col.isna().sum()
            message = f"'{label}': {', '.join(f'{group}: {count:,}' for group, count in label_counts.items())}."
            message = message if no_group == 0 else message + f" {no_group:,} cells are not in a class."
            report.say(message)

    build_adata(exp, table, labels)


def make_embeddings(
    exp: Experiment,
    runs: Sequence[Mapping[str, Any]],
    picked: Sequence[str],
    *,
    samples: Sequence[int] | None = None,
    ranking_k: int = 100,
    seed: int = 42,
    values: str = "original",
    cells_per_sample: int = 2000,
    subsample_seed: int = 42,
    overwrite: bool = False,
    n_workers: int | None = None,
) -> list[str]:
    """Subsample each sample (if not done, or with overwrite), then make the embeddings in ``runs``.

    Each run is ``{"method": "PHATE", "dims": 3, "top_n": 10, "settings": {"knn": 80}}``.
    Subsampling again replaces the subsample file and the embeddings in it.
    """
    import anndata as ad

    from fouri.embedding import Method, Run, embed_grid, subsample

    if overwrite or not adata_sub_path(exp).exists():
        subsample(exp, cells_per_sample=cells_per_sample, seed=subsample_seed)
    else:
        table_uns = ad.read_h5ad(adata_path(exp)).uns
        subsample_uns = ad.read_h5ad(adata_sub_path(exp)).uns
        check_subsample(table_uns, subsample_uns)
        report.say("Using the existing subsample. To subsample again, use overwrite=True")
    grid = [
        Run(Method(run["method"]), int(run["dims"]), int(run["top_n"]), dict(run.get("settings", {}))) for run in runs
    ]
    return embed_grid(
        exp, grid, picked, samples=samples, ranking_k=ranking_k, seed=seed, n_workers=n_workers, values=values
    )


def step_function(step: Step) -> Callable[..., Any]:
    """Return the function that does a step; its first argument is the Experiment."""
    if step is Step.CONVERT:
        from fouri.preprocess import preprocess

        return preprocess
    if step is Step.SEGMENT:
        from fouri.segment import segment_wells

        return segment_wells
    if step is Step.ALIGN:
        from fouri.align import find_transforms

        return find_transforms
    if step is Step.ARTIFACTS:
        from fouri.artifacts import find_artifacts

        return find_artifacts
    if step is Step.MEASURE:
        from fouri.measure import measure_wells

        return measure_wells
    if step is Step.BATCH:
        from fouri.batch import correct_batches

        return correct_batches
    if step is Step.EXPORT:
        from fouri.export import export_cells

        return export_cells
    if step is Step.FINETUNE:
        from fouri.finetune import finetune_cellpose

        return finetune_cellpose
    return build_cell_table if step is Step.TABLE else make_embeddings


def check_settings(step: Step, settings: Mapping[str, Any]) -> None:
    """Refuse settings the step doesn't take, and settings it needs that are missing."""
    parameters = list(inspect.signature(step_function(step)).parameters.values())[1:]
    names = [p.name for p in parameters]
    unknown = sorted(set(settings) - set(names))
    if unknown:
        raise ValueError(
            f"{STEP_NAMES[step]} has no setting {', '.join(unknown)}. Its settings are: {', '.join(names) or 'none'}."
        )
    missing = [p.name for p in parameters if p.default is inspect.Parameter.empty and p.name not in settings]
    if missing:
        raise ValueError(f"{STEP_NAMES[step]} needs {', '.join(missing)}.")
    for param in parameters:
        if param.name not in settings:
            continue
        if param.default is inspect.Parameter.empty:
            continue

        setting = settings.get(param.name)

        should_raise = False
        kind_message = ""
        match param.default:
            case bool():
                should_raise = not isinstance(setting, bool)
                kind_message = "a boolean (true or false)"
            case int():
                should_raise = not isinstance(setting, int) or isinstance(setting, bool)
                kind_message = "an integer (a whole number)"
            case float():
                should_raise = not isinstance(setting, float | int) or isinstance(setting, bool)
                kind_message = "a number"
            case str():
                should_raise = not isinstance(setting, str)
                kind_message = "text"
            case tuple():
                should_raise = not isinstance(setting, tuple | list)
                kind_message = "a list"
            case _:
                continue
        if should_raise:
            message = (
                f"{STEP_NAMES[step]} takes the {param.name} setting as {kind_message}. Got {json.dumps(setting)}."
            )
            raise ValueError(message)


def command(
    step: Step | str,
    config: str | Path,
    *,
    wells: Sequence[str] | None = None,
    settings: Mapping[str, Any] | None = None,
    settings_file: str | Path | None = None,
    progress: str | Path | None = None,
    run: str | None = None,
) -> list[str]:
    """Return the command line that runs a step as its own program, for subprocess."""
    argv = [sys.executable, "-m", "fouri.run", Step(step).value, "--config", str(config)]
    if wells is not None:
        argv += ["--wells", *wells]
    if settings_file:
        settings_file = str(Path(settings_file))
        argv += ["--settings-file", settings_file]
    elif settings:
        argv += ["--settings", json.dumps(settings)]
    if progress is not None:
        argv += ["--progress", str(progress)]
    if run is not None:
        argv += ["--run", run]
    return argv


def fouri_version() -> dict[str, str | None]:
    """Return the package version, and the git commit (+changes if edited) when run from a clone."""
    folder = Path(__file__).parent
    commit = None
    try:
        sha = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if sha.returncode == 0:
            dirty = subprocess.run(
                ["git", "-C", str(folder), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            commit = sha.stdout.strip() + ("+changes" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return {"version": importlib.metadata.version("4i-pipeline"), "commit": commit}


def _record(exp: Experiment, entry: dict[str, Any]) -> None:
    with open(run_record_path(exp), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def run_step(
    step: Step | str,
    config: str | Path,
    *,
    wells: Sequence[str] | None = None,
    settings: Mapping[str, Any] | None = None,
    run: str | None = None,
) -> Literal["done", "failed", "stopped"]:
    """Run one step in this process, report how it ended with ``report.finished``, and record it."""
    step = Step(step)
    settings = dict(settings or {})
    exp: Experiment | None = None
    entry: dict[str, Any] = {"run": run, "step": step.value, "settings": dict(settings)}  # wells go apart
    outcome: Literal["done", "failed", "stopped"] = "done"
    failed: list[str] = []
    error = ""
    started = time.monotonic()
    try:
        exp = load_config(config, announce=False)
        if step == Step.EMBEDDING and "runs" not in settings:
            from fouri.embedding_setup import embedding_settings

            embed_settings = embedding_settings(exp)
            if embed_settings is None:
                raise ValueError(
                    "No embedding setup is saved. Save one on the Embedding page's setup tab or supply "
                    "'runs' and 'picked' in the run settings."
                )
            if embed_settings["runs"] == []:
                raise ValueError(
                    "No embedding is selected in the saved setup. Select embedding parameters in the Embedding page's "
                    "setup tab, or supply 'runs' and 'picked' in the run settings."
                )

            settings = embed_settings | settings
            entry["settings"] = settings

        if step in WELL_STEPS:
            settings["wells"] = exp.pick_wells(wells if wells is not None else settings.get("wells"))
        elif wells is not None:
            raise ValueError(f"{STEP_NAMES[step]} always uses every well; it takes no wells.")
        entry["wells"] = settings.get("wells")
        check_settings(step, settings)
        entry.update(
            event="started", time=dt.datetime.now().isoformat(timespec="seconds"), pc=platform.node(),
            user=getpass.getuser(), fouri=fouri_version(),
        )  # fmt: skip
        _record(exp, entry)
        step_function(step)(exp, **settings)
    except report.Stopped:
        outcome = "stopped"
    except Exception as exception:  # any failure is reported to the app, then the program ends
        outcome = "failed"
        failed = [str(item) for item in getattr(exception, "failed", ())]  # StepFailed lists the failed items
        error = error_in_words(exception)
        traceback.print_exc()
    report.finished(outcome, failed=failed, error=error)
    if exp is not None and "event" in entry:
        ended = dt.datetime.now().isoformat(timespec="seconds")
        seconds = round(time.monotonic() - started, 1)
        _record(exp, {**entry, "event": "ended", "time": ended, "seconds": seconds, "outcome": outcome,
                      "failed": failed, "error": error})  # fmt: skip
    return outcome


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "reconvert":
        reconvert_parser = argparse.ArgumentParser(
            prog="python -m fouri.run reconvert",
            description="List or reconvert the wells whose rounds changed in info_exp.csv.",
        )
        reconvert_parser.add_argument("--config", required=True, help="the experiment's experiment.json")
        reconvert_parser.add_argument("--wells", nargs="*", help="wells to check; every well if left out")
        group = reconvert_parser.add_mutually_exclusive_group()
        group.add_argument(
            "--dry-run",
            action="store_true",
            help="list the changed wells that would be deleted without deleting anything.",
        )
        group.add_argument(
            "--force", action="store_true", help="delete the changed wells' outputs and reconvert them."
        )
        reconvert_args = reconvert_parser.parse_args(argv[1:])
        return reconvert(
            reconvert_args.config, reconvert_args.wells, dry_run=reconvert_args.dry_run, force=reconvert_args.force
        )

    parser = argparse.ArgumentParser(prog="python -m fouri.run", description="Run one 4i pipeline step.")
    parser.add_argument("step", choices=[s.value for s in Step])
    parser.add_argument(
        "overrides",
        nargs="*",
        help="settings as key=value, which replace those from --settings or --settings-file",
    )
    parser.add_argument("--config", required=True, help="the experiment's experiment.json")
    parser.add_argument("--wells", nargs="*", help="wells to work on (per-well steps); every well if left out")
    parser.add_argument("--settings", default=None, help="the step's settings as a JSON object")
    parser.add_argument("--settings-file", default=None, help="the step's settings in a YAML or JSON file")
    parser.add_argument("--progress", help="write messages and bars to this file as JSON lines")
    parser.add_argument("--run", help="the run this step belongs to, written to the run record")
    args = parser.parse_intermixed_args(argv)

    if args.wells is not None:
        args.overrides.extend(well for well in args.wells if "=" in well)
        args.wells = [well for well in args.wells if "=" not in well]

    if args.overrides:
        invalid = [over for over in args.overrides if "=" not in over or over.startswith("=")]
        if invalid:
            raise parser.error(
                "\n".join(
                    f"Unrecognized override argument: {arg}. Write overrides as key=value, e.g., ring_width=8."
                    for arg in invalid
                )
            )

    settings: Any = {}
    if args.settings is not None and args.settings_file is not None:
        raise parser.error("Use either --settings or --settings-file, not both.")

    if args.settings is not None:
        try:
            settings = json.loads(args.settings)
        except json.JSONDecodeError as e:
            raise parser.error("--settings must be a JSON object, e.g. '{\"overwrite\": true}'") from e

    if args.settings_file is not None:
        settings_file = Path(args.settings_file)
        if not settings_file.exists():
            raise parser.error(f"No settings file found at {settings_file}")
        try:
            settings_loaded = OmegaConf.load(settings_file)
        except (yaml.YAMLError, OSError) as e:
            raise parser.error(f"Cannot read settings file at {settings_file} as YAML or JSON.") from e
        if not isinstance(settings_loaded, DictConfig):
            raise parser.error("--settings-file must hold settings by name, e.g. overwrite: true.")

        settings = OmegaConf.to_container(settings_loaded)

    if not isinstance(settings, dict):
        parser.error("--settings must be a JSON object, e.g. '{\"overwrite\": true}'")

    overrides = []
    for over in args.overrides:
        try:
            override = OmegaConf.from_dotlist([over])
        except yaml.YAMLError as e:
            raise parser.error(
                f"Unrecognized override argument: {over}. Write overrides as key=value, e.g., ring_width=8."
            ) from e
        overrides.append(override)

    settings = OmegaConf.to_container(OmegaConf.merge(settings, *overrides))
    report.report_to(args.progress)
    outcome = run_step(args.step, args.config, wells=args.wells, settings=settings, run=args.run)
    return EXIT_CODES[outcome]


if __name__ == "__main__":
    sys.exit(main())
