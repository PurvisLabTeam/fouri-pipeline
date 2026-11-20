"""Well names, per-well metadata tables, feature names, and sort keys.

Well names are normalized to the zero-padded form (e.g., `B02`). A metadata table has one row per
well, with a `well` column and one column per metadata field (e.g., `sample_ID`, `treatment`,
`dose`, `treatment_time`).
"""

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

STANDARD_COLUMNS = ["sample_ID", "treatment", "dose", "treatment_time"]

# accepted spellings (compared lower-case, spaces/dashes as underscores)
_COLUMN_ALIASES = {
    "well": "well",
    "sample_id": "sample_ID",
    "sampleid": "sample_ID",
    "treatment": "treatment",
    "dose": "dose",
    "treatment_time": "treatment_time",
    "time": "treatment_time",
}

_WELL_RE = re.compile(r"^([A-Pa-p])0*([1-9]|1[0-9]|2[0-4])$")
_ND2_WELL_RE = re.compile(r"Well([A-P]\d{1,2})_Channel")
_DIGITS_RE = re.compile(r"(\d+)")


def normalize_well(well: str) -> str:
    """Normalize a well name (e.g., `b2` -> `B02`), raising `ValueError` if not a valid name."""
    m = _WELL_RE.match(str(well).strip())
    if not m:
        raise ValueError(f"Not a well name: {well!r}")
    return f"{m.group(1).upper()}{int(m.group(2)):02d}"


def plate_sort_key(well: str) -> tuple[str, int]:
    """Return a key that sorts wells by plate row, then by column (e.g., `B9` before `B10`)."""
    w = normalize_well(well)
    return (w[0], int(w[1:]))


def to_sample_id(value: Any) -> Any:
    """Return `value` as an int if it is an integer in text or float form, else unchanged."""
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def feature_prefix(round_name: str, signal: str) -> str:
    """Return the feature column prefix for a signal in a round (e.g., `R0`, `pRB` -> `R0_pRB`)."""
    return f"{str(round_name).zfill(2)}_{signal}"


def read_metadata_table(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read a per-well metadata table from a CSV or TSV file, or from a DataFrame.

    Column names are matched to the standard names regardless of case, spaces, or dashes (e.g.,
    `Sample ID` -> `sample_ID`, `time` -> `treatment_time`). Rows with a blank `well` are dropped.
    Every value other than `well` becomes stripped text. A blank or missing value becomes empty
    text.

    Parameters
    ----------
    source : str | Path | pd.DataFrame
        The path to a comma-separated `.csv` file or a tab-separated `.tsv` or `.txt` file, or a
        DataFrame, which is copied.

    Returns
    -------
    pd.DataFrame
        One row per well, sorted by plate row, then by column, with `well` normalized (e.g., `B2` ->
        `B02`).

    Raises
    ------
    ValueError
        If the table has no `well` column, or a `well` value is not a valid well name.

    Notes
    -----
    Wells listed multiple times are kept. `check_metadata` reports them.

    A file in the standard layout:

        well,treatment,dose,treatment_time,sample_ID
        B02,control,0,96h,
        B03,abema_fulv,100nM,96h,
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = os.path.expanduser(str(source))
        sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)

    renamed = {}
    for col in df.columns:
        key = re.sub(r"[\s\-]+", "_", str(col).strip().lower())
        renamed[col] = _COLUMN_ALIASES.get(key, str(col).strip())
    df = df.rename(columns=renamed)
    if "well" not in df.columns:
        raise ValueError(f"Metadata table needs a 'well' column (found: {list(df.columns)})")

    df = df.loc[df["well"].map(_clean) != ""].copy()
    bad = [w for w in df["well"] if not _WELL_RE.match(str(w).strip())]
    if bad:
        raise ValueError(f"Unrecognized well names: {bad}")
    df["well"] = df["well"].map(normalize_well)
    for col in df.columns:
        if col != "well":
            df[col] = df[col].map(_clean)
    df = df.sort_values("well", key=lambda s: s.map(plate_sort_key), kind="stable")
    return df.reset_index(drop=True)


def _listed(values: Iterable[Any]) -> str:
    return ", ".join(str(value) for value in values)


def _condition_text(values: Any) -> str:
    return " · ".join(value for value in values if value) or "(blank)"


def _id_order(value: Any) -> tuple[int, int, str]:
    text = str(value)
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def check_metadata(
    table: pd.DataFrame, condition_columns: Sequence[str], wells_with_data: Iterable[str] | None = None
) -> list[tuple[str, str]]:
    """Check a metadata table and return the errors and warnings found.

    Errors are one of:

    - A well listed multiple times.
    - A well without ND2 files, when `wells_with_data` is given.

    Warnings are one of:

    - A well with ND2 files that is not in the table, when `wells_with_data` is given.
    - A sample ID shared by wells of different conditions.
    - A condition whose wells have different sample IDs.

    A condition is a well's values in the `condition_columns` present in the table. The sample ID is
    checked only when the table has a `sample_ID` column and at least one condition column.

    Parameters
    ----------
    table : pd.DataFrame
        The metadata table, as returned by `read_metadata_table`.
    condition_columns : Sequence[str]
        The columns whose values define a condition.
    wells_with_data : Iterable[str] | None, default=None
        The wells with ND2 files, as returned by `wells_in_data`. None skips the check for ND2
        files.

    Returns
    -------
    list[tuple[str, str]]
        One `(level, message)` pair per problem, `level` being `"error"` or `"warning"`. Empty if
        no problem is found.
    """
    issues = []
    dup = sorted(table["well"][table["well"].duplicated()].unique(), key=plate_sort_key)
    if dup:
        issues.append(("error", f"Wells listed more than once: {_listed(dup)}."))
    if wells_with_data is not None:
        missing = sorted(set(table["well"]) - set(wells_with_data), key=plate_sort_key)
        if missing:
            issues.append(("error", f"No ND2 files for wells: {_listed(missing)}."))
        unused = sorted(set(wells_with_data) - set(table["well"]), key=plate_sort_key)
        if unused:
            issues.append(("warning", f"Wells with data that are not included: {_listed(unused)}."))
    if "sample_ID" in table.columns:
        cols = [c for c in condition_columns if c in table.columns]
        if cols:
            conds = table[cols].map(_clean).apply(tuple, axis=1)
            sid = table["sample_ID"].astype(str)
            for s, c in pd.DataFrame({"s": sid, "c": conds}).groupby("s")["c"]:
                if c.nunique() > 1:
                    covered = _listed(_condition_text(cond) for cond in sorted(set(c)))
                    issues.append(("warning", f"Sample ID {s} covers different conditions: {covered}."))
            for condition, sample_ids in pd.DataFrame({"s": sid, "c": conds}).groupby("c")["s"]:
                if sample_ids.nunique() > 1:
                    ids = _listed(sorted(set(sample_ids), key=_id_order))
                    issues.append(("warning", f"{_condition_text(condition)} has several sample IDs: {ids}."))
    return issues


def table_to_config(table: pd.DataFrame) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Convert a metadata table to the `well_list` and `well_metadata` of `experiment.json`.

    `sample_ID` is kept for every well, read as an int when an integer (`to_sample_id`). Any other
    column is kept only when it is not blank, as stripped text.

    Parameters
    ----------
    table : pd.DataFrame
        The metadata table, with a `well` column.

    Returns
    -------
    well_list : list[str]
        The wells, in table order.
    well_metadata : dict[str, dict[str, Any]]
        The metadata of each well, keyed by well.

    Notes
    -----
    A well listed multiple times appears multiple times in `well_list`. The last row's metadata is
    kept.
    """
    well_list = table["well"].tolist()
    meta = {}
    for _, row in table.iterrows():
        m = {}
        for col in table.columns:
            if col == "well":
                continue
            value = row[col]
            if col == "sample_ID":
                m[col] = to_sample_id(value)
            elif _clean(value) != "":
                m[col] = _clean(value)
        meta[row["well"]] = m
    return well_list, meta


def config_to_table(well_list: Sequence[str], well_metadata: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Convert `well_list` and `well_metadata` to a metadata table, reversing `table_to_config`."""
    rows = [{"well": w, **(well_metadata.get(w) or {})} for w in well_list]
    df = pd.DataFrame(rows, columns=None)
    if df.empty:
        df = pd.DataFrame(columns=["well"])
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    extras = [c for c in df.columns if c not in ["well", *STANDARD_COLUMNS]]
    df = df[["well", *STANDARD_COLUMNS, *extras]]
    for col in df.columns:
        if col not in ("well", "sample_ID"):
            df[col] = df[col].map(_clean)
    return df


def wells_in_data(data_dir: str | Path) -> list[str]:
    """List the wells with ND2 files in any round directory of the data directory.

    A file is counted when its name contains `Well<well>_Channel` (e.g., `WellB02_Channel`) and its
    suffix is `.nd2`, regardless of case. Only the direct subdirectories of `data_dir` are searched.

    Parameters
    ----------
    data_dir : str | Path
        The path to the experiment's `data` directory.

    Returns
    -------
    list[str]
        The wells, normalized and sorted by plate row, then by column. Empty if `data_dir` does not
        exist.
    """
    wells = set()
    if not os.path.isdir(data_dir):
        return []
    for sub in os.listdir(data_dir):
        sub_path = os.path.join(data_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        for name in os.listdir(sub_path):
            m = _ND2_WELL_RE.search(name)
            if m and name.lower().endswith(".nd2"):
                wells.add(normalize_well(m.group(1)))
    return sorted(wells, key=plate_sort_key)


def channel_names(info_file: str | Path) -> list[str]:
    """List the feature prefix of each signal in `info_exp.csv`, in file order (e.g., `R0_DNA`)."""
    info = pd.read_csv(info_file)
    names = []
    for _, row in info.iterrows():
        for col in info.columns[1:]:
            signal = row[col]
            if signal == signal and str(signal).strip():  # skip empty cells
                names.append(feature_prefix(row.iloc[0], signal))
    return names


def default_derived_features(info_file: str | Path) -> list[dict[str, str]]:
    """Return the starting derived features: `pRB_RB_ratio` when a round has both pRB and RB."""
    names = channel_names(info_file)
    features = []
    for n in names:
        rnd, signal = n.split("_", 1)
        if signal == "pRB" and f"{rnd}_RB" in names:
            features.append({"name": "pRB_RB_ratio", "formula": f"{rnd}_pRB_nuc_mean / {rnd}_RB_nuc_mean"})
            break
    return features


def natural_key(value: object) -> tuple[Any, ...]:
    """Return a natural sort key: numbers by value, text regardless of case (e.g., `B9` < `B10`)."""
    parts = _DIGITS_RE.split(str(value).casefold())
    # split puts the digit runs at the odd places
    return tuple((0, int(part), "") if place % 2 else (1, 0, part) for place, part in enumerate(parts) if part)
