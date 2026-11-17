"""@TODO."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd

from fouri import io
from fouri.config import Experiment
from fouri.features import metadata_obs_columns
from fouri.io import adata_sub_path, fullest_df_path
from fouri.metadata import natural_key, normalize_well
from fouri.report import say

PLATE_ROW: Final[str] = "plate row"
PLATE_COLUMN: Final[str] = "plate column"
NO_CONDITION: Final[str] = "all cells"
CONDITION_JOIN: Final[str] = " · "
SHARE_OVER: Final[float] = 0.10
TOP_FEATURES: Final[int] = 15
CLASSIFIER_CELLS: Final[int] = 5_000
CLASSIFIER_TREES: Final[int] = 100
CLASSIFIER_FOLDS: Final[int] = 3
NEAR_CHANCE: Final[float] = 0.05

LOG_MEASURES: Final[tuple[str, ...]] = ("nuc_mean", "ring_mean", "total_nuc_protein")


@dataclass(frozen=True)
class BatchMetrics:
    shares: pd.Series
    batch_median: float
    features_over: int
    classifier: float | None
    chance: float
    classifier_by_condition: pd.Series
    chance_by_condition: pd.Series
    condition_median: float | None
    top: list[str]


def log_columns(var: pd.DataFrame) -> list[str]:
    """@TODO."""
    mask = (var["kind"] == "channel") & (var["measure"].isin(LOG_MEASURES))
    return var[mask].index.to_list()


def batch_labels(obs: pd.DataFrame, batch: str) -> pd.Series:
    """@TODO."""
    wells = obs["well"].astype("string")
    present = list(dict.fromkeys(wells))
    if batch == PLATE_ROW:
        return wells.map({well: normalize_well(well)[0] for well in present}).rename(batch)
    elif batch == PLATE_COLUMN:
        return wells.map({well: str(int(normalize_well(well)[1:])) for well in present}).rename(batch)
    else:
        try:
            return obs[batch].astype("string").rename(batch)
        except KeyError as e:
            raise ValueError(f"The cells table has no column named '{batch}'.") from e


def condition_labels(obs: pd.DataFrame, keep: Sequence[str]) -> pd.Series:
    """@TODO."""
    if len(keep) == 0:
        return pd.Series(NO_CONDITION, index=obs.index)
    missing = [col for col in keep if col not in obs.columns]
    if missing:
        raise ValueError(
            f"The cells table does not have the following condition column(s): {', '.join(f'{m}' for m in missing)}"
        )
    selected = obs[keep].astype("string")
    return selected.iloc[:, 0].str.cat(selected.iloc[:, 1:], sep=CONDITION_JOIN)


def conditions_in_batches(conditions: pd.Series, batches: pd.Series) -> pd.DataFrame:
    """@TODO."""
    by_conditions = batches.groupby(conditions).unique()

    rows: dict[str, list[Any]] = {"condition": [], "batches": [], "usable": []}
    for condition in sorted(by_conditions.index, key=natural_key):
        condition_batches = sorted(by_conditions[condition], key=natural_key)
        usable = len(condition_batches) >= 2
        rows["condition"].append(condition)
        rows["batches"].append(condition_batches)
        rows["usable"].append(usable)

    return pd.DataFrame(rows)


def used_conditions(conditions: pd.Series, batches: pd.Series) -> tuple[pd.DataFrame, npt.NDArray[np.bool]]:
    in_batches = conditions_in_batches(conditions, batches)
    usable = in_batches[in_batches["usable"]]
    usable = usable.reset_index(drop=True)
    mask = conditions.isin(usable["condition"]).to_numpy()
    usable = usable.drop(columns="usable")
    return usable, mask


def condition_shares(conditions: pd.Series, batches: pd.Series) -> pd.DataFrame:
    """@TODO."""
    _, mask = used_conditions(conditions, batches)
    cond_masked = conditions[mask]
    batches_masked = batches[mask]
    result = pd.crosstab(batches_masked, cond_masked, normalize="index")
    result = result.where(result > 0)
    result = result.reindex(
        labels=sorted(result.index, key=natural_key), columns=sorted(result.columns, key=natural_key)
    )
    result = result.rename_axis(index="batch", columns=None)
    return result


def uneven_conditions(shares: pd.DataFrame) -> list[tuple[str, str, str]]:
    """@TODO."""
    result = []
    for condition, column_shares in shares.items():
        condition = str(condition)
        lowest_batch = str(column_shares.idxmin())
        highest_batch = str(column_shares.idxmax())
        lowest_share = column_shares.min()
        highest_share = column_shares.max()

        if lowest_share < highest_share / 2:
            result.append((condition, lowest_batch, highest_batch))
    return result


def batch_groups(conditions: pd.Series, batches: pd.Series) -> list[list[str]]:
    """@TODO."""
    used, _ = used_conditions(conditions, batches)
    groups: list[set[str]] = []
    for row in used["batches"]:
        current = set(row)
        overlap = [g for g in groups if not g.isdisjoint(current)]
        not_overlap = [g for g in groups if g.isdisjoint(current)]
        groups = [*not_overlap, current.union(*overlap)]

    ordered = [sorted(g, key=natural_key) for g in groups]
    ordered = sorted(ordered, key=lambda g: natural_key(g[0]))
    return ordered


def batch_metrics(
    values: npt.NDArray[np.float64], features: Sequence[str], conditions: pd.Series, batches: pd.Series
) -> BatchMetrics | None:
    """@TODO."""
    used, is_used = used_conditions(conditions, batches)
    if used.empty:
        return None

    cell_values = values[is_used]
    cell_conditions = conditions[is_used].to_numpy()
    cell_batches = batches[is_used].to_numpy()

    score_sum = 0.0
    weight_total = 0
    chance_sum = 0.0
    all_chance_sum = 0.0
    shares_arr = np.zeros(len(features))
    classifier_by_condition: dict[str, float] = {}
    chance_by_condition: dict[str, float] = {}
    for condition in used["condition"]:
        in_condition = cell_conditions == condition
        condition_values = cell_values[in_condition]
        condition_batches = cell_batches[in_condition]
        cell_count = int(in_condition.sum())
        shares_arr += cell_count * _share_explained(condition_values, condition_batches)

        condition_chance = 1 / len(np.unique(condition_batches))
        all_chance_sum += cell_count * condition_chance
        score = _classifier_accuracy(condition_values, condition_batches)
        if score is not None:
            classifier_by_condition[str(condition)] = score
            chance_by_condition[str(condition)] = condition_chance
            chance_sum += cell_count * condition_chance
            score_sum += cell_count * score
            weight_total += cell_count

    shares_arr /= len(cell_values)
    if weight_total != 0:
        classifier: float | None = score_sum / weight_total
        chance = chance_sum / weight_total
    else:
        classifier = None
        chance = all_chance_sum / len(cell_values)
    batch_median = float(np.median(shares_arr))
    features_over = int((shares_arr > SHARE_OVER).sum())
    shares = pd.Series(shares_arr, index=list(features))
    top: list[str] = list(shares.nlargest(n=TOP_FEATURES).index)

    condition_median = float(np.median(_share_explained(cell_values, cell_conditions))) if len(used) > 1 else None

    return BatchMetrics(
        shares=shares,
        batch_median=batch_median,
        features_over=features_over,
        classifier=classifier,
        chance=chance,
        classifier_by_condition=pd.Series(classifier_by_condition, dtype=float),
        chance_by_condition=pd.Series(chance_by_condition, dtype=float),
        condition_median=condition_median,
        top=top,
    )


def _share_explained(values: npt.NDArray[np.float64], groups: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    # total sum of squares
    col_means = np.mean(values, axis=0)
    total_sos = np.sum((values - col_means) ** 2, axis=0)

    # between group sum of squares
    group_sos = np.zeros(values.shape[1])
    for label in np.unique(groups):
        rows = values[groups == label]
        count = len(rows)
        group_mean = np.mean(rows, axis=0)
        group_sos += count * (group_mean - col_means) ** 2

    result = np.zeros(total_sos.shape)
    np.divide(group_sos, total_sos, out=result, where=total_sos > 0)
    return result


def _classifier_accuracy(values: npt.NDArray[np.float64], batches: npt.NDArray[Any]) -> float | None:
    """@TODO."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if len(values) > CLASSIFIER_CELLS:
        rng = np.random.default_rng(0)
        sample = rng.choice(len(values), size=CLASSIFIER_CELLS, replace=False)
        sampled_values = values[sample]
        sampled_batches = batches[sample]
    else:
        sampled_values = values
        sampled_batches = batches

    _, counts = np.unique(sampled_batches, return_counts=True)
    _, orig_counts = np.unique(batches, return_counts=True)

    if np.min(counts) < CLASSIFIER_FOLDS or len(counts) < len(orig_counts):
        return None

    rf = RandomForestClassifier(CLASSIFIER_TREES, random_state=0, n_jobs=-1)
    kfold = StratifiedKFold(CLASSIFIER_FOLDS, shuffle=True, random_state=0)

    scores = cross_val_score(rf, sampled_values, sampled_batches, cv=kfold, scoring="balanced_accuracy")
    return float(np.mean(scores))


def why_not_combat(conditions: pd.Series, batches: pd.Series) -> str | None:
    """@TODO."""
    batch_name = str(batches.name)
    usable, _ = used_conditions(conditions, batches)
    if usable.empty:
        return (
            f"No two {batch_name}s share a condition, so differences between {batch_name}s cannot be separated "
            f"from biological differences. Correction by {batch_name} is not possible for this experiment."
        )
    return None


def _combat_group(
    values: npt.NDArray[np.float64], batches: pd.Series, conditions: pd.Series
) -> npt.NDArray[np.float64]:
    import scanpy

    obs = pd.get_dummies(conditions, drop_first=True, prefix="condition", dtype=float)
    covariates = list(obs.columns) or None

    obs["batch"] = batches.to_numpy()
    obs = obs.reset_index(drop=True)
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=values, obs=obs)
    corrected = scanpy.pp.combat(adata, key="batch", covariates=covariates, inplace=False)

    return np.asarray(corrected, dtype=np.float64)


def combat(values: npt.NDArray[np.float64], batches: pd.Series, conditions: pd.Series) -> npt.NDArray[np.float64]:
    """@TODO."""
    _, mask = used_conditions(conditions, batches)
    groups = batch_groups(conditions, batches)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for group in groups:
        rows = mask & batches.isin(group).to_numpy()
        corrected = _combat_group(values[rows], batches[rows], conditions[rows])
        result[rows] = corrected

    return result


def correct_batches(
    exp: Experiment, batch: str, keep: Sequence[str] | None = None, log_intensities: bool = True
) -> None:
    """@TODO."""
    adata_path = io.adata_path(exp)
    cells = ad.read_h5ad(adata_path)
    included = (
        cells.obs["included"].to_numpy(dtype=np.bool)
        if "included" in cells.obs.columns
        else np.full(len(cells.obs), fill_value=True, dtype=np.bool)
    )
    obs = cast(pd.DataFrame, cells.obs)
    included_obs = obs[included]
    keep = list(keep) if keep is not None else default_keep(exp)

    batches = batch_labels(included_obs, batch)
    conditions = condition_labels(included_obs, keep)

    used, mask = used_conditions(conditions, batches)
    left_counts = conditions[~mask].value_counts().to_dict()

    cells_values = np.asarray(cells.X, dtype=np.float64)
    cells_values = cells_values[included]

    before = batch_metrics(cells_values, list(cells.var_names), conditions, batches)

    why = why_not_combat(conditions, batches)
    if why is not None:
        raise ValueError(why)

    subsampled_path = adata_sub_path(exp)
    subsampled: ad.AnnData | None = None
    if subsampled_path.exists():
        subsampled = ad.read_h5ad(subsampled_path)
        sub_obs = cast(pd.DataFrame, subsampled.obs)
        full_keys = pd.MultiIndex.from_frame(obs[["well", "label"]])
        sub_keys = pd.MultiIndex.from_frame(sub_obs[["well", "label"]])
        matching = full_keys.get_indexer(sub_keys)
        if np.any(matching == -1):
            raise RuntimeError(
                "The subsampled table has cells that the main table no longer has. Rerun Embedding with Overwrite to "
                "make a new subsample."
            )
    var = cast(pd.DataFrame, cells.var)
    log_cols = log_columns(var)
    formulas = cast(pd.DataFrame, cells.uns["derived_features"])
    original = _original_table(exp, cells).loc[included].reset_index(drop=True)
    combat_input = original[cells.var_names].to_numpy(dtype=np.float64)
    if log_intensities:
        logged = cells.var_names.isin(log_cols)
        combat_input[:, logged] = np.log1p(combat_input[:, logged])

    if left_counts:
        say(
            f"The following conditions were left out and not corrected because they are in one {batch} "
            f"only: {', '.join(f'{k} ({v:,} cells)' for k, v in left_counts.items())}."
        )

    groups = batch_groups(conditions, batches)
    if len(groups) > 1:
        say(
            f"The following groups of {batch}s do not share any condition, so each was corrected on its own. "
            "The differences between these groups were left as they are: "
            f"{'; '.join(f'({", ".join(grp)})' for grp in groups)}."
        )
    shares = condition_shares(conditions, batches)
    uneven = uneven_conditions(shares)
    for condition, lowest, highest in uneven:
        low_share = cast(float, shares.loc[lowest, condition])
        high_share = cast(float, shares.loc[highest, condition])
        say(
            f"{condition} is {low_share:.0%} of {batch} {lowest}'s cells but "
            f"{high_share:.0%} of {batch} {highest}'s cells. With conditions spread this unevenly, "
            "ComBat may make the conditions appear further apart than they really are."
        )
    if before is not None and before.classifier is not None and before.classifier - before.chance <= NEAR_CHANCE:
        say(
            f"The classifier's prediction of {batch} is {before.classifier:.0%}. Pure chance is "
            f"{before.chance:.0%}. Batch correction is not recommended."
        )
    corrected_values, corrected_counts = _to_zscores(
        combat(combat_input, batches, conditions), original, var, formulas, log_intensities
    )

    if corrected_counts:
        say(
            "Kept ComBat's own correction where the value could not be recomputed:\n"
            f"{'\n'.join(f'{name}: {count:,} cells' for name, count in corrected_counts.items())}\n"
        )

    after = batch_metrics(corrected_values, list(cells.var_names), conditions, batches)
    adjusted = pd.DataFrame(corrected_values[mask] - cells_values[mask], columns=list(cells.var_names))
    adjustment = adjusted.groupby(batches[mask].to_numpy()).mean()
    adjustment = adjustment.reindex(sorted(adjustment.index.to_list(), key=natural_key))

    shuffled = _shuffled_batches(conditions, batches)
    corrected_ref, _ = _to_zscores(
        combat(combat_input, shuffled, conditions), original, var, formulas, log_intensities
    )
    reference = batch_metrics(corrected_ref, list(cells.var_names), conditions, shuffled)

    corrected = np.full(cells.shape, np.nan, dtype=np.float64)
    corrected[included] = corrected_values
    cells.layers[io.COMBAT_LAYER] = corrected

    log_transformed = log_cols if log_intensities else []
    cells.uns["batch_effects"] = {
        "batch": batch,
        "keep": keep,
        "log_transformed": log_transformed,
        "conditions": used["condition"].to_list(),
        "left_out": list(left_counts.keys()),
        "groups": {
            "batches": [batch for group in groups for batch in group],
            "values": np.repeat(np.arange(len(groups)), [len(g) for g in groups]),
        },
        "shares": {
            "batches": shares.index.to_list(),
            "conditions": shares.columns.to_list(),
            "values": shares.to_numpy(),
        },
        "adjustment": {
            "batches": adjustment.index.to_list(),
            "features": adjustment.columns.to_list(),
            "values": adjustment.to_numpy(),
        },
        "uneven": uneven,
        "before": _metrics_entry(before),
        "after": _metrics_entry(after),
        "reference": _metrics_entry(reference),
    }
    with io.written_whole(adata_path) as partial:
        cells.write_h5ad(partial)
    if subsampled is not None:
        subsampled.layers[io.COMBAT_LAYER] = corrected[matching]
        with io.written_whole(subsampled_path) as partial:
            subsampled.write_h5ad(partial)

    message = (
        f"Corrected {mask.sum():,} cells by {batch}{' on log intensities' if log_transformed else ''}"
        f"{f', keeping column(s): {", ".join(keep)}' if keep else ''}."
    )
    if before is not None and after is not None and reference is not None:
        message += f"\nmedian share explained by {batch}: {before.batch_median:.2%} -> {after.batch_median:.2%}."
        message += f"\nfeatures over {SHARE_OVER:.0%}: {before.features_over} -> {after.features_over}."

        if before.classifier is not None and after.classifier is not None and reference.classifier is not None:
            message += (
                f"\nbatches told apart in {before.classifier:.2%} -> {after.classifier:.2%} of cells "
                f"(ComBat on shuffled batches {reference.classifier:.2%}, chance {before.chance:.2%})."
            )
        if before.condition_median is not None and after.condition_median is not None:
            message += f"\nCondition: median {before.condition_median:.2%} -> {after.condition_median:.2%}."

        message += "\n"

    say(message)


def _original_table(exp: Experiment, cells: ad.AnnData) -> pd.DataFrame:
    fullest = pd.read_csv(fullest_df_path(exp))
    fullest_idx = pd.MultiIndex.from_frame(fullest[["well", "label"]])
    cells_idx = pd.MultiIndex.from_frame(cast(pd.DataFrame, cells.obs[["well", "label"]]))
    positions = fullest_idx.get_indexer(cells_idx)
    if not cells.var_names.isin(fullest.columns).all() or np.any(positions == -1):
        # There are keys in cells adata that are missing from the fullest df (rerun) or
        # There are features in cells adata that are missing from the fullest df (also a rerun)
        raise ValueError(
            "standard_adata.h5ad contains cells or features that fullest_df.csv does not have. Rerun Build cell table."
        )
    fullest = fullest.iloc[positions].reset_index(drop=True)
    return fullest


def _to_zscores(
    corrected: npt.NDArray[np.float64],
    original: pd.DataFrame,
    var: pd.DataFrame,
    formulas: pd.DataFrame,
    log_intensities: bool,
) -> tuple[npt.NDArray[np.float64], dict[str, int]]:
    mean = original[var.index].mean().to_numpy()
    sd = original[var.index].std().to_numpy()
    log_cols = log_columns(var)

    own_units = corrected.copy()
    if log_intensities:
        mask = var.index.isin(log_cols)
        own_units[:, mask] = np.expm1(own_units[:, mask])

    working = original.copy()
    working[var.index] = own_units
    with_corrected = ~np.isnan(corrected).all(axis=1)  # a row that is not entirely NaN
    derived = set(var[var["kind"] == "derived"].index)

    counts: dict[str, int] = {}
    for name, formula in formulas.itertuples(index=False):
        try:
            evaluated = working.eval(formula)
        except (pd.errors.UndefinedVariableError, KeyError):
            if name in derived:
                counts[name] = int(with_corrected.sum())
            continue
        if name in derived:
            values = np.asarray(evaluated, dtype=np.float64)
            finite = np.isfinite(values)
            finite_corrected = finite & with_corrected
            working.loc[finite_corrected, name] = values[finite_corrected]
            count = int((with_corrected & ~finite).sum())
            if count > 0:
                counts[name] = count
        else:
            working[name] = evaluated

    result = (working[var.index].to_numpy(dtype=np.float64) - mean) / sd
    return result, counts


def _metrics_entry(metrics: BatchMetrics | None) -> dict[str, Any]:
    if metrics is None:
        return {}
    result: dict[str, Any] = {
        "batch_median": metrics.batch_median,
        "features_over": metrics.features_over,
        "chance": metrics.chance,
        "top": metrics.top,
        "classifier_by_condition": {
            "conditions": metrics.classifier_by_condition.index.to_list(),
            "values": metrics.classifier_by_condition.to_numpy(),
        },
        "chance_by_condition": {
            "conditions": metrics.chance_by_condition.index.to_list(),
            "values": metrics.chance_by_condition.to_numpy(),
        },
        "features": metrics.shares.index.to_list(),
        "shares": metrics.shares.to_numpy(),
    }
    if metrics.classifier is not None:
        result["classifier"] = metrics.classifier
    if metrics.condition_median is not None:
        result["condition_median"] = metrics.condition_median
    return result


def _shuffled_batches(conditions: pd.Series, batches: pd.Series) -> pd.Series:
    shuffled = batches.copy()
    rng = np.random.default_rng(0)
    used, _ = used_conditions(conditions, batches)
    for condition in used["condition"]:
        in_condition = conditions == condition
        permuted = rng.permutation(batches[in_condition].to_numpy())
        shuffled[in_condition] = permuted
    return shuffled


def default_keep(exp: Experiment) -> list[str]:
    return [
        col_obs
        for col, col_obs in zip(exp.metadata_columns, metadata_obs_columns(exp), strict=True)
        if col in exp.condition_columns
        and any((val := meta.get(col)) is not None and str(val).strip() != "" for meta in exp.well_metadata.values())
    ]
