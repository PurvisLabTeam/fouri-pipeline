from __future__ import annotations

import re
import tokenize
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

MISSING = "No column named {column}. Build cell table will skip this feature."
CANT_READ = "This formula can't be read."
CANT_COMPUTE = "This formula can't be computed: {error}"

_QUOTED_NAME = re.compile(r"'(.*)'")
_BACKTICKED = re.compile(r"`([^`]*)`")
_BACKTICK_PREFIX = "BACKTICK_QUOTED_STRING_"


@dataclass(frozen=True)
class FormulaCheck:
    name: str
    cells: int | None
    problem: str
    stops_build: bool


def _missing_column(error: Exception, formula: str, columns: pd.Index) -> str:
    if isinstance(error, KeyError):
        return str(error.args[0]) if error.args else formula
    found = _QUOTED_NAME.search(str(error))
    name = found.group(1) if found else str(error)
    if name.startswith(_BACKTICK_PREFIX):
        quoted = [column for column in _BACKTICKED.findall(formula) if column not in columns]
        if quoted:
            return quoted[0]
    return name


def _first_line(error: Exception) -> str:
    lines = str(error).strip().splitlines()
    return lines[0] if lines else type(error).__name__


def _cells_with_value(values: pd.Series) -> int:
    try:
        numbers = pd.to_numeric(values, errors="coerce")
        return int(np.isfinite(numbers.to_numpy(dtype=float, na_value=np.nan)).sum())
    except (TypeError, ValueError):
        return 0


def check_formulas(table: pd.DataFrame, specs: Sequence[Mapping[str, str]]) -> list[FormulaCheck]:
    work = table.copy(deep=False)
    checks = []
    for spec in specs:
        name, formula = spec["name"], spec["formula"]
        try:
            work[name] = work.eval(formula)
        except (pd.errors.UndefinedVariableError, KeyError) as error:
            problem = MISSING.format(column=_missing_column(error, formula, work.columns))
            checks.append(FormulaCheck(name, None, problem, stops_build=False))
            continue
        except (SyntaxError, tokenize.TokenError):
            checks.append(FormulaCheck(name, None, CANT_READ, stops_build=True))
            continue
        except Exception as error:
            problem = CANT_COMPUTE.format(error=_first_line(error))
            checks.append(FormulaCheck(name, None, problem, stops_build=True))
            continue
        checks.append(FormulaCheck(name, _cells_with_value(work[name]), "", stops_build=False))
    return checks
