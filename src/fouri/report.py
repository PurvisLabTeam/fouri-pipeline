"""Utils for printing and logging progress."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Never, overload

from tqdm.auto import tqdm


def say(text: str) -> None:
    tqdm.write(text)


@overload
def progress[T](
    items: Iterable[T],
    *,
    desc: str | None = None,
    unit: str = "it",
    total: float | None = None,
    leave: bool = True,
) -> tqdm[T]: ...
@overload
def progress(
    items: None = None,
    *,
    desc: str | None = None,
    unit: str = "it",
    total: float | None = None,
    leave: bool = True,
) -> tqdm[Never]: ...
def progress(
    items: Iterable[Any] | None = None,
    *,
    desc: str | None = None,
    unit: str = "it",
    total: float | None = None,
    leave: bool = True,
) -> tqdm[Any]:
    return tqdm(items, desc=desc, total=total, leave=leave, unit=unit)
