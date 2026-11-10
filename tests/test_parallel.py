"""parallel_map keeps going past a failed item, and a stop keeps the items that finished."""

import functools
from pathlib import Path

import jobs
import pytest

from fouri import report
from fouri.parallel import Dropped, Failed, check_results, parallel_map
from fouri.report import StepFailed


def test_one_failing_item_does_not_stop_the_others() -> None:
    results = parallel_map(
        functools.partial(jobs.square_unless, bad=3), [1, 2, 3, 4], desc="Squaring", unit="item", n_workers=2
    )

    assert results[:2] == [1, 4]
    assert isinstance(results[2], Failed)
    assert results[2].item == "3"
    assert results[3] == 16


def test_outcomes_come_back_in_item_order() -> None:
    # the first item waits longest, so the items finish in the reverse of their order
    waits = [1.2, 0.8, 0.4, 0.0]

    results = parallel_map(jobs.wait_then_report, waits, desc="Waiting", unit="item", n_workers=len(waits))

    assert [result[0] for result in results] == waits
    finish_times = [result[1] for result in results]
    assert finish_times[-1] < finish_times[0], "the items finished in order, so the test proved nothing"


def test_a_failure_carries_the_items_name() -> None:
    results = parallel_map(
        functools.partial(jobs.square_unless, bad=2),
        [1, 2, 3],
        desc="Squaring",
        unit="well",
        n_workers=2,
        names=["B02", "B03", "C02"],
    )

    assert isinstance(results[1], Failed)
    assert results[1].item == "B03"


def test_a_killed_worker_fails_every_unfinished_item() -> None:
    items = [1, 2, 3, 4]

    results = parallel_map(
        functools.partial(jobs.square_or_exit, bad=2), items, desc="Squaring", unit="item", n_workers=1
    )

    assert results[0] == 1
    for item, result in zip(items[1:], results[1:], strict=True):
        assert isinstance(result, Failed)
        assert result.item == str(item)


def test_a_stop_drops_waiting_items_and_keeps_finished(tmp_path: Path) -> None:
    progress_file = tmp_path / "step.jsonl"
    report.report_to(progress_file)
    items = list(range(1, 9))  # one worker takes up to four items before the stop is seen

    results = parallel_map(
        functools.partial(jobs.create_stop_file, stop_file=report.stop_path(progress_file)),
        items,
        desc="Stopping",
        unit="item",
        n_workers=1,
    )

    assert results[0] == 1
    assert isinstance(results[-1], Dropped)
    for item, result in zip(items, results, strict=True):
        assert result == item or result == Dropped(str(item))


def test_check_results_raises_step_failed_listing_each_failed_item_once() -> None:
    # a well with two failed rounds comes back as two outcomes named after it
    results = [
        1,
        Failed("B03", "round 1 did not align"),
        Failed("B03", "round 2 did not align"),
        Failed("C02", "no labels"),
    ]

    with pytest.raises(StepFailed) as raised:
        check_results(results, unit="well")

    assert raised.value.failed == ["B03", "C02"]


def test_check_results_raises_stopped_when_any_item_was_dropped() -> None:
    results = [1, Failed("B03", "no labels"), Dropped("C02")]

    with pytest.raises(report.Stopped):
        check_results(results, unit="well")


def test_check_results_is_quiet_when_every_item_finished(capsys: pytest.CaptureFixture[str]) -> None:
    check_results([1, 4, 9], unit="well")

    assert capsys.readouterr().out == ""
