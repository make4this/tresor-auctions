"""Tests de la ligne de commande."""

from __future__ import annotations

import pytest

from tresor import pipeline
from tresor.config import OUTPUT_DIR
from tresor.http import FetchError
from tresor.sources.collect import FETCH_STATUS_FILE


@pytest.fixture
def executed_steps(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Remplace chaque étape par une fonction qui note simplement son passage."""
    executed: list[str] = []

    def recorder(name: str) -> pipeline.StepFunction:
        return lambda: executed.append(name)

    fake_steps = {name: ("", recorder(name)) for name in pipeline.STEPS}
    monkeypatch.setattr(pipeline, "STEPS", fake_steps)
    return executed


def test_help_lists_every_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        pipeline.main(["--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for option in (
        "--all",
        "--fetch",
        "--clean",
        "--backtest",
        "--strategy",
        "--train",
        "--predict",
    ):
        assert option in output


def test_running_without_option_is_an_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        pipeline.main([])

    assert exit_info.value.code == 2


def test_steps_follow_pipeline_order_not_option_order(executed_steps: list[str]) -> None:
    assert pipeline.main(["--predict", "--fetch"]) == 0
    assert executed_steps == ["fetch", "predict"]


def test_all_runs_every_step(executed_steps: list[str]) -> None:
    assert pipeline.main(["--all"]) == 0
    assert executed_steps == ["fetch", "clean", "backtest", "strategy", "train", "predict"]


def test_a_failed_source_does_not_block_the_evening_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[str] = []

    def partial_fetch() -> None:
        executed.append("fetch")
        raise FetchError("Sources en échec : fred")

    steps = {
        "fetch": ("", partial_fetch),
        "clean": ("", lambda: executed.append("clean")),
        "predict": ("", lambda: executed.append("predict")),
    }
    monkeypatch.setattr(pipeline, "STEPS", steps)

    # L'échec reste signalé, avec un code distinct d'une erreur bloquante.
    assert pipeline.main(["--all"]) == pipeline.PARTIAL_FETCH_EXIT == 3
    assert executed == ["fetch", "clean", "predict"]


def test_all_stops_at_the_first_failing_step(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[str] = []

    def broken() -> None:
        executed.append("clean")
        raise FetchError("panne")

    steps = {
        "fetch": ("", lambda: executed.append("fetch")),
        "clean": ("", broken),
        "predict": ("", lambda: executed.append("predict")),
    }
    monkeypatch.setattr(pipeline, "STEPS", steps)

    assert pipeline.main(["--all"]) == 1
    assert executed == ["fetch", "clean"]


def test_unexpected_error_gives_an_error_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> None:
        raise TypeError("imprévu")

    monkeypatch.setattr(pipeline, "STEPS", {"fetch": ("", broken)})

    assert pipeline.main(["--fetch"]) == 1


def test_collection_failure_gives_an_error_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> None:
        raise FetchError("panne")

    monkeypatch.setattr(pipeline, "STEPS", {"fetch": ("", broken)})

    assert pipeline.main(["--fetch"]) == 3


def test_an_unexpected_error_in_a_middle_step_stops_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[str] = []

    def failed_fetch() -> None:
        executed.append("fetch")
        raise FetchError("Sources en échec : fred")

    def broken_clean() -> None:
        executed.append("clean")
        raise KeyError("colonne absente")

    steps = {
        "fetch": ("", failed_fetch),
        "clean": ("", broken_clean),
        "predict": ("", lambda: executed.append("predict")),
    }
    monkeypatch.setattr(pipeline, "STEPS", steps)

    # Une erreur bloquante l'emporte sur la panne de source : code 1, pas 3.
    assert pipeline.main(["--all"]) == pipeline.BLOCKING_ERROR_EXIT == 1
    assert executed == ["fetch", "clean"]


def test_the_evening_fetch_writes_its_report(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[dict[str, object]] = []
    monkeypatch.setattr(pipeline, "collect_all", lambda **kwargs: received.append(kwargs))

    pipeline._run_fetch()

    assert received == [{"status_path": OUTPUT_DIR / FETCH_STATUS_FILE}]
