"""Tests du contrôle du journal figé lancé par le workflow du soir."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tests.test_live import EMPTY, predictions

from tresor import check_log, live

NOW = pd.Timestamp("2026-11-18 23:17", tz="UTC")  # le soir de l'enchère du 18


def write(log: pd.DataFrame, path: Path) -> Path:
    log.to_parquet(path, index=False)
    return path


def logged() -> pd.DataFrame:
    return live.update_log(EMPTY, predictions(2.5), pd.Timestamp("2026-11-10 23:17", tz="UTC"))


def test_an_unchanged_log_passes(tmp_path: Path) -> None:
    old = write(logged(), tmp_path / "old.parquet")
    new = write(logged(), tmp_path / "new.parquet")

    assert check_log.check(old, new, NOW) is None


def test_a_changed_or_removed_frozen_row_is_refused(tmp_path: Path) -> None:
    old = write(logged(), tmp_path / "old.parquet")
    changed = write(logged().assign(pred=3.0), tmp_path / "changed.parquet")
    removed = write(logged().iloc[0:0], tmp_path / "removed.parquet")

    assert "changé" in check_log.check(old, changed, NOW)
    assert "disparu" in check_log.check(old, removed, NOW)
    assert "disparu" in check_log.check(old, tmp_path / "absent.parquet", NOW)


def test_a_row_not_yet_frozen_may_change(tmp_path: Path) -> None:
    old = write(logged(), tmp_path / "old.parquet")
    new = write(logged().assign(pred=3.0), tmp_path / "new.parquet")

    assert check_log.check(old, new, pd.Timestamp("2026-11-12 23:17", tz="UTC")) is None


def test_the_command_line_exit_codes(tmp_path: Path) -> None:
    # Enchère de 2020 : figée quelle que soit la date à laquelle le test tourne.
    past = live.update_log(EMPTY, predictions(2.5, auction=pd.Timestamp("2020-01-15")),
                           pd.Timestamp("2020-01-10 23:17", tz="UTC"))  # fmt: skip
    old = write(past, tmp_path / "old.parquet")
    changed = write(past.assign(pred=3.0), tmp_path / "changed.parquet")

    assert check_log.main([str(old), str(old)]) == 0
    assert check_log.main([str(old), str(changed)]) == 1
    assert check_log.main([]) == 2
    assert check_log.main([str(tmp_path / "premier.parquet"), str(changed)]) == 0


def test_a_frozen_row_added_after_the_fact_is_refused(tmp_path: Path) -> None:
    old = write(logged(), tmp_path / "old.parquet")
    other = live.update_log(EMPTY, predictions(2.5, cusip="B"),
                            pd.Timestamp("2026-11-10 23:17", tz="UTC"))  # fmt: skip
    new = write(pd.concat([logged(), other]), tmp_path / "new.parquet")

    assert "ajoutée après coup" in check_log.check(old, new, NOW)


def status(folder: Path, run_at: str) -> None:
    (folder / live.STATUS_FILE).write_text(json.dumps({"run_at": run_at}), encoding="utf-8")


def test_the_check_uses_the_date_of_the_evening_run(tmp_path: Path) -> None:
    """Prévision écrite à 23 h 59 à New York le 17, contrôle à 0 h 01 le 18 : la ligne du 18
    n'était pas figée au moment de l'écriture, elle ne doit pas être jugée figée."""
    old = write(logged(), tmp_path / "old.parquet")
    refreshed = live.update_log(logged(), predictions(3.0),
                                pd.Timestamp("2026-11-18 04:59", tz="UTC"))  # fmt: skip
    folder = tmp_path / "output"
    folder.mkdir()
    new = write(refreshed, folder / live.LOG_FILE)
    after_midnight = pd.Timestamp("2026-11-18 05:01", tz="UTC")

    assert "changé" in check_log.check(old, new, after_midnight)  # sans statut : refusé
    status(folder, "2026-11-18T04:59:00+00:00")
    assert check_log.check(old, new, after_midnight) is None
    status(folder, "2026-11-10T04:59:00+00:00")  # statut ancien : ignoré
    assert "changé" in check_log.check(old, new, after_midnight)
    status(folder, "2026-11-18T06:00:00+00:00")  # statut venu du futur : ignoré
    assert "changé" in check_log.check(old, new, after_midnight)
    (folder / live.STATUS_FILE).write_text("pas du json", encoding="utf-8")
    assert "changé" in check_log.check(old, new, after_midnight)


def test_a_status_from_the_future_is_ignored(tmp_path: Path) -> None:
    """Un statut daté du lendemain (horloge fausse) ferait juger figée une ligne qui ne l'est
    pas encore : il est ignoré."""
    next_day = pd.Timestamp("2026-11-19")
    first = live.update_log(EMPTY, predictions(2.5, auction=next_day),
                            pd.Timestamp("2026-11-10 23:17", tz="UTC"))  # fmt: skip
    old = write(first, tmp_path / "old.parquet")
    folder = tmp_path / "output"
    folder.mkdir()
    new = write(
        live.update_log(
            first, predictions(3.0, auction=next_day), pd.Timestamp("2026-11-18 04:59", tz="UTC")
        ),
        folder / live.LOG_FILE,
    )
    status(folder, "2026-11-19T06:00:00+00:00")

    assert check_log.check(old, new, pd.Timestamp("2026-11-18 05:01", tz="UTC")) is None
