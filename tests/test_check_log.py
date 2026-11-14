"""Tests du contrôle du journal figé lancé par le workflow du soir."""

from __future__ import annotations

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
