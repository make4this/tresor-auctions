"""Tests des nouvelles données de novembre 2026 : sources, calendrier, nettoyage et test."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tresor import features as ft
from tresor import models as md
from tresor import new_data as nd
from tresor.clean import checks
from tresor.clean import market as cm
from tresor.clean.events import build_events
from tresor.sources import fed_calendar, foreign_rates
from tresor.sources.raw import RawDataError

# --- Taux étrangers ------------------------------------------------------------------------

BUND_CSV = (
    '﻿"",BBSIS.D.I,BBSIS.D.I_FLAGS\n'
    '"",Term structure / residual maturity of 10.0 years / daily data,\n'
    "unit,Prozent,\n"
    "last update,2026-11-18 12:31:09,\n"
    "1999-12-31,5.30,\n"
    "2026-11-14,.,No value available\n"
    "2026-11-16,3.45,\n"
    "2026-11-17,3.42,\n"
)
JGB_CSV = (
    "Interest Rate (November 2026),,,,(Unit : %)\n"
    "Date,1Y,2Y,10Y,30Y\n"
    "1999/12/30,0.1,0.2,1.7,-\n"
    "2026/11/2,1.671,1.88,2.817,3.94\n"
    "2026/11/4,1.676,1.891,-,3.979\n"
)


def test_bund_keeps_data_lines_from_2000() -> None:
    frame = foreign_rates.parse_bund(BUND_CSV)

    assert list(frame.columns) == list(foreign_rates.COLUMNS)
    assert frame["date"].tolist() == ["2026-11-14", "2026-11-16", "2026-11-17"]
    assert frame["yield_10y"].tolist() == [".", "3.45", "3.42"]  # « . » reste du texte


def test_bund_without_data_is_refused() -> None:
    with pytest.raises(RawDataError):
        foreign_rates.parse_bund('"",header\nunit,Prozent,\n')


def test_jgb_dates_are_rewritten_and_old_rows_dropped() -> None:
    frame = foreign_rates.parse_jgb(JGB_CSV)

    assert frame["date"].tolist() == ["2026-11-02", "2026-11-04"]
    assert frame["yield_10y"].tolist() == ["2.817", "-"]


def test_jgb_without_header_is_refused() -> None:
    with pytest.raises(RawDataError):
        foreign_rates.parse_jgb("Interest Rate\n1,2,3\n")


def test_bund_is_known_the_next_business_day() -> None:
    raw = pd.DataFrame({"date": ["2026-11-13", "2026-11-14", "2026-11-16"],
                        "yield_10y": ["3.40", ".", "3.45"]})  # fmt: skip

    bund = cm.bund_long(raw).dropna(subset=["value"])

    # Vendredi 13 -> lundi 16 ; lundi 16 -> mardi 17.
    assert bund["available_at"].dt.strftime("%Y-%m-%d").tolist() == ["2026-11-16", "2026-11-17"]


def test_jgb_is_known_the_evening_before_the_next_tokyo_business_day() -> None:
    # Golden Week 2019 : après le vendredi 26 avril, Tokyo rouvre le mardi 7 mai. Le taux du
    # 26 est publié le 7 à 9 h 30 (Tokyo), soit le 6 au soir à New York.
    raw = pd.DataFrame({"date": ["2019-04-25", "2019-04-26", "2019-05-07"],
                        "yield_10y": ["-0.03", "-0.04", "-0.05"]})  # fmt: skip

    jgb = cm.jgb_long(raw)

    assert jgb["available_at"].dt.strftime("%Y-%m-%d").tolist() == [
        "2019-04-25",
        "2019-05-06",
        "2019-05-07",  # dernière ligne : jour de semaine suivant (mercredi 8) - 1 jour
    ]


def test_units_of_the_new_series() -> None:
    assert cm.unit_of("bund_10y") == cm.unit_of("jgb_10y") == "%"
    assert cm.unit_of("fred_wmtsecl1") == "millions de dollars"


def test_publication_rule_of_the_bund_is_enforced() -> None:
    frame = pd.DataFrame(
        {
            "source": "bundesbank_bund_yield",
            "series": "bund_10y",
            "date": pd.to_datetime(["2026-11-13"]),
            "value": [3.4],
            "available_at": pd.to_datetime(["2026-11-13"]),  # « publié » le jour même
        }
    )

    assert ft.publication_lag_violations(frame)["bund_10y"] == 1


def test_japanese_golden_week_gap_is_tolerated_but_not_a_longer_one() -> None:
    days = pd.bdate_range("2019-01-02", "2019-12-31")
    golden_week = (days > "2019-04-26") & (days < "2019-05-07")
    part = pd.DataFrame({"date": days[~golden_week], "available_at": days[~golden_week]})

    _, alerts = checks._check_series("jgb_10y", part, pd.Timestamp("2019-12-31").date())
    _, other = checks._check_series("bund_10y", part, pd.Timestamp("2019-12-31").date())

    assert alerts == []
    assert other and "trou de 6 jours ouvrés" in other[0]


# --- Calendrier de la Fed ------------------------------------------------------------------

HISTORICAL_PAGE = """
<h5>January 3 Conference Call - 2001</h5>
<h5>January 30-31 Meeting - 2001</h5>
<h5>March 2 (unscheduled) Meeting - 2001</h5>
<h5>March 17-18 (cancelled) Meeting - 2001</h5>
<h5>March 19 (notation vote) - 2001</h5>
<h5>April/May 30-1 Meeting - 2001</h5>
<h5>July 31-August 1 Meeting - 2001</h5>
"""


def block(month: str, days: str, note: str = "") -> str:
    """Un bloc de réunion comme sur la page de la Fed."""
    return (
        '<div class="row fomc-meeting">'
        f'<div class="fomc-meeting__month col-xs-5"><strong>{month}</strong></div>'
        f'<div class="fomc-meeting__date col-xs-4">{days}</div>'
        f'<div class="fomc-meeting__minutes"><p>{note}</p></div></div>'
    )


CURRENT_PAGE = "".join(
    [
        '<h4><a id="1">2026 FOMC Meetings</a></h4>',
        block("January", "27-28", "Statement on Longer-Run Goals and Monetary Policy Strategy"),
        block("Apr/May", "30-1"),
        block("August", "22 (notation vote)"),
        block("June", "16-17*"),
        '<h4><a id="2">2027 FOMC Meetings</a></h4>',
        block("March", "16-17"),
        block("October", "26-27", "Monetary Policy Strategy Meeting"),
    ]
)


def test_historical_page_keeps_scheduled_meetings_only() -> None:
    meetings = fed_calendar.parse_historical(HISTORICAL_PAGE, 2001)

    decisions = meetings["decision_date"].dt.strftime("%Y-%m-%d").tolist()
    # Pas d'appel, pas de réunion d'urgence, pas de vote écrit ; la réunion annulée reste.
    assert decisions == ["2001-01-31", "2001-03-18", "2001-05-01", "2001-08-01"]
    assert meetings["first_day"].iloc[2] == pd.Timestamp("2001-04-30")


def test_current_page_reads_each_year_and_short_month_names() -> None:
    meetings = fed_calendar.parse_current(CURRENT_PAGE)

    assert meetings["decision_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-28",
        "2026-05-01",
        "2026-06-17",
        "2027-03-17",
        "2027-10-27",
    ]
    # La mention de la stratégie de long terme (janvier) n'est pas une réunion de stratégie.
    assert meetings["decision"].tolist() == [True, True, True, True, False]


def test_calendar_round_trip(tmp_path) -> None:
    meetings = fed_calendar.parse_current(CURRENT_PAGE)[list(fed_calendar.COLUMNS)]
    path = tmp_path / "fomc.csv"

    fed_calendar.write_calendar(meetings, path)
    again = fed_calendar.read_calendar(path)

    pd.testing.assert_frame_equal(again, meetings)
    assert fed_calendar.read_calendar(tmp_path / "absent.csv").empty


def test_saved_calendar_has_eight_meetings_a_year() -> None:
    calendar = fed_calendar.read_calendar()
    if calendar.empty:
        pytest.skip("calendrier de la Fed absent")
    by_year = calendar["decision_date"].dt.year.value_counts()
    # 8 décisions par an ; 7 en 2027, l'année de la réunion de stratégie d'octobre.
    assert by_year.pop(2027) == 7
    assert (by_year == 8).all()
    assert pd.Timestamp("2012-08-01") in set(calendar["decision_date"])
    assert calendar["decision_date"].min().year == 2000


NO_RESCHEDULE = pd.DataFrame(
    {"release": [], "actual_date": pd.to_datetime([]), "scheduled_date": pd.to_datetime([])}
)


def test_events_are_known_from_december_of_the_year_before() -> None:
    releases = pd.DataFrame({"release": ["cpi", "jobs", "cpi"],
                             "date": ["2024-02-13", "2024-02-02", "2024-02-13"]})  # fmt: skip
    fomc = pd.DataFrame({"decision_date": pd.to_datetime(["2024-01-31"])})

    events = build_events(releases, fomc, NO_RESCHEDULE)

    assert events["event"].tolist() == ["cpi", "fomc", "jobs"]  # doublon retiré, trié
    assert (events["known_at"] == pd.Timestamp("2023-12-01")).all()
    assert build_events(None, fomc)["event"].tolist() == ["fomc"]


def test_only_real_releases_are_kept() -> None:
    releases = pd.DataFrame(
        {
            "release": ["cpi", "cpi", "cpi", "jobs", "jobs", "cpi_vintage", "cpi_vintage"],
            # 9/2 : révision saisonnière (pas une version de l'indice) ; 12/3 : à venir.
            "date": ["2024-02-09", "2024-02-13", "2024-03-12", "2024-08-02", "2024-08-21",
                     "2024-01-11", "2024-02-13"],
        }
    )  # fmt: skip
    fomc = pd.DataFrame({"decision_date": pd.to_datetime([])})

    events = build_events(releases, fomc, NO_RESCHEDULE)

    kept = events["date"].dt.strftime("%Y-%m-%d").tolist()
    assert kept == ["2024-02-13", "2024-03-12", "2024-08-02"]  # emploi : la première du mois


def test_moved_and_canceled_releases_keep_their_scheduled_date() -> None:
    releases = pd.DataFrame({"release": ["jobs", "jobs"], "date": ["2025-11-20", "2025-12-16"]})
    reschedules = pd.DataFrame(
        {
            "release": ["jobs", "jobs", "jobs"],
            "actual_date": pd.to_datetime(["2025-11-20", None, "2025-12-16"]),
            "scheduled_date": pd.to_datetime(["2025-10-03", "2025-11-07", "2025-12-05"]),
        }
    )
    fomc = pd.DataFrame({"decision_date": pd.to_datetime([])})

    events = build_events(releases, fomc, reschedules)

    assert events["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-10-03",
        "2025-11-07",
        "2025-12-05",
    ]
    assert (events["known_at"] == pd.Timestamp("2024-12-01")).all()


def test_saved_reschedules_come_with_a_source() -> None:
    from tresor.clean.events import RESCHEDULES_FILE

    table = pd.read_csv(RESCHEDULES_FILE, dtype=str, keep_default_na=False)
    assert (table["source"].str.startswith("https://www.bls.gov/")).all()
    assert set(table["release"]) == {"cpi", "jobs"}


# --- Le test avec et sans -------------------------------------------------------------------


def test_each_set_is_the_site_model_plus_one_group() -> None:
    base = set(md.FEATURE_SETS["toutes"])

    assert set(nd.FEATURE_SETS["toutes"]) == base
    for group, names in ft.CANDIDATE_GROUPS.items():
        assert set(nd.FEATURE_SETS[f"toutes+{group}"]) == base | set(names)
    assert set(nd.FEATURE_SETS["toutes+all_new"]) == base | set(ft.CANDIDATE_COLUMNS)
    assert len(nd.tasks()) == 2 * 4 * 5 * 5  # familles, cibles, modèles, jeux


def test_weak_signal_uses_the_site_rule() -> None:
    part = pd.DataFrame(
        {
            "actual": [2.0, 2.7, 2.4, 2.6],
            "prediction": [2.53, 2.50, 2.60, 2.58],
            "mean3": [2.61, 2.61, 2.61, 2.61],
        }
    )

    signals, right = nd.weak_signal(part)

    # −0,08 (arrondi) et −0,11 sont « plus faibles » ; −0,01 et −0,03 non.
    assert signals == 2
    assert right == 0.5
    assert nd.weak_signal(part.assign(prediction=2.61)) == (0, pytest.approx(math.nan, nan_ok=True))


def fake_predictions() -> pd.DataFrame:
    """Deux jeux sur six ans : la base se trompe de 0,2, le jeu « + calendrier » de 0,1."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2014-01-15", "2021-12-15", freq="MS") + pd.Timedelta(days=14)
    actual = 2.5 + rng.normal(0, 0.3, len(dates))
    mean3 = actual + rng.choice([-0.3, 0.3], len(dates))
    frames = []
    for feature_set, error in (("toutes", 0.2), ("toutes+calendar", 0.1)):
        for model in (*md.MODEL_NAMES, md.ENSEMBLE):
            frames.append(
                pd.DataFrame(
                    {
                        "family": "nominal_coupon",
                        "target": "btc",
                        "model": model,
                        "feature_set": feature_set,
                        "fold": dates.to_period("Q").start_time,
                        "cusip": [f"C{i}" for i in range(len(dates))],
                        "auction_date": dates,
                        "maturity_bucket": "10Y",
                        "actual": actual,
                        "prediction": actual + error,
                        "naive": mean3,
                        "mean3": mean3,
                        "constant": 2.5,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_compare_measures_the_gain_over_the_base() -> None:
    table = nd.compare(fake_predictions())

    rows = table[(table["period"] == "2014-2026") & (table["model"] == md.ENSEMBLE)]
    base = rows[rows["feature_set"] == "toutes"].iloc[0]
    new = rows[rows["feature_set"] == "toutes+calendar"].iloc[0]
    # Erreur 0,2 contre 0,3 pour mean3 : R² = 1 − 0,04/0,09 ; avec 0,1 : 1 − 0,01/0,09.
    assert base["r2_vs_mean3"] == pytest.approx(1 - 0.04 / 0.09)
    assert new["gain"] == pytest.approx(0.03 / 0.09)
    assert new["gain_ci_low"] > 0
    assert math.isnan(base["gain_ci_low"]) and base["gain"] == 0
    assert set(table["period"]) == set(nd.TEST_PERIODS)


def test_site_summary_gives_a_verdict_per_group() -> None:
    from tresor import app_data as ad

    rows = []
    for group, low, high in (("calendar", 0.01, 0.03), ("foreign_rates", -0.02, -0.01),
                             ("foreign_official", -0.01, 0.02)):  # fmt: skip
        rows.append(
            {
                "period": "2014-2026",
                "family": "bill",
                "target": "btc",
                "model": "moyenne_5",
                "feature_set": f"toutes+{group}",
                "r2_base": 0.2,
                "r2_vs_mean3": 0.21,
                "gain": (low + high) / 2,
                "gain_ci_low": low,
                "gain_ci_high": high,
                "weak_signals": 10,
                "weak_right": 0.7,
            }  # fmt: skip
        )
    base = dict(rows[0], feature_set="toutes")
    table = pd.DataFrame([*rows, base, dict(rows[0], period="2020-2026")])

    summary = ad.new_data_rows(table)

    assert summary["group"].tolist() == ["foreign_official", "calendar", "foreign_rates"]
    assert summary["verdict"].tolist() == [ad.NO_EFFECT, ad.HELPS, ad.HURTS]


def test_a_partial_run_keeps_the_other_targets(tmp_path, monkeypatch) -> None:
    previous = fake_predictions()  # bid-to-cover, déjà testé
    previous.to_parquet(tmp_path / nd.PREDICTIONS_FILE, index=False)
    new = previous[previous["model"].isin(md.MODEL_NAMES)].assign(target="tail_bp")
    new = new.drop(columns="constant")
    monkeypatch.setattr(nd.md, "run_walk_forward", lambda *args, **kwargs: new)
    sample = (
        new[["family", "auction_date"]].drop_duplicates().assign(status="completed", tail_bp=1.0)
    )
    sample.to_parquet(tmp_path / "features.parquet", index=False)

    table = nd.run_new_data_test(tmp_path, tmp_path, targets=("tail_bp",))

    assert set(table["target"]) == {"btc", "tail_bp"}
    saved = pd.read_parquet(tmp_path / nd.PREDICTIONS_FILE)
    assert set(saved["target"]) == {"btc", "tail_bp"}
    assert table.loc[table["target"] == "tail_bp", "weak_signals"].isna().all()
