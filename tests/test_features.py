"""Tests des variables point-in-time, dont le test anti-fuite obligatoire (section 7 du plan)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tresor import features as ft
from tresor.config import CLEAN_DIR

NAN = float("nan")
TARGET_COLUMNS = ["btc", "dealer_share", "indirect_share", "auction_rate"]


def auction(
    cusip: str,
    day: str,
    run: str = "bill_13W#1",
    btc: float = 2.5,
    rate: float = 5.10,
    status: str = "completed",
    offering: float = 6e10,
    family: str = "bill",
    bucket: str = "13W",
) -> dict[str, object]:
    date = pd.Timestamp(day)
    done = status == "completed"
    return {
        "cusip": cusip,
        "auction_date": date,
        "announcemt_date": date - pd.Timedelta(days=4),
        "family": family,
        "maturity_bucket": bucket,
        "bucket_years": 0.25 if family == "bill" else float(bucket.rstrip("Y")),
        "series_run": run,
        "status": status,
        "reopening": False,
        "offering_amt": offering,
        "auction_rate": rate if done else NAN,
        "btc": btc if done else NAN,
        "dealer_share": 0.3 if done else NAN,
        "indirect_share": 0.5 if done else NAN,
    }


def weekly_bills(count: int = 20, start: str = "2024-01-08") -> list[dict[str, object]]:
    days = pd.date_range(start, periods=count, freq="7D")
    return [auction(f"B{i}", str(d.date()), btc=2.0 + 0.1 * i) for i, d in enumerate(days)]


def monthly_notes(bucket: str = "10Y", count: int = 6) -> list[dict[str, object]]:
    days = pd.date_range("2024-01-10", periods=count, freq="28D")
    return [
        auction(f"N{bucket}{i}", str(d.date()), run=f"nominal_coupon_{bucket}#1", rate=4.0,
                family="nominal_coupon", bucket=bucket, btc=2.4 + 0.05 * i)
        for i, d in enumerate(days)
    ]  # fmt: skip


def daily(name: str, start: str, end: str, value: float = 5.0, lag_days: int = 0) -> pd.DataFrame:
    days = pd.bdate_range(start, end)
    values = value + 0.01 * np.arange(len(days))
    return pd.DataFrame(
        {
            "series": name,
            "date": days,
            "value": values,
            "available_at": days + pd.Timedelta(days=lag_days),
        }
    )


def market() -> pd.DataFrame:
    weekly = pd.date_range("2023-10-04", "2024-07-31", freq="7D")  # mercredis
    rank = np.arange(1, len(weekly) + 1)
    dealers = pd.DataFrame(
        {
            "series": "dealer_bills",
            "date": weekly,
            "value": 1000.0 * rank**2,  # en millions : k² milliards la semaine k
            "available_at": weekly + pd.Timedelta(days=8),  # publié le jeudi suivant
        }
    )
    return pd.concat(
        [
            daily("bill_13w", "2023-10-02", "2024-07-31"),
            daily("yield_10y", "2023-10-02", "2024-07-31", 4.0),
            daily("yield_7y", "2023-10-02", "2024-07-31", 3.0),
            daily("vix_close", "2023-10-02", "2024-07-31", 15.0),
            dealers,
        ],
        ignore_index=True,
    )


def build(
    rows: list[dict[str, object]],
    market_frame: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
):
    frame = market() if market_frame is None else market_frame
    return ft.build_features(pd.DataFrame(rows), frame, events)


def row_of(features: pd.DataFrame, cusip: str) -> pd.Series:
    return features.set_index("cusip").loc[cusip]


def value_on(frame: pd.DataFrame, series: str, day: str) -> float:
    rows = frame["series"].eq(series) & frame["date"].eq(pd.Timestamp(day))
    return float(frame.loc[rows, "value"].iloc[0])


# --- Historique de la même série -----------------------------------------------------------


def test_lags_are_the_previous_auctions_of_the_same_series() -> None:
    features, _ = build(weekly_bills())

    row = row_of(features, "B5")
    assert row["btc_lag1"] == pytest.approx(2.4)
    assert row["btc_lag2"] == pytest.approx(2.3)
    assert row["btc_lag3"] == pytest.approx(2.2)
    assert row["btc_mean3"] == pytest.approx(2.3)


def test_an_auction_on_the_same_day_is_not_a_previous_one() -> None:
    rows = weekly_bills(4) + [auction("SAME", "2024-01-29", btc=9.9)]  # même jour que B3

    features, _ = build(rows)

    assert row_of(features, "B3")["btc_lag1"] == pytest.approx(2.2)  # B2, pas SAME
    assert row_of(features, "SAME")["btc_lag1"] == pytest.approx(2.2)


def test_a_new_series_run_starts_without_history() -> None:
    rows = weekly_bills(4) + [auction("RESTART", "2024-09-02", run="bill_13W#2")]

    features, _ = build(rows)

    assert math.isnan(row_of(features, "RESTART")["btc_lag1"])


def test_series_do_not_mix() -> None:
    features, _ = build(weekly_bills(6) + monthly_notes())

    assert row_of(features, "N10Y2")["btc_lag1"] == pytest.approx(2.45)  # la note précédente


def test_upcoming_auction_uses_past_results_and_an_estimated_amount() -> None:
    rows = weekly_bills(4) + [auction("NEXT", "2024-02-05", status="upcoming", offering=NAN)]

    features, _ = build(rows)

    row = row_of(features, "NEXT")
    assert row["btc_lag1"] == pytest.approx(2.3)
    assert row["offering_estimated"] == 1
    assert row["log_offering"] == pytest.approx(math.log(6e10))
    assert math.isnan(row["btc"])  # la cible est inconnue


def test_amount_change_and_days_since_previous() -> None:
    rows = [auction("A", "2024-01-08", offering=5e10), auction("B", "2024-01-15", offering=6e10)]

    features, _ = build(rows)

    row = row_of(features, "B")
    assert row["offering_change"] == pytest.approx(math.log(6 / 5))
    assert row["days_since_previous"] == 7


# --- Correspondances maturité -> série ------------------------------------------------------


def test_each_maturity_uses_its_own_rate() -> None:
    market_frame = market()

    features, _ = build(monthly_notes("10Y") + monthly_notes("7Y"), market_frame)

    # Note 10 ans du 07/02/2024 : clôture du 10 ans (pas du 7 ans) de la veille.
    assert row_of(features, "N10Y1")["rate_level"] == pytest.approx(
        value_on(market_frame, "yield_10y", "2024-02-06")
    )
    assert row_of(features, "N7Y1")["rate_level"] == pytest.approx(
        value_on(market_frame, "yield_7y", "2024-02-06")
    )


def test_maturity_mappings() -> None:
    for bucket, series in ft.RATE_SERIES.items():
        unit = "y" if bucket.endswith("Y") else "w"
        assert series.endswith(f"_{bucket[:-1]}{unit}")
    assert ft.DEALER_SERIES["2Y"] == ft.DEALER_SERIES["3Y"] == "dealer_coupons_le3y"
    assert ft.DEALER_SERIES["5Y"] == "dealer_coupons_3to6y"
    assert ft.DEALER_SERIES["7Y"] == ft.DEALER_SERIES["10Y"] == "dealer_coupons_6to11y"
    assert ft.DEALER_SERIES["20Y"] == ft.DEALER_SERIES["30Y"] == "dealer_coupons_gt11y"
    assert all(ft.DEALER_SERIES[b] == "dealer_bills" for b in ft.BILL_BUCKETS)
    assert ft.CFTC_CONTRACTS["30Y"] == "ultrabond"  # le T-Bond ne livre que 15 à 25 ans
    assert ft.CFTC_CONTRACTS["20Y"] == "bond"
    assert ft.CFTC_CONTRACTS["10Y"] == ft.CFTC_CONTRACTS["7Y"] == "10y"


# --- Marché à t-1 --------------------------------------------------------------------------


def test_tail_compares_the_auction_rate_with_the_previous_close() -> None:
    market_frame = market()
    # La clôture du jour même (publiée le soir) ne doit jamais servir.
    same_day = market_frame["series"].eq("bill_13w") & market_frame["date"].eq("2024-01-29")
    market_frame.loc[same_day, "value"] = 99.0
    previous = value_on(market_frame, "bill_13w", "2024-01-26")

    features, _ = build(weekly_bills(4), market_frame)

    row = row_of(features, "B3")  # enchère du lundi 29/01/2024
    assert row["rate_level"] == pytest.approx(previous)
    assert row["tail_bp"] == pytest.approx((5.10 - previous) * 100)


def test_weekly_release_is_used_only_once_published() -> None:
    features, provenance = build(weekly_bills(4))

    # Lundi 15/01/2024 : le relevé du mercredi 03/01 (14e semaine) est publié le jeudi 11/01,
    # celui du 10/01 seulement le 18/01. Variation sur 4 semaines : 14² - 10² = 96 milliards.
    row = row_of(features, "B1")
    assert row["dealer_net_chg_4w_bn"] == pytest.approx(96.0)
    assert provenance.loc[1, "dealer_net_chg_4w_bn"] == pd.Timestamp("2024-01-11")


def test_missing_series_gives_an_empty_feature() -> None:
    features, _ = build(weekly_bills(4))

    assert features["credit_spread_baa"].isna().all()


def test_values_too_old_are_ignored() -> None:
    market_frame = pd.concat(
        [
            daily("bill_13w", "2023-10-02", "2024-07-31"),
            daily("vix_close", "2023-10-02", "2023-12-01"),
        ]
    )

    features, _ = build(weekly_bills(4), market_frame)

    assert features["vix"].isna().all()  # dernière valeur bien plus vieille que 10 jours


def test_new_dollar_index_is_preferred_and_the_old_one_fills_the_past() -> None:
    old = daily("fred_dtwexb", "2023-06-01", "2024-07-31", 100.0, lag_days=3)
    new = daily("fred_dtwexbgs", "2024-03-01", "2024-07-31", 120.0, lag_days=3)
    market_frame = pd.concat([market(), old, new], ignore_index=True)
    old_index = old.set_index("date")["value"]

    features, provenance = build(weekly_bills(20), market_frame)

    # Lundi 22/01/2024 : seul l'ancien indice existe. Avec 3 jours de délai, la valeur du
    # vendredi 19/01 n'est publiée que le 22/01 : la dernière connue avant est celle du 18/01.
    early = row_of(features, "B2")
    known_early = pd.Timestamp("2024-01-18")
    assert early["dollar_chg_20d"] == pytest.approx(
        old_index[known_early] / old_index.shift(20)[known_early] - 1
    )
    # Lundi 20/05/2024 : le nouvel indice a assez d'historique, il est préféré à l'ancien.
    late = row_of(features, "B19")
    new_index = new.set_index("date")["value"]
    known_late = pd.Timestamp("2024-05-16")
    assert late["dollar_chg_20d"] == pytest.approx(
        new_index[known_late] / new_index.shift(20)[known_late] - 1
    )
    assert provenance.loc[19, "dollar_chg_20d"] == pd.Timestamp("2024-05-19")


# --- Séries dérivées -----------------------------------------------------------------------


def frame_of(values: list[float], available: list[str] | None = None) -> pd.DataFrame:
    days = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.DataFrame(
        {
            "date": days,
            "value": values,
            "available_at": pd.to_datetime(available) if available else days,
        }
    )


def test_change_is_known_only_when_its_whole_window_is_known() -> None:
    frame = frame_of(list(np.arange(7.0)))
    frame.loc[2, "available_at"] = pd.Timestamp("2024-01-25")  # publiée 22 jours en retard

    changes = ft.change(frame, 5)

    assert changes["value"].tolist() == [5.0, 5.0]
    assert changes["available_at"].tolist() == [pd.Timestamp("2024-01-25")] * 2


def test_value_filled_years_later_is_left_out_of_windows() -> None:
    # En temps réel, la valeur du 3e jour manquait : elle n'a été complétée qu'un an plus tard.
    frame = frame_of([1.0, 2.0, 3.0, 4.0, 5.0])
    frame.loc[2, "available_at"] = pd.Timestamp("2025-01-01")

    changes = ft.change(frame, 1)

    assert changes["value"].tolist() == [1.0, 2.0, 1.0]  # 2 -> 1 -> (3 ignoré) -> 4 -> 5
    assert changes["available_at"].max() < pd.Timestamp("2024-02-01")


def test_pct_change_volatility_and_zscore() -> None:
    frame = frame_of([100.0, 110.0, 99.0, 108.9])

    assert ft.pct_change(frame, 1)["value"].tolist() == pytest.approx([0.10, -0.10, 0.10])
    vol = ft.volatility(frame, 2)["value"].tolist()
    assert vol == pytest.approx([np.std([10, -11], ddof=1), np.std([-11, 9.9], ddof=1)])
    z = ft.zscore(frame, 4)["value"].iloc[0]
    assert z == pytest.approx((108.9 - np.mean(frame["value"])) / np.std(frame["value"], ddof=1))


def test_combine_is_known_when_both_inputs_are() -> None:
    left = frame_of([4.0, 4.1])
    right = frame_of([2.0, 2.2], available=["2024-01-01", "2024-01-05"])

    slope = ft.combine(left, right, lambda a, b: a - b)

    assert slope["value"].tolist() == pytest.approx([2.0, 1.9])
    assert slope["available_at"].iloc[1] == pd.Timestamp("2024-01-05")


# --- Demande récente et calendrier ---------------------------------------------------------


def test_recent_relative_demand_uses_past_auctions_of_the_same_family() -> None:
    # Bid-to-cover alterné 2 / 3 : le bid-to-cover relatif varie et teste vraiment la fenêtre.
    days = pd.date_range("2024-01-08", periods=16, freq="7D")
    rows = [auction(f"B{i}", str(d.date()), btc=2.0 + (i % 2)) for i, d in enumerate(days)]
    notes = [{**row, "btc": 10.0} for row in monthly_notes(count=4)]  # autre famille, ignorée

    features, _ = build(rows + notes)

    done = pd.DataFrame(rows)
    past = done["btc"].shift(1).rolling(12, min_periods=6).mean()
    relative = done["btc"] / past
    expected = relative.rolling(5, min_periods=3).mean().iloc[14]  # 5 dernières avant B15
    assert row_of(features, "B15")["recent_relative_btc"] == pytest.approx(expected)
    assert math.isnan(row_of(features, "B3")["recent_relative_btc"])


def test_calendar_features() -> None:
    rows = [auction("FEB", "2024-02-26"), auction("MAR", "2024-03-25")]

    features, _ = build(rows)

    assert row_of(features, "FEB")["refunding_month"] == 1
    assert row_of(features, "MAR")["is_quarter_end"] == 1
    assert row_of(features, "MAR")["days_to_quarter_end"] == 6


def test_sample_missing_rates_and_feature_groups() -> None:
    features, _ = build(weekly_bills())
    old = features.assign(auction_date=features["auction_date"] - pd.DateOffset(years=20))

    assert ft.model_sample(old).empty  # avant 2009 : hors échantillon
    rates = ft.missing_rates(features)
    assert {"bill 2009+", "bill 2014+"} <= set(rates.columns)
    assert rates.loc["credit_spread_baa", "bill 2014+"] == 100.0
    without = ft.feature_groups_without(ft.FEATURE_GROUPS, ft.NEW_SOURCE_GROUPS)
    assert "dealer_net_z52" not in without and "rate_level" in without


# --- Variables candidates (nouvelles données de novembre 2026) --------------------------------


def with_candidates() -> pd.DataFrame:
    """Le marché de test, plus les taux allemand et japonais et la garde étrangère hebdomadaire."""
    weekly = pd.date_range("2023-10-04", "2024-07-31", freq="7D")  # mercredis
    custody = pd.DataFrame(
        {
            "series": "fred_wmtsecl1",
            "date": weekly,
            "value": 3e6 * (1 + 0.01 * np.arange(len(weekly))),
            "available_at": weekly + pd.Timedelta(days=1),  # publié le jeudi
        }
    )
    return pd.concat(
        [
            market(),
            daily("bund_10y", "2023-10-02", "2024-07-31", 2.5, lag_days=1),
            daily("jgb_10y", "2023-10-02", "2024-07-31", 0.8),
            custody,
        ],
        ignore_index=True,
    )


def events() -> pd.DataFrame:
    """Décisions de la Fed, inflation et emploi, connues le 1er décembre de l'année d'avant."""
    rows = [
        ("fomc", "2024-01-31"),
        ("fomc", "2024-03-20"),
        ("fomc", "2024-05-01"),
        ("cpi", "2024-02-13"),
        ("cpi", "2024-03-12"),
        ("jobs", "2024-02-02"),
        ("jobs", "2024-03-08"),
        ("jobs", "2025-01-10"),  # l'année suivante : pas encore publique en 2024
    ]
    frame = pd.DataFrame(rows, columns=["event", "date"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["known_at"] = pd.to_datetime((frame["date"].dt.year - 1).astype(str) + "-12-01")
    return frame


def test_days_to_and_since_announcements() -> None:
    rows = [auction("FEB", "2024-02-13"), auction("MAR", "2024-03-21")]

    features, provenance = build(rows, with_candidates(), events())

    feb, mar = row_of(features, "FEB"), row_of(features, "MAR")
    assert feb["days_to_cpi"] == 0  # inflation publiée le jour même de l'enchère
    assert feb["days_to_fomc"] == 36  # 13/2 -> 20/3
    assert feb["days_since_fomc"] == 13  # 31/1 -> 13/2
    assert feb["days_to_jobs"] == 24  # 13/2 -> 8/3
    assert mar["days_since_fomc"] == 1  # décision de la veille
    assert mar["days_to_fomc"] == 41
    assert ft.look_ahead_violations(features, provenance).sum() == 0


def test_an_announcement_not_yet_scheduled_is_left_empty() -> None:
    # Après le 8/3/2024, la prochaine publication de l'emploi connue est en janvier 2025, dont
    # la date n'est publique qu'au 1er décembre 2024 : la variable reste vide en avril 2024.
    features, _ = build([auction("APR", "2024-04-15")], with_candidates(), events())

    assert math.isnan(row_of(features, "APR")["days_to_jobs"])
    assert math.isnan(row_of(features, "APR")["days_to_cpi"])  # aucune date après mars


def test_days_are_capped() -> None:
    far = events().assign(date=lambda e: e["date"] + pd.DateOffset(years=1))
    far["known_at"] = pd.Timestamp("2023-12-01")

    features, _ = build([auction("FEB", "2024-02-13")], with_candidates(), far)

    assert row_of(features, "FEB")["days_to_fomc"] == ft.CALENDAR_CAP_DAYS


def test_without_events_the_calendar_is_empty() -> None:
    features, _ = build([auction("FEB", "2024-02-13")], with_candidates())

    assert row_of(features, "FEB")[list(ft.CALENDAR_FEATURES)].isna().all()


def test_foreign_candidates_use_the_previous_publications() -> None:
    frame = with_candidates()

    features, _ = build([auction("X", "2024-04-17")], frame)  # un mercredi

    x = row_of(features, "X")
    # Taux allemand publié le lendemain : la dernière valeur connue le 17/4 est celle du 15/4.
    expected = value_on(frame, "yield_10y", "2024-04-15") - value_on(
        frame, "bund_10y", "2024-04-15"
    )
    assert x["us_bund_spread_10y"] == pytest.approx(expected)
    # Garde étrangère : niveau du mercredi 10/4 (publié le 11/4) contre celui du 13/3.
    expected = (
        value_on(frame, "fred_wmtsecl1", "2024-04-10")
        / value_on(frame, "fred_wmtsecl1", "2024-03-13")
        - 1
    )
    assert x["foreign_custody_chg_4w"] == pytest.approx(expected)
    assert x["jgb_10y_chg_20d"] == pytest.approx(0.2)  # 20 jours de hausse de 0,01


def test_candidates_stay_out_of_the_site_model() -> None:
    assert not set(ft.CANDIDATE_COLUMNS) & set(ft.FEATURE_COLUMNS)
    assert set(ft.CANDIDATE_GROUPS) == {"foreign_official", "calendar", "foreign_rates"}


# --- Test anti-fuite obligatoire ------------------------------------------------------------


def truncated_inputs(
    auctions: pd.DataFrame,
    market_frame: pd.DataFrame,
    cusip: str,
    day: pd.Timestamp,
    events: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Ce qu'on savait le matin de l'enchère : enchères passées, données publiées avant t,
    annonces déjà au calendrier, et l'enchère du jour sans ses résultats."""
    past = auctions[auctions["auction_date"] < day]
    current = auctions[auctions["cusip"].eq(cusip) & auctions["auction_date"].eq(day)]
    current = current.assign(**{column: NAN for column in TARGET_COLUMNS}, status="upcoming")
    past_market = market_frame[market_frame["available_at"] < day]
    known_events = None if events is None else events[events["known_at"] < day]
    return pd.concat([past, current], ignore_index=True), past_market, known_events


def assert_same_features_with_truncated_data(
    auctions: pd.DataFrame,
    market_frame: pd.DataFrame,
    sample: pd.DataFrame,
    events: pd.DataFrame | None = None,
) -> None:
    """Variables du modèle et variables candidates : identiques sans le futur."""
    columns = list(ft.ALL_FEATURE_COLUMNS)
    full, _ = ft.build_features(auctions, market_frame, events)
    full = full.set_index(["cusip", "auction_date"])
    for cusip, day in zip(sample["cusip"], sample["auction_date"], strict=True):
        past_auctions, past_market, past_events = truncated_inputs(
            auctions, market_frame, cusip, day, events
        )
        partial, _ = ft.build_features(past_auctions, past_market, past_events)
        partial = partial.set_index(["cusip", "auction_date"])
        expected = full.loc[(cusip, day), columns].astype(float)
        actual = partial.loc[(cusip, day), columns].astype(float)
        pd.testing.assert_series_equal(
            actual, expected, check_names=False, obj=f"{cusip} {day:%Y-%m-%d}"
        )


def test_no_feature_uses_information_published_too_late() -> None:
    features, provenance = build(weekly_bills() + monthly_notes(), with_candidates(), events())

    assert ft.look_ahead_violations(features, provenance).sum() == 0
    assert set(ft.CANDIDATE_COLUMNS) <= set(provenance.columns)


def test_features_are_identical_when_the_future_is_erased() -> None:
    rows = weekly_bills() + monthly_notes()
    rows.append(auction("NEXT", "2024-05-27", status="upcoming", offering=NAN))
    auctions = pd.DataFrame(rows)
    sample = auctions[auctions["cusip"].isin(["B3", "B8", "B15", "B19", "N10Y2", "N10Y5", "NEXT"])]

    assert_same_features_with_truncated_data(auctions, with_candidates(), sample, events())


def test_a_leak_would_be_detected() -> None:
    features, provenance = build(weekly_bills())
    provenance.loc[5, "vix"] = features.loc[5, "auction_date"]  # « publié » le jour même

    assert ft.look_ahead_violations(features, provenance)["vix"] == 1


def test_publication_dates_earlier_than_the_source_allows_are_detected() -> None:
    market_frame = market()
    dealers = market_frame["series"].eq("dealer_bills")
    market_frame["source"] = np.where(dealers, "nyfed_dealer_positions", "treasury_yield_curve")
    assert ft.publication_lag_violations(market_frame).sum() == 0

    market_frame.loc[dealers, "available_at"] = market_frame.loc[dealers, "date"] + pd.Timedelta(
        days=1
    )  # dealers « publiés » le lendemain au lieu du jeudi suivant

    assert ft.publication_lag_violations(market_frame)["dealer_bills"] == dealers.sum()


# --- Mêmes contrôles sur les vraies données -------------------------------------------------

REAL_DATA = (CLEAN_DIR / "auctions.parquet").exists() and (CLEAN_DIR / "market.parquet").exists()
REAL_EVENTS = CLEAN_DIR / "events.parquet"
PERIODS = (("2009-01-01", "2013-12-31"), ("2014-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31"))


@pytest.fixture(scope="module")
def real_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    auctions = pd.read_parquet(CLEAN_DIR / "auctions.parquet")
    return auctions, pd.read_parquet(CLEAN_DIR / "market.parquet")


@pytest.fixture(scope="module")
def real_events() -> pd.DataFrame | None:
    return pd.read_parquet(REAL_EVENTS) if REAL_EVENTS.exists() else None


def stratified_sample(auctions: pd.DataFrame) -> pd.DataFrame:
    """Une enchère par maturité et par période, plus la première de chaque série depuis 2009."""
    candidates = auctions[
        auctions["family"].isin(ft.MODELED_FAMILIES)
        & (auctions["status"] == "completed")
        & (auctions["auction_date"] >= "2009-01-01")
    ]
    picks = []
    for start, end in PERIODS:
        period = candidates[candidates["auction_date"].between(start, end)]
        for _, part in period.groupby("maturity_bucket"):
            picks.append(part.sample(1, random_state=42))
    starts = candidates.sort_values("auction_date").groupby("series_run").head(1)
    picks.append(starts[starts["auction_date"] >= "2009-01-01"])
    return pd.concat(picks).drop_duplicates(["cusip", "auction_date"])


@pytest.mark.skipif(not REAL_DATA, reason="données nettoyées absentes")
def test_real_data_has_no_look_ahead(
    real_inputs: tuple[pd.DataFrame, pd.DataFrame], real_events: pd.DataFrame | None
) -> None:
    auctions, market_frame = real_inputs
    features, provenance = ft.build_features(auctions, market_frame, real_events)

    assert ft.look_ahead_violations(features, provenance).sum() == 0
    assert ft.publication_lag_violations(market_frame).sum() == 0


@pytest.mark.skipif(not REAL_DATA, reason="données nettoyées absentes")
def test_real_features_are_identical_when_the_future_is_erased(
    real_inputs: tuple[pd.DataFrame, pd.DataFrame], real_events: pd.DataFrame | None
) -> None:
    auctions, market_frame = real_inputs
    sample = stratified_sample(auctions)
    assert set(sample["maturity_bucket"]) >= {"2Y", "5Y", "7Y", "10Y", "20Y", "13W", "8W", "17W"}

    assert_same_features_with_truncated_data(auctions, market_frame, sample, real_events)
