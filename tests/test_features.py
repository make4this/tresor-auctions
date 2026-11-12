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
        "bucket_years": 0.25 if family == "bill" else 10.0,
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
    dealers = pd.DataFrame(
        {
            "series": "dealer_bills",
            "date": weekly,
            "value": 1000.0 * np.arange(1, len(weekly) + 1),
            "available_at": weekly + pd.Timedelta(days=8),  # publié le jeudi suivant
        }
    )
    return pd.concat(
        [
            daily("bill_13w", "2023-10-02", "2024-07-31"),
            daily("vix_close", "2023-10-02", "2024-07-31", 15.0),
            dealers,
        ],
        ignore_index=True,
    )


def build(rows: list[dict[str, object]], market_frame: pd.DataFrame | None = None):
    frame = market() if market_frame is None else market_frame
    return ft.build_features(pd.DataFrame(rows), frame)


def row_of(features: pd.DataFrame, cusip: str) -> pd.Series:
    return features.set_index("cusip").loc[cusip]


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


def test_upcoming_auction_uses_past_results_and_an_estimated_amount() -> None:
    rows = weekly_bills(4) + [auction("NEXT", "2024-02-05", status="upcoming", offering=NAN)]

    features, _ = build(rows)

    row = row_of(features, "NEXT")
    assert row["btc_lag1"] == pytest.approx(2.3)
    assert row["offering_estimated"] == 1
    assert row["log_offering"] == pytest.approx(math.log(6e10))
    assert math.isnan(row["btc"])  # la cible est inconnue


# --- Marché à t-1 --------------------------------------------------------------------------


def test_tail_compares_the_auction_rate_with_the_previous_close() -> None:
    market_frame = market()
    # La clôture du jour même (publiée le soir) ne doit jamais servir.
    same_day = market_frame["series"].eq("bill_13w") & market_frame["date"].eq("2024-01-29")
    market_frame.loc[same_day, "value"] = 99.0
    previous = market_frame.loc[
        market_frame["series"].eq("bill_13w") & market_frame["date"].eq("2024-01-26"), "value"
    ].iloc[0]

    features, _ = build(weekly_bills(4), market_frame)

    row = row_of(features, "B3")  # enchère du lundi 29/01/2024
    assert row["rate_level"] == pytest.approx(previous)
    assert row["tail_bp"] == pytest.approx((5.10 - previous) * 100)


def test_weekly_release_is_used_only_once_published() -> None:
    features, _ = build(weekly_bills(4))

    # Lundi 15/01/2024 : le relevé du mercredi 03/01 est publié le jeudi 11/01, celui du 10/01
    # seulement le 18/01. On doit donc lire le relevé du 03/01.
    row = row_of(features, "B1")
    assert row["dealer_net_bn"] == pytest.approx(14.0)  # 14e mercredi depuis le 04/10/2023


def test_missing_series_gives_an_empty_feature() -> None:
    features, _ = build(weekly_bills(4))

    assert features["credit_spread_baa"].isna().all()


def test_change_is_known_only_when_its_whole_window_is_known() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=7),
            "value": np.arange(7.0),
            "available_at": pd.bdate_range("2024-01-01", periods=7),
        }
    )
    frame.loc[2, "available_at"] = pd.Timestamp("2024-03-01")  # une valeur publiée en retard

    changes = ft.change(frame, 5)

    assert changes["available_at"].tolist() == [pd.Timestamp("2024-03-01")] * 2


def test_values_too_old_are_ignored() -> None:
    old_vix = daily("vix_close", "2023-10-02", "2023-12-01", 15.0)

    features, _ = build(
        weekly_bills(4), pd.concat([daily("bill_13w", "2023-10-02", "2024-07-31"), old_vix])
    )

    assert features["vix"].isna().all()  # dernière valeur bien plus vieille que 10 jours


# --- Demande récente et calendrier ---------------------------------------------------------


def test_recent_relative_demand() -> None:
    rows = [
        auction(f"B{i}", str(d.date()), btc=2.0)
        for i, d in enumerate(pd.date_range("2024-01-08", periods=15, freq="7D"))
    ]

    features, _ = build(rows)

    # Bid-to-cover constant : rapport de 1 dès qu'il y a assez d'historique.
    assert row_of(features, "B14")["recent_relative_btc"] == pytest.approx(1.0)
    assert math.isnan(row_of(features, "B3")["recent_relative_btc"])


def test_calendar_features() -> None:
    rows = [auction("FEB", "2024-02-26"), auction("MAR", "2024-03-25")]

    features, _ = build(rows)

    assert row_of(features, "FEB")["refunding_month"] == 1
    assert row_of(features, "MAR")["is_quarter_end"] == 1
    assert row_of(features, "MAR")["days_to_quarter_end"] == 6


# --- Test anti-fuite obligatoire ------------------------------------------------------------


def truncated_inputs(
    auctions: pd.DataFrame, market_frame: pd.DataFrame, cusip: str, day: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ce qu'on savait le matin de l'enchère : enchères passées, données publiées avant t, et
    l'enchère du jour sans ses résultats."""
    past = auctions[auctions["auction_date"] < day]
    current = auctions[auctions["cusip"].eq(cusip) & auctions["auction_date"].eq(day)]
    current = current.assign(**{column: NAN for column in TARGET_COLUMNS}, status="upcoming")
    return pd.concat([past, current], ignore_index=True), market_frame[
        market_frame["available_at"] < day
    ]


def assert_same_features_with_truncated_data(
    auctions: pd.DataFrame, market_frame: pd.DataFrame, sample: pd.DataFrame
) -> None:
    full, _ = ft.build_features(auctions, market_frame)
    full = full.set_index(["cusip", "auction_date"])
    for cusip, day in zip(sample["cusip"], sample["auction_date"], strict=True):
        past_auctions, past_market = truncated_inputs(auctions, market_frame, cusip, day)
        partial, _ = ft.build_features(past_auctions, past_market)
        expected = full.loc[(cusip, day), list(ft.FEATURE_COLUMNS)].astype(float)
        actual = (
            partial.set_index(["cusip", "auction_date"])
            .loc[(cusip, day), list(ft.FEATURE_COLUMNS)]
            .astype(float)
        )
        pd.testing.assert_series_equal(
            actual, expected, check_names=False, obj=f"{cusip} {day:%Y-%m-%d}"
        )


def test_no_feature_uses_information_published_too_late() -> None:
    features, provenance = build(weekly_bills())

    assert ft.look_ahead_violations(features, provenance).sum() == 0


def test_features_are_identical_when_the_future_is_erased() -> None:
    auctions = pd.DataFrame(
        weekly_bills() + [auction("NEXT", "2024-05-27", status="upcoming", offering=NAN)]
    )
    sample = auctions.iloc[[3, 8, 15, 19]]

    assert_same_features_with_truncated_data(auctions, market(), sample)


def test_a_leak_would_be_detected() -> None:
    features, provenance = build(weekly_bills())
    provenance.loc[5, "vix"] = features.loc[5, "auction_date"]  # « publié » le jour même

    assert ft.look_ahead_violations(features, provenance)["vix"] == 1


# --- Mêmes contrôles sur les vraies données -------------------------------------------------

REAL_DATA = (CLEAN_DIR / "auctions.parquet").exists() and (CLEAN_DIR / "market.parquet").exists()


@pytest.fixture(scope="module")
def real_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(CLEAN_DIR / "auctions.parquet"), pd.read_parquet(
        CLEAN_DIR / "market.parquet"
    )


@pytest.mark.skipif(not REAL_DATA, reason="données nettoyées absentes")
def test_real_data_has_no_look_ahead(real_inputs: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    features, provenance = ft.build_features(*real_inputs)

    assert ft.look_ahead_violations(features, provenance).sum() == 0


@pytest.mark.skipif(not REAL_DATA, reason="données nettoyées absentes")
def test_real_features_are_identical_when_the_future_is_erased(
    real_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    auctions, market_frame = real_inputs
    candidates = auctions[
        auctions["family"].isin(ft.MODELED_FAMILIES)
        & (auctions["status"] == "completed")
        & (auctions["auction_date"] >= "2009-01-01")
    ]
    sample = candidates.sample(8, random_state=42)

    assert_same_features_with_truncated_data(auctions, market_frame, sample)
