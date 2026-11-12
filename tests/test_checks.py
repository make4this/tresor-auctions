"""Tests des contrôles de qualité."""

from __future__ import annotations

from datetime import date

import pandas as pd

from tresor.clean import checks
from tresor.sources.collect import RawSource

TODAY = date(2026, 11, 12)


def source(name: str = "table") -> RawSource:
    return RawSource(name, lambda _: pd.DataFrame(), "day", "%Y-%m-%d", ("day",), ("day",))


def days(*values: str) -> pd.DataFrame:
    return pd.DataFrame({"day": list(values)}, dtype=str)


def test_business_days_skip_weekends_and_federal_holidays() -> None:
    # Jeudi 5/11 -> jeudi 12/11 : vendredi, lundi, mardi, jeudi ; le 11/11 est férié.
    assert int(checks.business_days_between(date(2026, 11, 5), TODAY)) == 4


def test_stale_table_raises_an_alert() -> None:
    result = checks.check_raw_table(days("2026-10-01"), source(), None, TODAY)

    assert result["status"] == "rouge"
    assert "jours ouvrés de retard" in result["alerts"][0]


def test_fresh_table_is_green() -> None:
    assert checks.check_raw_table(days("2026-11-10"), source(), None, TODAY)["status"] == "vert"


def test_freshness_uses_publication_date_when_known() -> None:
    frame = days("2026-10-28").assign(available_at="2026-11-05")

    assert checks.check_raw_table(frame, source(), None, TODAY)["status"] == "vert"


def test_known_lag_of_fiscal_data_is_tolerated() -> None:
    result = checks.check_raw_table(days("2026-10-15"), source("fiscaldata_auctions"), None, TODAY)

    assert result["status"] == "vert"


def test_drop_in_a_past_year_is_detected() -> None:
    previous = {"rows_by_year": {"2020": 10, "2026": 3}}

    result = checks.check_raw_table(
        days(*["2020-01-02"] * 9, "2026-11-10"), source(), previous, TODAY
    )

    assert result["alerts"] == ["2020 : 9 lignes contre 10 à l'exécution précédente"]


def auctions(**values: float) -> pd.DataFrame:
    row = {
        "cusip": "X",
        "auction_date": pd.Timestamp("2026-11-10"),
        "family": "nominal_coupon",
        "maturity_bucket": "10Y",
        "status": "completed",
        "btc": 2.5,
        "dealer_share": 0.2,
        "direct_share": 0.2,
        "indirect_share": 0.6,
        "auction_rate": 4.1,
    }
    row.update(values)
    return pd.DataFrame([row])


def test_values_outside_bounds_are_flagged() -> None:
    result = checks.check_auctions(auctions(btc=7.0))

    assert result["ranges"]["btc"]["outside"] == 1
    assert result["status"] == "orange"  # valeur extrême à vérifier, pas une erreur


def test_negative_real_yield_of_tips_is_not_flagged() -> None:
    result = checks.check_auctions(auctions(family="tips", auction_rate=-1.6))

    assert result["ranges"]["auction_rate"]["outside"] == 0
    assert result["status"] == "vert"


def test_duplicate_auction_is_an_error() -> None:
    result = checks.check_auctions(pd.concat([auctions(), auctions()]))

    assert result["status"] == "rouge"


def test_gap_in_a_daily_series_is_flagged() -> None:
    dates = pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-15"])
    market = pd.DataFrame(
        {"series": "yield_10y", "date": dates, "value": 4.0, "available_at": dates}
    )

    result = checks.check_market(market)

    # Du vendredi 2/10 au jeudi 15/10 : 7 jours ouvrés manquants (le 12/10 est férié).
    assert result["largest_gaps"]["yield_10y"][0]["missing_business_days"] == 7
    assert result["status"] == "rouge"
