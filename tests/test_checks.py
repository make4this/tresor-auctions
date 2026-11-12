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


def test_a_drop_stays_flagged_until_it_is_repaired() -> None:
    previous = {"rows_by_year": {"2020": 10, "2026": 3}}
    damaged = days(*["2020-01-02"] * 9, "2026-11-10", "2026-11-10", "2026-11-10")

    first = checks.check_raw_table(damaged, source(), previous, TODAY)
    second = checks.check_raw_table(damaged, source(), first, TODAY)

    expected = ["2020 : 9 lignes contre 10 au maximum déjà vu"]
    assert first["alerts"] == expected
    assert second["alerts"] == expected  # la référence garde le maximum, l'alerte reste


def test_a_drop_in_the_current_year_is_also_detected() -> None:
    previous = {"rows_by_year_reference": {"2026": 100}}

    result = checks.check_raw_table(days("2026-11-10"), source(), previous, TODAY)

    assert result["alerts"] == ["2026 : 1 lignes contre 100 au maximum déjà vu"]


def test_cumulative_treasurydirect_table_is_protected_against_drops() -> None:
    previous = {"rows_by_year_reference": {"2026": 100}}

    result = checks.check_raw_table(
        days("2026-11-10"), source("treasurydirect_recent"), previous, TODAY
    )

    assert result["status"] == "rouge"


def auctions(**values: object) -> pd.DataFrame:
    row = {
        "cusip": "X",
        "auction_date": pd.Timestamp("2026-11-10"),
        "family": "nominal_coupon",
        "maturity_bucket": "10Y",
        "series_key": "nominal_coupon_10Y",
        "status": "completed",
        "btc": 2.5,
        "dealer_share": 0.2,
        "direct_share": 0.2,
        "indirect_share": 0.6,
        "auction_rate": 4.1,
    }
    row.update(values)
    return pd.DataFrame([row])


def test_recent_value_outside_bounds_is_a_warning() -> None:
    result = checks.check_auctions(auctions(btc=7.0), TODAY)

    assert result["ranges"]["btc"]["outside"] == 1
    assert result["status"] == "orange"  # valeur extrême à vérifier, pas une erreur


def test_old_extreme_value_is_reported_but_not_a_warning() -> None:
    result = checks.check_auctions(
        auctions(btc=10.7, auction_date=pd.Timestamp("2015-09-29")), TODAY
    )

    assert result["ranges"]["btc"]["outside"] == 1
    assert result["status"] == "vert"


def test_negative_real_yield_of_tips_is_not_flagged() -> None:
    result = checks.check_auctions(auctions(family="tips", auction_rate=-1.6), TODAY)

    assert result["ranges"]["auction_rate"]["outside"] == 0
    assert result["status"] == "vert"


def test_duplicate_auction_is_an_error() -> None:
    result = checks.check_auctions(pd.concat([auctions(), auctions()]), TODAY)

    assert result["status"] == "rouge"


def weekly_auctions(last: str) -> pd.DataFrame:
    dates = pd.date_range(end=last, periods=10, freq="7D")
    return pd.concat(
        [
            auctions(cusip=f"B{i}", auction_date=d, family="bill", series_key="bill_13W")
            for i, d in enumerate(dates)
        ]
    )


def test_regular_series_without_a_recent_auction_is_flagged() -> None:
    # Bill hebdomadaire dont la dernière enchère connue date d'un mois : une enchère manque.
    result = checks.check_auctions(weekly_auctions("2026-10-12"), TODAY)

    assert result["stale_regular_series"][0]["series_key"] == "bill_13W"
    assert result["status"] == "rouge"


def test_regular_series_up_to_date_is_green() -> None:
    assert checks.check_auctions(weekly_auctions("2026-11-09"), TODAY)["status"] == "vert"


def market_series(name: str, dates: list[str], published: list[str] | None = None) -> pd.DataFrame:
    days_ = pd.to_datetime(dates)
    available = pd.to_datetime(published) if published else days_
    return pd.DataFrame({"series": name, "date": days_, "value": 4.0, "available_at": available})


def test_gap_in_a_daily_series_is_flagged() -> None:
    later = [str(d.date()) for d in pd.bdate_range("2026-10-15", "2026-11-12")]
    market = market_series("yield_10y", ["2026-10-01", "2026-10-02", *later])

    result = checks.check_market(market, TODAY)

    # Du vendredi 2/10 au jeudi 15/10 : 7 jours ouvrés manquants (le 12/10 est férié).
    gaps = result["series"]["yield_10y"]["largest_gaps_since_2009"]
    assert gaps[0]["missing_business_days"] == 7
    assert result["status"] == "rouge"


def test_series_that_stopped_is_flagged() -> None:
    dates = [str(d.date()) for d in pd.bdate_range("2026-08-03", "2026-09-01")]

    result = checks.check_market(market_series("yield_10y", dates), TODAY)

    assert "dernière publication" in result["alerts"][0]


def test_weekly_release_is_judged_on_its_own_rhythm() -> None:
    # Indice dollar : valeurs quotidiennes publiées chaque lundi ; dernière publication le 9/11.
    dates = [str(d.date()) for d in pd.bdate_range("2026-09-28", "2026-11-06")]
    mondays = [str((pd.Timestamp(d) + pd.offsets.Week(weekday=0)).date()) for d in dates]

    result = checks.check_market(market_series("fred_dtwexbgs", dates, mondays), TODAY)

    assert result["series"]["fred_dtwexbgs"]["publication"] == "hebdomadaire"
    assert result["status"] == "vert"


def test_value_published_before_its_date_is_an_error() -> None:
    market = market_series("debt_total", ["2026-11-10"], ["2026-11-09"])

    result = checks.check_market(market, TODAY)

    assert result["values_without_valid_publication_date"] == 1
    assert result["status"] == "rouge"
