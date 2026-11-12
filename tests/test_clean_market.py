"""Tests des séries de marché et de la jointure sans fuite d'information."""

from __future__ import annotations

import pandas as pd
import pytest

from tresor.clean import market as cm


def frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, dtype=str)


def test_tga_follows_its_three_successive_formats() -> None:
    rows = [
        {
            "record_date": "2021-09-30",
            "account_type": "Federal Reserve Account",
            "open_today_bal": "1",
            "close_today_bal": "215160",
        },
        {
            "record_date": "2021-10-01",
            "account_type": "Treasury General Account (TGA)",
            "open_today_bal": "215160",
            "close_today_bal": "132452",
        },
        {
            "record_date": "2022-04-18",
            "account_type": "Treasury General Account (TGA) Closing Balance",
            "open_today_bal": "841253",
            "close_today_bal": "null",
        },
        {
            "record_date": "2022-04-18",
            "account_type": "Treasury General Account (TGA) Opening Balance",
            "open_today_bal": "578473",
            "close_today_bal": "null",
        },
    ]
    raw = frame(rows).assign(available_at=["2021-10-01", "2021-10-04", "2022-04-19", "2022-04-19"])

    tga = cm.tga_long(raw)

    assert tga["value"].tolist() == [215160, 132452, 841253]


def dealer_rows(date: str, values: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"As Of Date": date, "Time Series": key, "Value (millions)": value, "available_at": "x"}
        for key, value in values.items()
    ]


def test_dealer_buckets_join_old_and_new_keys() -> None:
    old = {
        "PDPUSGTBNOP": "1",
        "PDPUSGCS3LNOP": "57827",
        "PDPUSGCS36NOP": "3",
        "PDPUSGCS611NOP": "4",
        "PDPUSGCSM11NOP": "5",
    }
    new = {
        "PDPOSGS-B": "1",
        "PDPOSGSC-L2": "49433",
        "PDPOSGSC-G2L3": "7777",
        "PDPOSGSC-G3L6": "3",
        "PDPOSGSC-G6L7": "2",
        "PDPOSGSC-G7L11": "2",
        "PDPOSGSC-G11L21": "*",  # valeur masquée : la tranche reste inconnue ce jour-là
        "PDPOSGSC-G21": "5",
    }
    raw = frame(dealer_rows("2013-03-27", old) + dealer_rows("2022-01-05", new))
    raw["available_at"] = raw["As Of Date"]

    dealers, overlaps = cm.dealers_long(raw)
    dealers = dealers.dropna(subset=["value"])  # comme build_market

    le3y = dealers[dealers["series"] == "dealer_coupons_le3y"]["value"].tolist()
    assert le3y == [57827, 49433 + 7777]
    assert dealers[dealers["series"] == "dealer_coupons_gt11y"]["value"].tolist() == [5]
    assert dealers[dealers["series"] == "dealer_total"]["value"].tolist() == [1 + 57827 + 3 + 4 + 5]
    assert overlaps == 0


def test_yield_curve_skips_empty_cells_and_reports_new_columns() -> None:
    raw = frame([{"Date": "11/10/2026", "2 Yr": "3.50", "10 Yr": "", "15 Yr": "4.0"}])

    curve, unknown = cm.yield_curve_long(raw)

    assert curve.dropna(subset=["value"])["series"].tolist() == ["yield_2y"]
    assert unknown == ["15 Yr"]
    assert (curve["available_at"] == curve["date"]).all()


def test_cftc_nets_are_long_minus_short() -> None:
    raw = frame(
        [
            {
                "report_date_as_yyyy_mm_dd": "2026-11-03T00:00:00.000",
                "cftc_contract_market_code": "043602",
                "asset_mgr_positions_long": "100",
                "asset_mgr_positions_short": "30",
                "lev_money_positions_long": "10",
                "lev_money_positions_short": "50",
                "open_interest_all": "500",
                "available_at": "2026-11-06",
            }
        ]
    )

    nets = cm.cftc_long(raw).set_index("series")["value"]

    assert nets["cftc_10y_am_net"] == 70
    assert nets["cftc_10y_lev_net"] == -40
    assert nets["cftc_10y_oi"] == 500


def series(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime([r[0] for r in rows]),
            "value": [r[1] for r in rows],
            "available_at": pd.to_datetime([r[2] for r in rows]),
        }
    )


def test_point_in_time_never_uses_a_value_published_on_the_auction_day() -> None:
    values = series([("2026-11-09", 1.0, "2026-11-09"), ("2026-11-10", 2.0, "2026-11-10")])
    auctions = pd.Series(pd.to_datetime(["2026-11-10", "2026-11-11"]))

    known = cm.point_in_time(auctions, values)

    assert known["value"].tolist() == [1.0, 2.0]


def test_point_in_time_takes_the_latest_observation_of_a_weekly_release() -> None:
    # Indice dollar : les valeurs du lundi au vendredi sont publiées ensemble le lundi suivant.
    week = [(f"2026-11-0{day}", float(day), "2026-11-09") for day in range(2, 7)]
    auctions = pd.Series(pd.to_datetime(["2026-11-06", "2026-11-10"]))

    known = cm.point_in_time(auctions, series(week))

    assert pd.isna(known["value"].iloc[0])  # rien n'était encore publié
    assert known["value"].iloc[1] == 6.0
    assert known["observed_on"].iloc[1] == pd.Timestamp("2026-11-06")


def test_old_value_published_late_does_not_replace_a_newer_one() -> None:
    values = series([("2016-10-07", 1.0, "2017-03-22"), ("2017-03-20", 2.0, "2017-03-21")])

    known = cm.point_in_time(pd.Series(pd.to_datetime(["2017-03-23"])), values)

    assert known["value"].iloc[0] == 2.0


def test_point_in_time_keeps_the_order_of_the_input() -> None:
    values = series([("2026-11-02", 1.0, "2026-11-02"), ("2026-11-05", 2.0, "2026-11-05")])
    auctions = pd.Series(pd.to_datetime(["2026-11-06", "2026-11-03"]), index=[10, 20])

    known = cm.point_in_time(auctions, values)

    assert known.index.tolist() == [10, 20]
    assert known["value"].tolist() == [2.0, 1.0]


@pytest.mark.parametrize("marker", ["", ".", "null", "*", "N/A"])
def test_missing_markers_become_missing(marker: str) -> None:
    assert cm._numbers(pd.Series([marker], dtype=str)).isna().all()


def test_reverse_repo_without_operation_is_zero_on_business_days_only() -> None:
    raw = frame(
        [
            {
                "series_id": "RRPONTSYD",
                "date": "2010-06-01",
                "value": ".",
                "available_at": "2010-06-02",
            },
            {
                "series_id": "RRPONTSYD",
                "date": "2010-07-05",
                "value": ".",
                "available_at": "2010-07-06",
            },
            {
                "series_id": "WALCL",
                "date": "2010-06-02",
                "value": ".",
                "available_at": "2010-06-04",
            },
        ]
    )

    fred, zeros = cm.fred_long(raw)

    # Mardi 1/6/2010 : aucune opération, donc 0. Lundi 5/7/2010 : jour férié, reste manquant.
    assert fred["value"].iloc[0] == 0.0
    assert pd.isna(fred["value"].iloc[1])
    assert pd.isna(fred["value"].iloc[2])  # la règle ne concerne que le reverse repo
    assert zeros == 1


def test_every_series_has_a_unit() -> None:
    assert cm.unit_of("fred_rrpontsyd") == "milliards de dollars"
    assert cm.unit_of("dealer_bills") == "millions de dollars"
    assert cm.unit_of("cftc_10y_am_net") == "contrats"
    with pytest.raises(KeyError):
        cm.unit_of("serie_inconnue")


@pytest.mark.parametrize(
    "dates",
    [
        pd.Series(pd.to_datetime(["2026-11-05 10:00"])),  # une heure
        pd.Series(pd.to_datetime(["2026-11-05"])).dt.tz_localize("America/New_York"),  # un fuseau
    ],
)
def test_auction_dates_with_time_or_timezone_are_refused(dates: pd.Series) -> None:
    values = series([("2026-11-05", 1.0, "2026-11-05")])

    with pytest.raises(ValueError, match="dates d'enchère"):
        cm.point_in_time(dates, values)


def test_values_older_than_max_age_are_ignored() -> None:
    values = series([("2008-12-16", 25.0, "2008-12-17")])
    auctions = pd.Series(pd.to_datetime(["2008-12-20", "2009-12-07"]))

    known = cm.point_in_time(auctions, values, max_age=pd.Timedelta(days=10))

    assert known["value"].iloc[0] == 25.0
    assert pd.isna(known["value"].iloc[1])  # 356 jours plus tard : trop vieux
