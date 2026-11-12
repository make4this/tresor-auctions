"""Tests du nettoyage des adjudications."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tresor.clean import auctions as ca

TODAY = date(2026, 11, 12)


def fiscal_row(**overrides: str) -> dict[str, str]:
    row = {column: "null" for column in ca.FISCALDATA_COLUMNS}
    row.update(
        cusip="912828SY7",
        auction_date="2015-05-26",
        announcemt_date="2015-05-21",
        issue_date="2015-06-01",
        maturity_date="2017-05-31",
        security_type="Note",
        security_term="2-Year",
        original_security_term="2-Year",
        reopening="No",
        inflation_index_security="No",
        floating_rate="No",
        cash_management_bill_cmb="No",
        offering_amt="26000000000",
        high_yield="0.648",
        bid_to_cover_ratio="3.00",
        comp_accepted="100",
        primary_dealer_accepted="30",
        direct_bidder_accepted="10",
        indirect_bidder_accepted="60",
    )
    row.update(overrides)
    return row


def td_row(**overrides: str) -> dict[str, str]:
    row = {field: "" for field in ca.TREASURYDIRECT_TO_FISCALDATA}
    row.update(
        competitiveAuctionCloseDate="2026-11-10T00:00:00",
        cusip="91282CRX1",
        announcementDate="2026-11-04T00:00:00",
        issueDate="2026-11-16T00:00:00",
        maturityDate="2036-11-15T00:00:00",
        securityType="Note",
        securityTerm="10-Year",
        originalSecurityTerm="10-Year",
        reopening="No",
        inflationProtected="No",
        floatingRate="No",
        cashManagementBill="No",
        offeringAmount="42000000000",
        highYield="4.100",
        bidToCoverRatio="2.50",
        primaryDealersCompetitiveParAwarded="10",
        directBiddersCompetitiveParAwarded="20",
        indirectBiddersCompetitiveParAwarded="70",
    )
    row.update(overrides)
    return row


def table(*rows: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), dtype=str)


def empty_td() -> pd.DataFrame:
    return pd.DataFrame(columns=list(td_row()), dtype=str)


def clean(
    fiscal: pd.DataFrame, recent: pd.DataFrame | None = None, upcoming: pd.DataFrame | None = None
):
    return ca.clean_auctions(
        fiscal,
        empty_td() if recent is None else recent,
        empty_td() if upcoming is None else upcoming,
        TODAY,
    )


def test_known_reopening_is_classified_on_its_remaining_maturity() -> None:
    # Vrai cas de Fiscal Data : un ancien 5 ans revendu lors de l'enchère 2 ans du 26/05/2015.
    row = fiscal_row(original_security_term="5-Year", reopening="Yes")

    auctions, _ = clean(table(row))

    assert auctions.loc[0, "maturity_bucket"] == "2Y"
    assert auctions.loc[0, "family"] == "nominal_coupon"
    assert bool(auctions.loc[0, "reopening"]) is True


@pytest.mark.parametrize(
    ("term", "years"),
    [
        ("9-Year 10-Month", 9 + 10 / 12),
        ("2-Year", 2.0),
        ("13-Week", 13 * 7 / 365.25),
        ("42-Day", 42 / 365.25),
    ],
)
def test_term_in_years(term: str, years: float) -> None:
    assert ca.term_in_years(term) == pytest.approx(years)


def test_unreadable_term_is_an_error() -> None:
    with pytest.raises(ValueError, match="durée illisible"):
        ca.term_in_years("Ten years")


@pytest.mark.parametrize(
    ("term", "family", "bucket"),
    [
        ("4-Year 10-Month", "nominal_coupon", "5Y"),
        ("29-Year 10-Month", "nominal_coupon", "30Y"),
        ("9-Year 8-Month", "tips", "10Y"),
        ("1-Year 11-Month", "frn", "2Y"),
        ("42-Day", "cmb", "6W"),
        ("119-Day", "cmb", "17W"),
        ("52-Week", "bill", "52W"),
    ],
)
def test_maturity_bucket(term: str, family: str, bucket: str) -> None:
    assert ca.maturity_bucket(ca.term_in_years(term), family)[0] == bucket


def test_families_follow_flags_not_security_type() -> None:
    rows = [
        fiscal_row(cusip="A", inflation_index_security="Yes"),
        fiscal_row(cusip="B", floating_rate="Yes"),
        fiscal_row(
            cusip="C", security_type="Bill", security_term="42-Day", cash_management_bill_cmb="Yes"
        ),
        fiscal_row(cusip="D", security_type="Bill", security_term="13-Week"),
        fiscal_row(cusip="E", security_type="Bond", security_term="30-Year"),
    ]

    auctions, _ = clean(table(*rows))

    assert dict(zip(auctions["cusip"], auctions["family"], strict=True)) == {
        "A": "tips",
        "B": "frn",
        "C": "cmb",
        "D": "bill",
        "E": "nominal_coupon",
    }


def test_treasurydirect_results_replace_a_fiscaldata_row_without_results() -> None:
    fiscal = table(
        fiscal_row(cusip="91282CRX1", auction_date="2026-11-10", bid_to_cover_ratio="null")
    )

    auctions, report = clean(fiscal, recent=table(td_row()))

    assert len(auctions) == 1
    assert auctions.loc[0, "source"] == "treasurydirect"
    assert auctions.loc[0, "btc"] == 2.5
    assert report["duplicates_removed"] == 1


def test_when_both_have_results_treasurydirect_wins() -> None:
    fiscal = table(
        fiscal_row(cusip="91282CRX1", auction_date="2026-11-10", bid_to_cover_ratio="2.40")
    )

    auctions, _ = clean(fiscal, recent=table(td_row()))

    assert auctions.loc[0, "btc"] == 2.5


def test_treasurydirect_competitive_total_is_the_sum_of_the_three_bidders() -> None:
    auctions, _ = clean(table(fiscal_row()), recent=table(td_row()))

    recent = auctions[auctions["source"] == "treasurydirect"].iloc[0]
    assert recent["comp_accepted"] == 100
    assert recent["indirect_share"] == pytest.approx(0.7)


def test_exclusions_are_reported_with_their_reason() -> None:
    rows = [
        fiscal_row(cusip="SMALL", offering_amt="25000000"),
        fiscal_row(cusip="NEVER", auction_date="2001-09-11", bid_to_cover_ratio="null"),
        fiscal_row(cusip="FUTURE", auction_date="2026-12-01", bid_to_cover_ratio="null"),
        fiscal_row(cusip="KEPT"),
    ]

    auctions, report = clean(table(*rows))

    assert auctions["cusip"].tolist() == ["KEPT"]
    assert {reason: part["rows"] for reason, part in report["excluded"].items()} == {
        "petite_emission_hors_calendrier": 1,
        "passee_sans_resultat": 1,
        "future_hors_calendrier": 1,
    }


def test_calendar_rows_are_upcoming_even_on_the_auction_day() -> None:
    upcoming = table(
        td_row(
            cusip="TODAY",
            competitiveAuctionCloseDate="2026-11-12T00:00:00",
            bidToCoverRatio="",
            offeringAmount="",
        )
    )

    auctions, _ = clean(table(fiscal_row()), upcoming=upcoming)

    today_row = auctions[auctions["cusip"] == "TODAY"].iloc[0]
    assert today_row["status"] == "upcoming"
    assert pd.isna(today_row["offering_amt"])  # montant pas encore annoncé


def test_bills_use_the_bond_equivalent_rate() -> None:
    row = fiscal_row(
        security_type="Bill",
        security_term="13-Week",
        high_yield="null",
        high_investment_rate="3.733",
    )

    auctions, _ = clean(table(row))

    assert auctions.loc[0, "auction_rate"] == pytest.approx(3.733)
    assert auctions.loc[0, "dealer_share"] == pytest.approx(0.3)


def test_unexpected_flag_value_is_an_error() -> None:
    with pytest.raises(ValueError, match="reopening"):
        clean(table(fiscal_row(reopening="Maybe")))
