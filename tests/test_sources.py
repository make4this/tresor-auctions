"""Tests de la collecte, sans réseau : httpx.MockTransport joue le rôle des serveurs."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from tresor.http import FetchError, build_client
from tresor.sources import cboe, fiscaldata, treasury_rates, treasurydirect
from tresor.sources.collect import RawSource, check_raw, collect_all
from tresor.sources.raw import RawDataError

Handler = Callable[[httpx.Request], httpx.Response]
Fetcher = Callable[[httpx.Client], pd.DataFrame]


def make_client(handler: Handler) -> httpx.Client:
    return build_client(transport=httpx.MockTransport(handler))


# --- Fiscal Data ---------------------------------------------------------------------------


def fiscal_page(page: int, total_pages: int, total_count: int) -> httpx.Response:
    data = [{"cusip": f"C{page}", "auction_date": "2026-11-10", "offering_amt": "null"}]
    meta = {"total-pages": total_pages, "total-count": total_count}
    return httpx.Response(200, json={"data": data, "meta": meta})


def test_fiscaldata_reads_every_page_and_keeps_raw_text() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return fiscal_page(int(request.url.params["page[number]"]), total_pages=2, total_count=2)

    frame = fiscaldata.fetch_auctions(make_client(handler))

    assert [r.url.params["page[number]"] for r in requests] == ["1", "2"]
    assert requests[0].url.params["filter"] == "auction_date:gte:2000-01-01"
    assert requests[0].url.params["sort"] == "auction_date,cusip"  # tri unique
    assert frame["cusip"].tolist() == ["C1", "C2"]
    assert frame["offering_amt"].tolist() == ["null", "null"]  # rien n'est interprété ici


def test_fiscaldata_detects_a_truncated_download() -> None:
    client = make_client(lambda request: fiscal_page(1, total_pages=1, total_count=5))

    with pytest.raises(RawDataError, match="1 lignes reçues sur 5"):
        fiscaldata.fetch_auctions(client)


def test_fiscaldata_detects_a_row_repeated_across_pages() -> None:
    # Le total est bon (2 lignes), mais la même enchère revient deux fois : une autre manque.
    client = make_client(lambda request: fiscal_page(1, total_pages=2, total_count=2))

    with pytest.raises(RawDataError, match="en double"):
        fiscaldata.fetch_auctions(client)


# --- TreasuryDirect ------------------------------------------------------------------------


def test_treasurydirect_asks_for_tips_and_frn_and_removes_duplicates() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        security_type = request.url.params["type"]
        requested.append(security_type)
        # Le type CMB renvoie ici le même titre que Bill : il ne doit compter qu'une fois.
        cusip = "C-Bill" if security_type == "CMB" else f"C-{security_type}"
        row = {"cusip": cusip, "competitiveAuctionCloseDate": "2026-11-09T00:00:00"}
        return httpx.Response(200, json=[row])

    frame = treasurydirect.fetch_recent(make_client(handler))

    assert requested == ["Bill", "Note", "Bond", "CMB", "TIPS", "FRN"]
    assert sorted(frame["cusip"]) == ["C-Bill", "C-Bond", "C-FRN", "C-Note", "C-TIPS"]


def test_treasurydirect_upcoming_keeps_unannounced_amount_empty() -> None:
    rows = [{"cusip": "A", "offeringAmount": ""}, {"cusip": "B", "offeringAmount": "42000000000"}]
    client = make_client(lambda request: httpx.Response(200, json=rows))

    frame = treasurydirect.fetch_upcoming(client)

    assert frame["offeringAmount"].tolist() == ["", "42000000000"]


# --- Treasury.gov --------------------------------------------------------------------------


def treasury_server(bodies: dict[int, str], calls: list[int]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        year = int(request.url.params["field_tdr_date_value"])
        calls.append(year)
        return httpx.Response(200, text=bodies[year])

    return make_client(handler)


CURVE_BODIES = {
    2000: "Date,3 Mo\n01/03/2000,5.48\n",
    2001: "Date,3 Mo,1 Mo\n01/02/2001,5.87,\n",
    2002: "Date,1 Mo\n01/02/2002,1.73\n",
}


def test_yield_curve_accepts_columns_that_change_with_the_year(tmp_path: Path) -> None:
    client = treasury_server(CURVE_BODIES, [])

    frame = treasury_rates.fetch_yield_curve(client, current_year=2002, cache_dir=tmp_path)

    assert list(frame.columns) == ["Date", "3 Mo", "1 Mo"]
    assert frame["3 Mo"].tolist()[:2] == ["5.48", "5.87"]
    assert pd.isna(frame["3 Mo"].iloc[2])  # colonne absente cette année-là


def test_only_finished_years_are_cached(tmp_path: Path) -> None:
    calls: list[int] = []
    client = treasury_server(CURVE_BODIES, calls)

    treasury_rates.fetch_yield_curve(client, current_year=2002, cache_dir=tmp_path)
    treasury_rates.fetch_yield_curve(client, current_year=2002, cache_dir=tmp_path)

    assert calls == [2000, 2001, 2002, 2002]


def test_empty_finished_year_is_an_error_and_is_not_cached(tmp_path: Path) -> None:
    client = treasury_server({**CURVE_BODIES, 2001: ""}, [])

    with pytest.raises(RawDataError, match="2001"):
        treasury_rates.fetch_yield_curve(client, current_year=2002, cache_dir=tmp_path)

    assert not list(tmp_path.rglob("*_2001.csv"))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("<html>Maintenance</html>", "contenu inattendu"),
        ("Date,3 Mo\n01/03/2000,5.48\n", "autres années"),  # l'année 2000 renvoyée pour 2001
    ],
)
def test_abnormal_year_is_rejected_and_never_cached(
    tmp_path: Path, body: str, message: str
) -> None:
    client = treasury_server({**CURVE_BODIES, 2001: body}, [])

    with pytest.raises(RawDataError, match=message):
        treasury_rates.fetch_yield_curve(client, current_year=2002, cache_dir=tmp_path)

    assert not list(tmp_path.rglob("*_2001.csv"))


def test_empty_current_year_is_accepted(tmp_path: Path) -> None:
    client = treasury_server({**CURVE_BODIES, 2002: ""}, [])

    frame = treasury_rates.fetch_yield_curve(client, current_year=2002, cache_dir=tmp_path)

    assert len(frame) == 2


def test_bill_rates_start_in_2002(tmp_path: Path) -> None:
    calls: list[int] = []
    bodies = {2002: "Date,x\n01/02/2002,1\n", 2003: "Date,x\n01/02/2003,2\n"}

    frame = treasury_rates.fetch_bill_rates(
        treasury_server(bodies, calls), current_year=2003, cache_dir=tmp_path
    )

    assert calls == [2002, 2003]
    assert len(frame) == 2


# --- CBOE ----------------------------------------------------------------------------------


def test_vix_values_stay_as_received() -> None:
    text = "DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,17.240,17.24,17.24,17.24\n"
    client = make_client(lambda request: httpx.Response(200, text=text))

    frame = cboe.fetch_vix(client)

    assert frame["OPEN"].tolist() == ["17.240"]


# --- Collecte ------------------------------------------------------------------------------


def make_source(name: str, fetch: Fetcher) -> RawSource:
    return RawSource(name, fetch, "day", "%Y-%m-%d", ("day",), ("day",))


def days(*values: str) -> pd.DataFrame:
    return pd.DataFrame({"day": list(values)}, dtype=str)


@pytest.fixture
def client() -> httpx.Client:
    return make_client(lambda request: httpx.Response(200))


def test_collect_saves_parquet_and_reports_rows_and_dates(
    client: httpx.Client, tmp_path: Path
) -> None:
    frame = days("2020-01-03", "2020-01-01", "2020-01-04", "2020-01-02")

    [summary] = collect_all([make_source("a", lambda _: frame)], client, tmp_path)

    assert pd.read_parquet(tmp_path / "a.parquet")["day"].tolist() == frame["day"].tolist()
    assert summary.rows == 4
    assert (summary.first_date, summary.last_date) == (date(2020, 1, 1), date(2020, 1, 4))
    assert summary.preview["day"].tolist() == ["2020-01-02", "2020-01-03", "2020-01-04"]


def test_failure_keeps_old_file_and_does_not_stop_other_sources(
    client: httpx.Client, tmp_path: Path
) -> None:
    def broken(_: httpx.Client) -> pd.DataFrame:
        raise FetchError("panne")

    days("2019-01-01").to_parquet(tmp_path / "a.parquet", index=False)
    sources = [make_source("a", broken), make_source("b", lambda _: days("2020-01-01"))]

    with pytest.raises(FetchError, match="Sources en échec : a"):
        collect_all(sources, client, tmp_path)

    assert pd.read_parquet(tmp_path / "a.parquet")["day"].tolist() == ["2019-01-01"]
    assert (tmp_path / "b.parquet").exists()


def test_unreadable_date_is_rejected_before_saving(client: httpx.Client, tmp_path: Path) -> None:
    sources = [make_source("a", lambda _: days("pas une date"))]

    with pytest.raises(FetchError, match="Sources en échec : a"):
        collect_all(sources, client, tmp_path)

    assert not (tmp_path / "a.parquet").exists()


def test_empty_table_is_rejected() -> None:
    with pytest.raises(RawDataError, match="vide"):
        check_raw(days(), make_source("a", lambda _: days()))


def test_changed_format_is_rejected() -> None:
    frame = pd.DataFrame({"autre": ["x"]}, dtype=str)

    with pytest.raises(RawDataError, match="colonnes absentes"):
        check_raw(frame, make_source("a", lambda _: frame))


def test_missing_api_key_only_fails_its_own_source(client: httpx.Client, tmp_path: Path) -> None:
    def needs_key(_: httpx.Client) -> pd.DataFrame:
        raise RuntimeError("FRED_API_KEY manquante")

    sources = [make_source("fred", needs_key), make_source("b", lambda _: days("2020-01-01"))]

    with pytest.raises(FetchError, match="Sources en échec : fred"):
        collect_all(sources, client, tmp_path)

    assert (tmp_path / "b.parquet").exists()


def test_unexpected_error_type_does_not_stop_other_sources(
    client: httpx.Client, tmp_path: Path
) -> None:
    def odd_format(_: httpx.Client) -> pd.DataFrame:
        raise TypeError("réponse JSON imprévue")

    sources = [make_source("a", odd_format), make_source("b", lambda _: days("2020-01-01"))]

    with pytest.raises(FetchError, match="Sources en échec : a"):
        collect_all(sources, client, tmp_path)

    assert (tmp_path / "b.parquet").exists()


def test_missing_date_is_rejected(client: httpx.Client, tmp_path: Path) -> None:
    frame = pd.DataFrame({"day": ["2020-01-01", None]}, dtype=str)

    with pytest.raises(FetchError, match="Sources en échec : a"):
        collect_all([make_source("a", lambda _: frame)], client, tmp_path)


def test_accumulating_source_keeps_rows_the_api_no_longer_returns(
    client: httpx.Client, tmp_path: Path
) -> None:
    previous = pd.DataFrame(
        {"day": ["2020-01-01", "2020-01-02"], "value": ["old", "old"]}, dtype=str
    )
    previous.to_parquet(tmp_path / "a.parquet", index=False)
    new = pd.DataFrame({"day": ["2020-01-02", "2020-01-03"], "value": ["new", "new"]}, dtype=str)
    source = RawSource("a", lambda _: new, "day", "%Y-%m-%d", ("day",), ("day",), ("day",))

    collect_all([source], client, tmp_path)

    saved = pd.read_parquet(tmp_path / "a.parquet").sort_values("day")
    assert saved["day"].tolist() == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert saved["value"].tolist() == ["old", "new", "new"]  # la version la plus récente gagne
