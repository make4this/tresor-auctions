"""Tests des dates de disponibilité : c'est ce qui empêche d'utiliser une donnée trop tôt."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pandas as pd
import pytest

from tresor.http import build_client
from tresor.sources import cftc, fiscaldata, nyfed
from tresor.sources.raw import RawDataError, add_business_days, to_iso_text

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler) -> httpx.Client:
    return build_client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    ("start", "count", "expected"),
    [
        ("2026-11-05", 1, "2026-11-06"),  # jeudi -> vendredi
        ("2026-11-06", 1, "2026-11-09"),  # vendredi -> lundi, le week-end ne compte pas
        ("2026-11-05", 2, "2026-11-09"),  # jeudi + 2 jours ouvrés -> lundi
        ("2026-09-04", 1, "2026-09-08"),  # le lundi 7/9 est férié (Labor Day)
        ("2026-11-07", 1, "2026-11-09"),  # une date de week-end repart du vendredi
    ],
)
def test_add_business_days(start: str, count: int, expected: str) -> None:
    result = add_business_days(pd.Series(pd.to_datetime([start])), count)

    assert to_iso_text(result).tolist() == [expected]


def fiscal_payload(rows: list[dict[str, str]]) -> dict[str, object]:
    return {"data": rows, "meta": {"total-pages": 1, "total-count": len(rows)}}


def test_debt_becomes_available_two_business_days_later() -> None:
    requests: list[httpx.Request] = []
    rows = [{"record_date": "2026-11-05", "tot_pub_debt_out_amt": "1"}]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=fiscal_payload(rows))

    frame = fiscaldata.fetch_debt(make_client(handler))

    assert requests[0].url.path.endswith("/v2/accounting/od/debt_to_penny")
    assert frame["available_at"].tolist() == ["2026-11-09"]


def test_cash_balance_becomes_available_one_business_day_later() -> None:
    rows = [{"record_date": "2026-11-06", "account_type": "x"}]
    client = make_client(lambda request: httpx.Response(200, json=fiscal_payload(rows)))

    frame = fiscaldata.fetch_cash_balance(client)

    assert frame["available_at"].tolist() == ["2026-11-09"]


NYFED_HEADER = '"As Of Date","Time Series","Value (millions)"'


def nyfed_key(request: httpx.Request) -> str:
    return request.url.path.rsplit("/", 1)[1].removesuffix(".csv")


def test_dealer_positions_are_published_the_following_thursday() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = nyfed_key(request)
        requested.append(key)
        return httpx.Response(200, text=f'{NYFED_HEADER}\n"2026-10-28","{key}","*"\n')

    frame = nyfed.fetch_dealer_positions(make_client(handler))

    assert requested == [*nyfed.LEGACY_SERIES, *nyfed.CURRENT_SERIES]  # une requête par série
    assert sorted(frame[nyfed.SERIES_COLUMN]) == sorted(requested)
    assert set(frame["available_at"]) == {"2026-11-05"}  # mercredi 28/10 -> jeudi 5/11
    assert set(frame[nyfed.VALUE_COLUMN]) == {"*"}  # valeur masquée conservée telle quelle


def test_dealer_release_on_a_holiday_thursday_moves_to_the_next_business_day() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        row = f'"2026-11-18","{nyfed_key(request)}","1"'
        return httpx.Response(200, text=f"{NYFED_HEADER}\n{row}\n")

    frame = nyfed.fetch_dealer_positions(make_client(handler))

    assert set(frame["available_at"]) == {"2026-11-27"}  # le jeudi 26/11 est Thanksgiving


def test_empty_dealer_series_is_detected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        key = nyfed_key(request)
        row = "" if key == "PDPOSGS-B" else f'\n"2026-10-28","{key}","1"\n'
        return httpx.Response(200, text=NYFED_HEADER + row)

    with pytest.raises(RawDataError, match="série vide : PDPOSGS-B"):
        nyfed.fetch_dealer_positions(make_client(handler))


@pytest.mark.parametrize(
    "report_date",
    ["2026-11-02T00:00:00.000", "2026-11-03T00:00:00.000", "2026-11-04T00:00:00.000"],
)
def test_cftc_report_becomes_public_on_the_friday_of_its_week(report_date: str) -> None:
    rows = [{cftc.DATE_COLUMN: report_date, cftc.CODE_COLUMN: "043602"}]
    client = make_client(lambda request: httpx.Response(200, json=rows))

    frame = cftc.fetch_positions(client)

    assert frame["available_at"].tolist() == ["2026-11-06"]


@pytest.mark.parametrize(
    ("report_date", "published"),
    [
        # Calendrier officiel 2026 : publications du lundi « à cause d'un jour férié fédéral ».
        ("2025-12-30", "2026-01-05"),  # Nouvel An le jeudi
        ("2026-06-16", "2026-06-22"),  # Juneteenth le vendredi
        ("2026-11-10", "2026-11-16"),  # Veterans Day le mercredi
        ("2026-11-24", "2026-11-30"),  # Thanksgiving le jeudi
        ("2026-01-20", "2026-01-23"),  # le lundi férié précède le rapport : pas de retard
        ("2025-01-07", "2025-01-13"),  # deuil national (Carter) le jeudi 9/1/2025
        # Annonces officielles de retards.
        ("2019-01-08", "2019-02-08"),  # fermeture de l'administration 2018-2019
        ("2023-01-31", "2023-02-24"),  # incident informatique chez ION
        ("2025-09-30", "2025-11-19"),  # fermeture de l'administration 2025
        ("2013-10-08", "2013-12-31"),  # 2013 : pas de date officielle, choix prudent
        ("2008-03-04", "2010-12-03"),  # historique TFF publié le 3/12/2010
    ],
)
def test_cftc_release_dates_match_official_announcements(report_date: str, published: str) -> None:
    dates = pd.Series(pd.to_datetime([report_date]))

    assert to_iso_text(cftc.release_dates(dates)).tolist() == [published]


def test_cftc_asks_for_the_six_contracts_and_reads_every_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cftc, "PAGE_SIZE", 2)
    requests: list[httpx.Request] = []
    row = {cftc.DATE_COLUMN: "2026-11-03T00:00:00.000", cftc.CODE_COLUMN: "043602"}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[row, row] if len(requests) == 1 else [row])

    frame = cftc.fetch_positions(make_client(handler))

    assert [r.url.params["$offset"] for r in requests] == ["0", "2"]
    assert len(frame) == 3
    for code in ("042601", "044601", "043602", "043607", "020601", "020604"):
        assert code in requests[0].url.params["$where"]
