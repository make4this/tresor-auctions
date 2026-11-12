"""Tests de la source FRED, sans réseau ni vraie clé."""

from __future__ import annotations

import httpx
import pandas as pd
import pytest

from tresor.http import build_client
from tresor.sources import fred
from tresor.sources.raw import to_iso_text

VINTAGES = ["2019-02-04", "2019-02-11", "2019-02-19"]

# Toutes les versions archivées (output_type=1), comme les renvoie ALFRED.
ALL_VERSIONS = [
    # Valeur ancienne entrée dans l'archive le jour de son ouverture : pas une vraie date.
    {"date": "2019-01-28", "value": "1", "realtime_start": "2019-02-04"},
    {"date": "2019-02-04", "value": "2", "realtime_start": "2019-02-11"},
    # D'abord publiée vide, puis chiffrée une semaine plus tard : c'est le chiffre qui compte.
    {"date": "2019-02-08", "value": ".", "realtime_start": "2019-02-11"},
    {"date": "2019-02-08", "value": "3", "realtime_start": "2019-02-19"},
]

LATEST = [
    {"date": d, "value": "1", "realtime_start": "2026-11-12", "realtime_end": "2026-11-12"}
    for d in ("2019-01-28", "2019-02-04", "2019-02-08")
]


def fred_client(
    requests: list[httpx.Request], latest: list[dict[str, str]] = LATEST
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/vintagedates"):
            return httpx.Response(200, json={"vintage_dates": VINTAGES, "count": len(VINTAGES)})
        rows = ALL_VERSIONS if "realtime_start" in request.url.params else latest
        return httpx.Response(200, json={"observations": rows, "count": len(rows)})

    return build_client(transport=httpx.MockTransport(handler))


def test_available_at_uses_the_first_published_number() -> None:
    series = fred.FredSeries("X", "test", fred.friday_after)

    frame = fred.fetch_series(fred_client([]), series, api_key="cle-de-test")

    assert frame["available_at"].tolist() == [
        "2019-01-30",  # avant l'archive : règle (date + 2 jours)
        "2019-02-11",  # vraie date de première publication
        "2019-02-19",  # date du premier chiffre, pas celle de la version vide
    ]
    assert frame["available_at_source"].tolist() == ["regle", "alfred", "alfred"]


def test_series_created_with_its_archive_has_no_earlier_public_history() -> None:
    # Indice dollar : la série n'existait pas avant l'ouverture de son archive.
    series = fred.FredSeries("X", "test", None)

    frame = fred.fetch_series(fred_client([]), series, api_key="cle-de-test")

    assert frame["available_at"].iloc[0] == "2019-02-04"


def test_availability_is_never_before_the_value_date() -> None:
    # Jour férié après l'ouverture de l'archive : valeur vide, jamais publiée en chiffre.
    holiday = {"date": "2019-02-18", "value": ".", "realtime_start": "x", "realtime_end": "x"}
    series = fred.FredSeries("X", "test", None)

    frame = fred.fetch_series(fred_client([], [*LATEST, holiday]), series, api_key="cle-de-test")

    assert frame["available_at"].iloc[-1] == "2019-02-18"


def test_vintages_are_split_into_windows_the_api_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fred, "MAX_VINTAGES_PER_REQUEST", 2)

    assert fred.vintage_windows(VINTAGES) == [
        ("2019-02-04", "2019-02-11"),
        ("2019-02-19", "2019-02-19"),
    ]


def test_each_window_is_requested_with_all_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fred, "MAX_VINTAGES_PER_REQUEST", 2)
    requests: list[httpx.Request] = []

    fred.first_release_dates(fred_client(requests), "X", api_key="cle-de-test")

    windows = [
        (r.url.params["realtime_start"], r.url.params["realtime_end"], r.url.params["output_type"])
        for r in requests
        if "realtime_start" in r.url.params
    ]
    assert windows == [("2019-02-04", "2019-02-11", "1"), ("2019-02-19", "2019-02-19", "1")]


def dates(*values: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(values)))


def test_publication_rules() -> None:
    assert to_iso_text(fred.friday_after(dates("2026-11-04"))).tolist() == ["2026-11-06"]
    assert to_iso_text(fred.next_business_day(dates("2026-11-06"))).tolist() == ["2026-11-09"]
    # Vendredi 6/11 + 2 jours ouvrés = mardi 10/11.
    assert to_iso_text(fred.two_business_days(dates("2026-11-06"))).tolist() == ["2026-11-10"]
