"""Fiscal Data : historique des adjudications, dette totale et trésorerie quotidiennes."""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd

from tresor.config import HISTORY_START
from tresor.http import fetch_json
from tresor.sources.raw import (
    RawDataError,
    add_business_days,
    records_to_raw_frame,
    to_iso_text,
)

BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
AUCTIONS_ENDPOINT = "v1/accounting/od/auctions_query"
DEBT_ENDPOINT = "v2/accounting/od/debt_to_penny"
CASH_ENDPOINT = "v1/accounting/dts/operating_cash_balance"
PAGE_SIZE = 10_000

# Observé le 9/11/2026 vers 16h45 à New York : la dette du vendredi 6/11 n'était pas en ligne.
DEBT_PUBLICATION_LAG_BUSINESS_DAYS = 2
# Observé le même jour : la trésorerie du vendredi 6/11 était en ligne.
CASH_PUBLICATION_LAG_BUSINESS_DAYS = 1

DEBT_COLUMNS = (
    "record_date",
    "debt_held_public_amt",
    "intragov_hold_amt",
    "tot_pub_debt_out_amt",
    "available_at",
)
CASH_COLUMNS = ("record_date", "account_type", "open_today_bal", "close_today_bal", "available_at")

AUCTION_COLUMNS = (
    "cusip",
    "auction_date",
    "announcemt_date",
    "issue_date",
    "maturity_date",
    "security_type",
    "security_term",
    "original_security_term",
    "reopening",
    "offering_amt",
    "high_yield",
    "avg_med_yield",
    "low_yield",
    "high_discnt_rate",
    "high_investment_rate",
    "bid_to_cover_ratio",
    "comp_accepted",
    "comp_tendered",
    "total_accepted",
    "total_tendered",
    "primary_dealer_accepted",
    "direct_bidder_accepted",
    "indirect_bidder_accepted",
    "soma_accepted",
    "inflation_index_security",
    "floating_rate",
    "cash_management_bill_cmb",
)


def fetch_all_pages(
    client: httpx.Client,
    endpoint: str,
    filter_expression: str,
    sort: str,
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Lit toutes les pages d'une table et vérifie que rien ne manque ni ne se répète.

    Le tri doit désigner chaque ligne de façon unique (date puis cusip, par exemple) : sinon,
    d'une page à l'autre, l'API peut répéter une ligne et en sauter une sans que le total change.
    """
    records: list[dict[str, Any]] = []
    page_number = 1
    while True:
        params = {
            "page[size]": PAGE_SIZE,
            "page[number]": page_number,
            "filter": filter_expression,
            "sort": sort,
        }
        payload = fetch_json(f"{BASE_URL}/{endpoint}", params=params, client=client)
        records.extend(payload["data"])
        if page_number >= payload["meta"]["total-pages"]:
            break
        page_number += 1

    expected_rows = payload["meta"]["total-count"]
    if len(records) != expected_rows:
        raise RawDataError(f"{endpoint} : {len(records)} lignes reçues sur {expected_rows}")
    frame = records_to_raw_frame(records)
    if len(frame) and frame.duplicated(list(key_columns)).any():
        raise RawDataError(f"{endpoint} : lignes en double sur {key_columns}")
    return frame


def fetch_auctions(client: httpx.Client) -> pd.DataFrame:
    """Télécharge toutes les adjudications depuis le début de l'historique.

    Pas de filtre par type : la table ne contient que Bill, Note et Bond (vérifié), et les
    TIPS, FRN et CMB s'y repèrent par leurs indicateurs. Un seul appel ne peut rien oublier.
    """
    return fetch_all_pages(
        client,
        AUCTIONS_ENDPOINT,
        filter_expression=f"auction_date:gte:{HISTORY_START.isoformat()}",
        sort="auction_date,cusip",
        key_columns=("cusip", "auction_date"),
    )


def _fetch_daily_table(
    client: httpx.Client, endpoint: str, lag: int, key_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Télécharge une table quotidienne et note la date à laquelle chaque ligne était connue."""
    frame = fetch_all_pages(
        client,
        endpoint,
        filter_expression=f"record_date:gte:{HISTORY_START.isoformat()}",
        sort=",".join(key_columns),
        key_columns=key_columns,
    )
    record_dates = pd.to_datetime(frame["record_date"], format="%Y-%m-%d")
    return frame.assign(available_at=to_iso_text(add_business_days(record_dates, lag)))


def fetch_debt(client: httpx.Client) -> pd.DataFrame:
    """Télécharge la dette publique totale quotidienne."""
    return _fetch_daily_table(
        client, DEBT_ENDPOINT, DEBT_PUBLICATION_LAG_BUSINESS_DAYS, ("record_date",)
    )


def fetch_cash_balance(client: httpx.Client) -> pd.DataFrame:
    """Télécharge la trésorerie quotidienne.

    Piège à traiter au nettoyage : depuis avril 2022, le solde de fin de journée du compte du
    Trésor est rangé dans `open_today_bal`, et `close_today_bal` vaut « null ».
    """
    return _fetch_daily_table(
        client, CASH_ENDPOINT, CASH_PUBLICATION_LAG_BUSINESS_DAYS, ("record_date", "account_type")
    )
