"""Treasury.gov : courbe des taux quotidienne et taux des bills."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

from tresor.config import CACHE_DIR, HISTORY_START, NEW_YORK_TZ
from tresor.http import fetch_text
from tresor.sources.raw import RawDataError, csv_to_raw_frame

URL_TEMPLATE = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
)
YIELD_CURVE = "daily_treasury_yield_curve"
BILL_RATES = "daily_treasury_bill_rates"

# Vérifié : la table des bills renvoie une réponse vide pour 2000 et 2001.
BILL_RATES_FIRST_YEAR = 2002


def _year_checker(kind: str, year: int) -> Callable[[str], None]:
    """Vérifie qu'une réponse est bien le CSV de l'année demandée (pas une page d'erreur)."""

    def check(text: str) -> None:
        if not text.startswith("Date,"):
            raise RawDataError(f"{kind} {year} : contenu inattendu, pas un CSV de taux")
        years = pd.to_datetime(csv_to_raw_frame(text)["Date"], format="%m/%d/%Y").dt.year
        if not years.eq(year).all():
            raise RawDataError(f"{kind} {year} : le fichier contient d'autres années")

    return check


def _fetch_year(
    client: httpx.Client, kind: str, year: int, current_year: int, cache_dir: Path
) -> pd.DataFrame | None:
    """Lit une année. Renvoie None si l'année en cours n'a encore aucune donnée."""
    # Une année terminée ne change plus : on la garde. L'année en cours est toujours relue.
    is_finished = year < current_year
    cache_path = cache_dir / "treasury_rates" / f"{kind}_{year}.csv" if is_finished else None
    params = {"type": kind, "field_tdr_date_value": year, "page": "", "_format": "csv"}
    url = URL_TEMPLATE.format(year=year)
    text = fetch_text(
        url, params=params, cache_path=cache_path, client=client, validate=_year_checker(kind, year)
    )
    if text.strip():
        return csv_to_raw_frame(text)
    if year == current_year:
        return None  # normal début janvier, avant la première publication de l'année
    raise RawDataError(f"{kind} : réponse vide pour l'année {year}")


def _fetch_history(
    client: httpx.Client, kind: str, first_year: int, current_year: int | None, cache_dir: Path
) -> pd.DataFrame:
    """Empile les années. Les colonnes varient : une colonne absente reste simplement vide."""
    last_year = current_year if current_year is not None else datetime.now(NEW_YORK_TZ).year
    frames = [
        _fetch_year(client, kind, year, last_year, cache_dir)
        for year in range(first_year, last_year + 1)
    ]
    return pd.concat([f for f in frames if f is not None], ignore_index=True, sort=False)


def fetch_yield_curve(
    client: httpx.Client, current_year: int | None = None, cache_dir: Path = CACHE_DIR
) -> pd.DataFrame:
    """Télécharge la courbe des taux depuis le début de l'historique."""
    return _fetch_history(client, YIELD_CURVE, HISTORY_START.year, current_year, cache_dir)


def fetch_bill_rates(
    client: httpx.Client, current_year: int | None = None, cache_dir: Path = CACHE_DIR
) -> pd.DataFrame:
    """Télécharge les taux des bills, publiés à partir de 2002."""
    return _fetch_history(client, BILL_RATES, BILL_RATES_FIRST_YEAR, current_year, cache_dir)
