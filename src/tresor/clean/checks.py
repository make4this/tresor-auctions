"""Contrôles de qualité des données, recalculés à chaque exécution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from tresor.sources.collect import RawSource
from tresor.sources.raw import us_holidays

# Même calendrier que la collecte : jours fériés fédéraux et journées de deuil national.
HOLIDAYS = us_holidays("1990-01-01", "2035-12-31")

MAX_DELAY_BUSINESS_DAYS = 5
# Retard connu et compensé par TreasuryDirect : sans tolérance, l'alerte serait permanente.
MAX_DELAY_OVERRIDES = {"fiscaldata_auctions": 30}
NO_FRESHNESS_CHECK = {"treasurydirect_upcoming"}  # ce sont des dates futures
# Fenêtres glissantes (180 jours, calendrier) : leurs effectifs par année varient normalement.
NO_ROW_DROP_CHECK = {"treasurydirect_recent", "treasurydirect_upcoming"}
ROW_DROP_TOLERANCE = 0.95  # une année passée qui perd plus de 5 % de ses lignes est suspecte

AUCTION_RANGES = {
    "btc": (1.0, 6.0),
    "dealer_share": (0.0, 1.0),
    "direct_share": (0.0, 1.0),
    "indirect_share": (0.0, 1.0),
    "auction_rate": (-1.0, 20.0),
}
# Les TIPS ont un taux réel, légitimement inférieur à -1 % en 2012 et 2020-2021 : pas de borne.
RATE_CHECKED_FAMILIES = ("nominal_coupon", "bill", "cmb")
RATE_RANGE = (-1.0, 20.0)
RATE_SERIES_PREFIXES = ("yield_", "bill_")
GAP_SERIES = ("yield_2y", "yield_10y", "bill_13w", "vix_close")
MAX_GAP_BUSINESS_DAYS = 5


def business_days_between(start: pd.Series | date, end: date) -> np.ndarray:
    """Jours ouvrés américains de start (exclu) à end (inclus)."""
    starts = np.asarray(pd.to_datetime(start), dtype="datetime64[D]") + np.timedelta64(1, "D")
    return np.busday_count(starts, np.datetime64(end) + np.timedelta64(1, "D"), holidays=HOLIDAYS)


def status_of(alerts: Sequence[str], warnings: Sequence[str] = ()) -> str:
    """rouge : erreur à corriger ; orange : valeurs extrêmes à vérifier ; vert : rien à signaler."""
    if alerts:
        return "rouge"
    return "orange" if warnings else "vert"


def _rows_by_year(dates: pd.Series) -> dict[str, int]:
    counts = dates.dt.year.value_counts().sort_index()
    return {str(year): int(count) for year, count in counts.items()}


def check_raw_table(
    frame: pd.DataFrame,
    source: RawSource,
    previous: Mapping[str, Any] | None,
    today: date,
) -> dict[str, Any]:
    """Effectifs par année, fraîcheur et baisse anormale par rapport à l'exécution précédente."""
    dates = pd.to_datetime(frame[source.date_column], format=source.date_format)
    by_year = _rows_by_year(dates)
    alerts: list[str] = []

    # La fraîcheur se mesure sur la date de publication quand on la connaît.
    reference = pd.to_datetime(frame["available_at"]) if "available_at" in frame else dates
    last = reference.max()
    delay = int(business_days_between(last, today))
    tolerance = MAX_DELAY_OVERRIDES.get(source.name, MAX_DELAY_BUSINESS_DAYS)
    if source.name not in NO_FRESHNESS_CHECK and delay > tolerance:
        alerts.append(f"dernière donnée du {last:%Y-%m-%d} : {delay} jours ouvrés de retard")

    if previous and source.name not in NO_ROW_DROP_CHECK:
        for year, before in previous.get("rows_by_year", {}).items():
            now = by_year.get(year, 0)
            if int(year) < today.year and now < ROW_DROP_TOLERANCE * before:
                alerts.append(f"{year} : {now} lignes contre {before} à l'exécution précédente")

    return {
        "rows": len(frame),
        "rows_by_year": by_year,
        "first_date": dates.min().strftime("%Y-%m-%d"),
        "last_date": dates.max().strftime("%Y-%m-%d"),
        "last_published": last.strftime("%Y-%m-%d"),
        "business_days_since_last": delay,
        "alerts": alerts,
        "status": "rouge" if alerts else "vert",
    }


def _out_of_range(values: pd.Series, low: float, high: float) -> pd.Series:
    return values.notna() & ((values < low) | (values > high))


def check_auctions(
    auctions: pd.DataFrame, recent_without_results: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """Bornes des cibles et du taux, doublons de clé, enchères récentes restées sans résultat.

    Une valeur hors bornes est un avertissement (orange) : sur ces données, ce sont de vraies
    valeurs extrêmes, comme un bid-to-cover de 10,7 sur un bill à 4 semaines en 2009-2015.
    """
    completed = auctions[auctions["status"] == "completed"]
    ranges = {}
    for column, (low, high) in AUCTION_RANGES.items():
        checked = completed
        if column == "auction_rate":
            checked = completed[completed["family"].isin(RATE_CHECKED_FAMILIES)]
        bad = checked[_out_of_range(checked[column], low, high)]
        examples = (
            bad[["cusip", "auction_date", "family", "maturity_bucket", column]].head(5).copy()
        )
        examples["auction_date"] = examples["auction_date"].dt.strftime("%Y-%m-%d")
        ranges[column] = {
            "bounds": [low, high],
            "outside": len(bad),
            "examples": examples.astype(object).to_dict("records"),
        }
    duplicates = int(auctions.duplicated(["cusip", "auction_date"]).sum())
    warnings = [
        f"{column} : {r['outside']} valeurs hors de {r['bounds']}"
        for column, r in ranges.items()
        if r["outside"]
    ]
    alerts = []
    if duplicates:
        alerts.append(f"{duplicates} doublons (cusip, date)")
    for row in recent_without_results:
        alerts.append(
            f"enchère du {row['auction_date']} ({row['cusip']}) sans résultat dans les deux sources"
        )
    return {
        "ranges": ranges,
        "duplicate_keys": duplicates,
        "recent_without_results": list(recent_without_results),
        "alerts": alerts,
        "warnings": warnings,
        "status": status_of(alerts, warnings),
    }


def _largest_gaps(dates: pd.Series, count: int = 3) -> list[dict[str, Any]]:
    ordered = dates.sort_values().reset_index(drop=True)
    starts = ordered[:-1].to_numpy(dtype="datetime64[D]")
    ends = ordered[1:].to_numpy(dtype="datetime64[D]")
    # Jours ouvrés strictement entre deux observations consécutives : ce sont les jours manquants.
    gaps = pd.Series(np.busday_count(starts + 1, ends, holidays=HOLIDAYS))
    top = gaps.nlargest(count)
    return [
        {"after": str(starts[i]), "next": str(ends[i]), "missing_business_days": int(gap)}
        for i, gap in top.items()
    ]


def check_market(market: pd.DataFrame) -> dict[str, Any]:
    """Bornes des taux, doublons (série, date) et trous dans les séries quotidiennes clés."""
    rates = market[market["series"].str.startswith(RATE_SERIES_PREFIXES)]
    bad_rates = rates[_out_of_range(rates["value"], *RATE_RANGE)]
    duplicates = int(market.duplicated(["series", "date"]).sum())
    gaps = {
        name: _largest_gaps(market.loc[market["series"] == name, "date"])
        for name in GAP_SERIES
        if (market["series"] == name).any()
    }
    alerts, warnings = [], []
    if len(bad_rates):
        warnings.append(f"{len(bad_rates)} taux hors de [{RATE_RANGE[0]}, {RATE_RANGE[1]}] %")
    if duplicates:
        alerts.append(f"{duplicates} doublons (série, date)")
    for name, top in gaps.items():
        if top and top[0]["missing_business_days"] > MAX_GAP_BUSINESS_DAYS:
            alerts.append(
                f"{name} : trou de {top[0]['missing_business_days']} jours ouvrés"
                f" après le {top[0]['after']}"
            )
    return {
        "rates_outside_bounds": len(bad_rates),
        "duplicate_keys": duplicates,
        "largest_gaps": gaps,
        "alerts": alerts,
        "warnings": warnings,
        "status": status_of(alerts, warnings),
    }


def run_checks(
    raw_tables: Mapping[str, pd.DataFrame],
    sources: Sequence[RawSource],
    auctions: pd.DataFrame,
    market: pd.DataFrame,
    previous: Mapping[str, Any] | None,
    today: date,
    recent_without_results: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Rassemble tous les contrôles dans un seul rapport."""
    previous_sources = (previous or {}).get("sources", {})
    report_sources = {
        source.name: check_raw_table(
            raw_tables[source.name], source, previous_sources.get(source.name), today
        )
        for source in sources
        if source.name in raw_tables
    }
    sections = {
        "auctions": check_auctions(auctions, recent_without_results),
        "market": check_market(market),
    }
    alerts = [
        f"{name} : {alert}" for name, part in report_sources.items() for alert in part["alerts"]
    ]
    alerts += [f"{name} : {alert}" for name, part in sections.items() for alert in part["alerts"]]
    warnings = [f"{name} : {w}" for name, part in sections.items() for w in part["warnings"]]
    return {
        "today_new_york": today.isoformat(),
        "sources": report_sources,
        **sections,
        "alerts": alerts,
        "warnings": warnings,
        "status": status_of(alerts, warnings),
    }
