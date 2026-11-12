"""Contrôles de qualité des données, recalculés à chaque exécution.

Trois niveaux : rouge (erreur à corriger), orange (valeur extrême récente à vérifier), vert.
"""

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
# Le calendrier ne contient que des enchères à venir : ses effectifs varient normalement.
NO_ROW_DROP_CHECK = {"treasurydirect_upcoming"}
ROW_DROP_TOLERANCE = 0.95  # une année qui perd plus de 5 % de ses lignes est suspecte

AUCTION_RANGES = {
    "btc": (1.0, 6.0),
    "dealer_share": (0.0, 1.0),
    "direct_share": (0.0, 1.0),
    "indirect_share": (0.0, 1.0),
    "auction_rate": (-1.0, 20.0),
}
# Les TIPS ont un taux réel, légitimement inférieur à -1 % en 2012 et 2020-2021 : pas de borne.
RATE_CHECKED_FAMILIES = ("nominal_coupon", "bill", "cmb")
# Les valeurs extrêmes anciennes sont connues (bid-to-cover de 10,7 en 2015) : seules les
# récentes deviennent un avertissement, pour qu'une nouvelle anomalie se voie.
RECENT_RANGE_DAYS = 365

# Séries d'enchères régulières : alerte si la dernière enchère (ou la prochaine annoncée) est
# plus vieille que deux intervalles habituels, ce qui révèle une enchère absente des deux sources.
REGULAR_FAMILIES = ("nominal_coupon", "bill")
REGULAR_LOOKBACK_DAYS = 365
REGULAR_MIN_AUCTIONS = 6
REGULAR_GAP_FACTOR = 2.0

RATE_RANGE = (-1.0, 20.0)
RATE_SERIES_PREFIXES = ("yield_", "bill_")
GAP_CHECK_START = pd.Timestamp("2009-01-01")  # début de l'échantillon des modèles
# Jours ouvrés manquants tolérés entre deux observations, et depuis la dernière publication.
MAX_GAP = {"quotidienne": 5, "hebdomadaire": 10}
MAX_SERIES_DELAY = {"quotidienne": 5, "hebdomadaire": 10}
# Séries arrêtées par leur producteur : pas d'alerte de fraîcheur (leurs trous restent contrôlés).
ENDED_SERIES = {"fred_dtwexb": "ancien indice dollar, arrêté par la Fed fin 2019"}


def status_of(alerts: Sequence[str], warnings: Sequence[str] = ()) -> str:
    """rouge : erreur à corriger ; orange : valeurs extrêmes à vérifier ; vert : rien à signaler."""
    if alerts:
        return "rouge"
    return "orange" if warnings else "vert"


def business_days_between(start: pd.Series | date, end: date) -> np.ndarray:
    """Jours ouvrés américains de start (exclu) à end (inclus)."""
    starts = np.asarray(pd.to_datetime(start), dtype="datetime64[D]") + np.timedelta64(1, "D")
    return np.busday_count(starts, np.datetime64(end) + np.timedelta64(1, "D"), holidays=HOLIDAYS)


def _rows_by_year(dates: pd.Series) -> dict[str, int]:
    counts = dates.dt.year.value_counts().sort_index()
    return {str(year): int(count) for year, count in counts.items()}


def check_raw_table(
    frame: pd.DataFrame,
    source: RawSource,
    previous: Mapping[str, Any] | None,
    today: date,
) -> dict[str, Any]:
    """Effectifs par année, fraîcheur et baisse anormale par rapport aux exécutions précédentes."""
    dates = pd.to_datetime(frame[source.date_column], format=source.date_format)
    by_year = _rows_by_year(dates)
    alerts: list[str] = []

    # La fraîcheur se mesure sur la date de publication quand on la connaît.
    reference_dates = pd.to_datetime(frame["available_at"]) if "available_at" in frame else dates
    last = reference_dates.max()
    delay = int(business_days_between(last, today))
    tolerance = MAX_DELAY_OVERRIDES.get(source.name, MAX_DELAY_BUSINESS_DAYS)
    if source.name not in NO_FRESHNESS_CHECK and delay > tolerance:
        alerts.append(f"dernière donnée du {last:%Y-%m-%d} : {delay} jours ouvrés de retard")

    # La référence garde le plus grand effectif déjà vu pour chaque année : une baisse reste
    # signalée à chaque exécution tant qu'elle n'est pas réparée.
    previous = previous or {}
    reference = dict(previous.get("rows_by_year_reference", previous.get("rows_by_year", {})))
    if source.name not in NO_ROW_DROP_CHECK:
        for year, before in reference.items():
            now = by_year.get(year, 0)
            if now < ROW_DROP_TOLERANCE * before:
                alerts.append(f"{year} : {now} lignes contre {before} au maximum déjà vu")
    for year, now in by_year.items():
        reference[year] = max(now, reference.get(year, 0))

    return {
        "rows": len(frame),
        "rows_by_year": by_year,
        "rows_by_year_reference": dict(sorted(reference.items())),
        "first_date": dates.min().strftime("%Y-%m-%d"),
        "last_date": dates.max().strftime("%Y-%m-%d"),
        "last_published": last.strftime("%Y-%m-%d"),
        "business_days_since_last": delay,
        "alerts": alerts,
        "status": status_of(alerts),
    }


def _out_of_range(values: pd.Series, low: float, high: float) -> pd.Series:
    return values.notna() & ((values < low) | (values > high))


def _stale_regular_series(auctions: pd.DataFrame, today: date) -> list[dict[str, Any]]:
    """Séries régulières dont la dernière enchère connue est anormalement ancienne."""
    since = pd.Timestamp(today) - pd.Timedelta(days=REGULAR_LOOKBACK_DAYS)
    recent = auctions[
        auctions["family"].isin(REGULAR_FAMILIES)
        & auctions["series_key"].notna()
        & (auctions["auction_date"] >= since)
    ]
    stale = []
    for key, part in recent.groupby("series_key"):
        days = part["auction_date"].sort_values()
        if len(days) < REGULAR_MIN_AUCTIONS:
            continue
        usual_gap = days.diff().median()
        if pd.Timestamp(today) - days.max() > REGULAR_GAP_FACTOR * usual_gap:
            stale.append(
                {
                    "series_key": key,
                    "last_auction": days.max().strftime("%Y-%m-%d"),
                    "usual_gap_days": int(usual_gap.days),
                }
            )
    return stale


def check_auctions(
    auctions: pd.DataFrame, today: date, recent_without_results: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """Bornes des cibles et du taux, doublons de clé, enchères manquantes ou sans résultat."""
    completed = auctions[auctions["status"] == "completed"]
    recent_since = pd.Timestamp(today) - pd.Timedelta(days=RECENT_RANGE_DAYS)
    ranges, warnings = {}, []
    for column, (low, high) in AUCTION_RANGES.items():
        checked = completed
        if column == "auction_rate":
            checked = completed[completed["family"].isin(RATE_CHECKED_FAMILIES)]
        bad = checked[_out_of_range(checked[column], low, high)]
        recent_bad = bad[bad["auction_date"] >= recent_since]
        examples = bad[["cusip", "auction_date", "family", "maturity_bucket", column]].tail(5)
        examples = examples.assign(auction_date=examples["auction_date"].dt.strftime("%Y-%m-%d"))
        ranges[column] = {
            "bounds": [low, high],
            "outside": len(bad),
            "outside_last_365_days": len(recent_bad),
            "examples": examples.astype(object).to_dict("records"),
        }
        if len(recent_bad):
            warnings.append(f"{column} : {len(recent_bad)} valeurs récentes hors de {[low, high]}")

    alerts = []
    duplicates = int(auctions.duplicated(["cusip", "auction_date"]).sum())
    if duplicates:
        alerts.append(f"{duplicates} doublons (cusip, date)")
    for row in recent_without_results:
        alerts.append(
            f"enchère du {row['auction_date']} ({row['cusip']}) sans résultat dans les deux sources"
        )
    stale = _stale_regular_series(auctions, today)
    for row in stale:
        alerts.append(
            f"{row['series_key']} : aucune enchère depuis le {row['last_auction']}"
            f" (habituellement tous les {row['usual_gap_days']} jours)"
        )
    return {
        "ranges": ranges,
        "duplicate_keys": duplicates,
        "recent_without_results": list(recent_without_results),
        "stale_regular_series": stale,
        "alerts": alerts,
        "warnings": warnings,
        "status": status_of(alerts, warnings),
    }


def _frequency(dates: pd.Series) -> str:
    """Rythme d'une suite de dates sur ses deux dernières années : quotidien ou hebdomadaire."""
    unique = pd.Series(dates.drop_duplicates().sort_values().to_numpy())
    recent = unique[unique >= unique.max() - pd.Timedelta(days=730)]
    return "quotidienne" if recent.diff().median() <= pd.Timedelta(days=3) else "hebdomadaire"


def _largest_gaps(dates: pd.Series, count: int = 3) -> list[dict[str, Any]]:
    ordered = dates.sort_values().reset_index(drop=True)
    if len(ordered) < 2:
        return []
    starts = ordered[:-1].to_numpy(dtype="datetime64[D]")
    ends = ordered[1:].to_numpy(dtype="datetime64[D]")
    # Jours ouvrés strictement entre deux observations consécutives : ce sont les jours manquants.
    gaps = pd.Series(np.busday_count(starts + 1, ends, holidays=HOLIDAYS))
    top = gaps.nlargest(count)
    return [
        {"after": str(starts[i]), "next": str(ends[i]), "missing_business_days": int(gap)}
        for i, gap in top.items()
    ]


def _check_series(name: str, part: pd.DataFrame, today: date) -> tuple[dict[str, Any], list[str]]:
    """Fraîcheur d'une série (selon son rythme de publication) et trous depuis 2009."""
    publication = _frequency(part["available_at"])
    observation = _frequency(part["date"])
    last_published = part["available_at"].max()
    delay = int(business_days_between(last_published, today))
    gaps = _largest_gaps(part.loc[part["date"] >= GAP_CHECK_START, "date"])
    alerts = []
    if name not in ENDED_SERIES and delay > MAX_SERIES_DELAY[publication]:
        alerts.append(f"{name} : dernière publication le {last_published:%Y-%m-%d}")
    if gaps and gaps[0]["missing_business_days"] > MAX_GAP[observation]:
        alerts.append(
            f"{name} : trou de {gaps[0]['missing_business_days']} jours ouvrés"
            f" après le {gaps[0]['after']}"
        )
    summary = {
        "publication": publication,
        "observation": observation,
        "last_date": part["date"].max().strftime("%Y-%m-%d"),
        "last_published": last_published.strftime("%Y-%m-%d"),
        "business_days_since_last_publication": delay,
        "largest_gaps_since_2009": gaps,
    }
    return summary, alerts


def check_market(market: pd.DataFrame, today: date) -> dict[str, Any]:
    """Règle anti-fuite, bornes des taux, doublons, fraîcheur et trous de chaque série."""
    alerts, warnings = [], []
    # Règle centrale du projet : une valeur n'est jamais connue avant sa date, ni sans date connue.
    leaks = int((market["available_at"].isna() | (market["available_at"] < market["date"])).sum())
    if leaks:
        alerts.append(f"{leaks} valeurs sans date de publication ou publiées avant leur date")
    duplicates = int(market.duplicated(["series", "date"]).sum())
    if duplicates:
        alerts.append(f"{duplicates} doublons (série, date)")

    rates = market[market["series"].str.startswith(RATE_SERIES_PREFIXES)]
    bad_rates = rates[_out_of_range(rates["value"], *RATE_RANGE)]
    recent_bad = bad_rates[bad_rates["date"] >= pd.Timestamp(today) - pd.Timedelta(days=365)]
    if len(recent_bad):
        warnings.append(f"{len(recent_bad)} taux récents hors de {list(RATE_RANGE)} %")

    series = {}
    for name, part in market.groupby("series"):
        series[name], series_alerts = _check_series(name, part, today)
        alerts += series_alerts
    return {
        "values_without_valid_publication_date": leaks,
        "duplicate_keys": duplicates,
        "rates_outside_bounds": len(bad_rates),
        "series": series,
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
        "auctions": check_auctions(auctions, today, recent_without_results),
        "market": check_market(market, today),
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
