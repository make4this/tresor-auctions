"""Calendrier des grandes annonces : décisions de la Fed, inflation (CPI) et emploi.

Principe : on utilise le calendrier tel qu'il était publié à l'avance. La Fed publie ses
réunions vers l'été de l'année d'avant et le BLS son calendrier annuel à l'automne : chaque date
reçoit donc une date de connaissance prudente, le 1er décembre de l'année précédente.

Nettoyage des dates FRED de l'inflation et de l'emploi :

- une publication déplacée ou annulée par une fermeture de l'administration (2013, 2025, 2026)
  reprend sa date prévue à l'origine (`data/static/release_reschedules.csv`, sources BLS) : la
  nouvelle date n'était pas connue au 1er décembre ;
- inflation : seules comptent les dates où l'indice non corrigé (CPIAUCNS) a reçu une nouvelle
  version, plus les dates à venir ; la révision des coefficients saisonniers de février, par
  exemple, n'est pas une publication ;
- emploi : une seule date par mois, la première (les autres sont des révisions ou des annonces).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tresor.config import STATIC_DIR
from tresor.sources.fred import CPI_VINTAGE

EVENT_COLUMNS = ("event", "date", "known_at")
EVENTS = ("fomc", "cpi", "jobs")
RESCHEDULES_FILE = STATIC_DIR / "release_reschedules.csv"


def known_from(dates: pd.Series) -> pd.Series:
    """Le 1er décembre de l'année précédente."""
    return pd.to_datetime((dates.dt.year - 1).astype(str) + "-12-01", format="%Y-%m-%d")


def read_reschedules(path: Path = RESCHEDULES_FILE) -> pd.DataFrame:
    """Publications déplacées (date réelle -> date prévue) ou annulées (date réelle vide)."""
    if not path.exists():
        return pd.DataFrame(columns=["release", "actual_date", "scheduled_date"])
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ("actual_date", "scheduled_date"):
        table[column] = pd.to_datetime(table[column].replace("", None), format="%Y-%m-%d")
    return table


def scheduled_releases(release_dates: pd.DataFrame, reschedules: pd.DataFrame) -> pd.DataFrame:
    """Dates de l'inflation et de l'emploi comme les donnait le calendrier publié à l'avance."""
    dates = release_dates.assign(date=pd.to_datetime(release_dates["date"], format="%Y-%m-%d"))
    vintages = set(dates.loc[dates["release"] == CPI_VINTAGE, "date"])
    releases = dates[dates["release"].isin(["cpi", "jobs"])]
    if vintages:  # inflation : une vraie publication, ou une date encore à venir
        last = max(vintages)
        keep = (releases["release"] != "cpi") | releases["date"].isin(vintages)
        releases = releases[keep | (releases["date"] > last)]
    # Une seule publication par mois : la première (2000-09 pour l'inflation, révisions et
    # annonces pour l'emploi).
    month = releases["date"].dt.to_period("M")
    releases = releases.sort_values("date")
    releases = releases[~releases.assign(month=month).duplicated(["release", "month"])]
    # Dates déplacées : la date prévue remplace la date réelle ; annulées : la date prévue revient.
    moved = reschedules.dropna(subset=["actual_date"])
    swap = {(r.release, r.actual_date): r.scheduled_date for r in moved.itertuples()}
    releases = releases.assign(
        date=[
            swap.get((r, d), d) for r, d in zip(releases["release"], releases["date"], strict=True)
        ]
    )
    canceled = reschedules[reschedules["actual_date"].isna()]
    extra = pd.DataFrame({"release": canceled["release"], "date": canceled["scheduled_date"]})
    return pd.concat([releases[["release", "date"]], extra], ignore_index=True)


def build_events(
    release_dates: pd.DataFrame | None,
    fomc: pd.DataFrame,
    reschedules: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Une ligne par annonce : son type, sa date et la date à partir de laquelle on la connaît."""
    parts = [pd.DataFrame({"event": "fomc", "date": pd.to_datetime(fomc["decision_date"])})]
    if release_dates is not None and not release_dates.empty:
        reschedules = read_reschedules() if reschedules is None else reschedules
        releases = scheduled_releases(release_dates, reschedules)
        parts.append(
            pd.DataFrame({"event": releases["release"].to_numpy(), "date": releases["date"]})
        )
    events = pd.concat(parts, ignore_index=True)
    events = events[events["event"].isin(EVENTS)].drop_duplicates(["event", "date"])
    events = events.assign(known_at=known_from(events["date"]))
    return events.sort_values(["event", "date"], ignore_index=True)[list(EVENT_COLUMNS)]
