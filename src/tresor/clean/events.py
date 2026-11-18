"""Calendrier des grandes annonces : décisions de la Fed, inflation (CPI) et emploi.

Ces dates sont fixées à l'avance. Pour rester point-in-time, chaque date reçoit une date de
connaissance prudente : le 1er décembre de l'année précédente. La Fed publie ses réunions vers
l'été de l'année d'avant et le BLS son calendrier annuel à l'automne ; au 1er décembre, les deux
sont publics.

Limite assumée : une publication déplacée en cours d'année (fermeture de l'administration
américaine d'octobre 2013 et de l'automne 2025) garde sa date réelle, connue quelques jours
seulement à l'avance. Les réunions d'urgence de la Fed, elles, sont exclues du calendrier.
"""

from __future__ import annotations

import pandas as pd

EVENT_COLUMNS = ("event", "date", "known_at")
EVENTS = ("fomc", "cpi", "jobs")


def known_from(dates: pd.Series) -> pd.Series:
    """Le 1er décembre de l'année précédente."""
    return pd.to_datetime((dates.dt.year - 1).astype(str) + "-12-01", format="%Y-%m-%d")


def build_events(release_dates: pd.DataFrame | None, fomc: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par annonce : son type, sa date et la date à partir de laquelle on la connaît."""
    parts = [pd.DataFrame({"event": "fomc", "date": pd.to_datetime(fomc["decision_date"])})]
    if release_dates is not None and not release_dates.empty:
        parts.append(
            pd.DataFrame(
                {
                    "event": release_dates["release"].to_numpy(),
                    "date": pd.to_datetime(release_dates["date"], format="%Y-%m-%d").to_numpy(),
                }
            )
        )
    events = pd.concat(parts, ignore_index=True)
    events = events[events["event"].isin(EVENTS)].drop_duplicates(["event", "date"])
    events = events.assign(known_at=known_from(events["date"]))
    return events.sort_values(["event", "date"], ignore_index=True)[list(EVENT_COLUMNS)]
