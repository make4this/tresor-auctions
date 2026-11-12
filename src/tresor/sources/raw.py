"""Mise en table des réponses brutes : tout reste en texte jusqu'au nettoyage (phase 4)."""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


class RawDataError(ValueError):
    """La réponse d'une source est inutilisable : vide, tronquée ou de format inattendu."""


# Journées de deuil national : l'administration fédérale était fermée, rien n'a été publié.
SPECIAL_CLOSURES = ("2004-06-11", "2007-01-02", "2018-12-05", "2025-01-09")


def us_holidays(start: Any, end: Any) -> np.ndarray:
    """Jours fériés fédéraux et fermetures exceptionnelles, triés, au format numpy."""
    federal = USFederalHolidayCalendar().holidays(start=start, end=end)
    return federal.union(pd.to_datetime(list(SPECIAL_CLOSURES))).to_numpy(dtype="datetime64[D]")


def _holidays_for(dates: pd.Series) -> np.ndarray:
    return us_holidays(dates.min() - pd.Timedelta(days=10), dates.max() + pd.Timedelta(days=30))


def add_business_days(dates: pd.Series, count: int) -> pd.Series:
    """Ajoute des jours ouvrés américains, sans compter week-ends et jours fériés fédéraux."""
    shifted = np.busday_offset(
        dates.to_numpy(dtype="datetime64[D]"),
        count,
        roll="backward",  # une date tombant un week-end repart du vendredi précédent
        holidays=_holidays_for(dates),
    )
    return pd.Series(shifted, index=dates.index)


def business_day_on_or_after(dates: pd.Series) -> pd.Series:
    """La date elle-même si c'est un jour ouvré, sinon le jour ouvré suivant."""
    shifted = np.busday_offset(
        dates.to_numpy(dtype="datetime64[D]"), 0, roll="forward", holidays=_holidays_for(dates)
    )
    return pd.Series(shifted, index=dates.index)


def to_iso_text(dates: pd.Series) -> pd.Series:
    """Écrit des dates en texte AAAA-MM-JJ, comme le reste des tables brutes."""
    return dates.dt.strftime("%Y-%m-%d")


def records_to_raw_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Transforme une liste d'objets JSON en table dont toutes les colonnes sont du texte."""
    return pd.DataFrame(list(records), dtype=str)


def csv_to_raw_frame(text: str) -> pd.DataFrame:
    """Lit un CSV sans interpréter les valeurs : une case vide reste une chaîne vide."""
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
