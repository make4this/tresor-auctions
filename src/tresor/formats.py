"""Mise en forme des nombres : à la française pour le rapport (reports/), à l'anglaise pour
l'application en ligne."""

from __future__ import annotations

import pandas as pd

MISSING = "–"


def _number(value: float | None, decimals: int, sign: bool) -> str | None:
    """Nombre au format anglais de Python (« 1,234.5 »), ou None s'il manque. Un zéro arrondi
    ne garde pas de signe moins."""
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if round(value, decimals) == 0:
        value = 0.0
    return f"{value:{'+' if sign else ''},.{decimals}f}"


def fr(value: float | None, decimals: int = 2, sign: bool = False, suffix: str = "") -> str:
    """Nombre à la française : virgule décimale, espace insécable des milliers, vrai signe
    moins ; « – » si la valeur manque."""
    text = _number(value, decimals, sign)
    if text is None:
        return MISSING
    text = text.replace(",", " ").replace(".", ",").replace("-", "−")
    if suffix.startswith(" "):
        suffix = " " + suffix[1:]  # pas de retour à la ligne avant « % » ou « pb »
    return text + suffix


def en(value: float | None, decimals: int = 2, sign: bool = False, suffix: str = "") -> str:
    """Nombre à l'anglaise, pour l'application : point décimal, virgule des milliers, vrai
    signe moins ; « – » si la valeur manque."""
    text = _number(value, decimals, sign)
    if text is None:
        return MISSING
    text = text.replace("-", "−")
    if suffix.startswith(" "):
        suffix = " " + suffix[1:]
    return text + suffix
