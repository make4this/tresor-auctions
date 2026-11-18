"""Taux à 10 ans allemand (Bundesbank) et japonais (ministère des Finances du Japon).

Si ces taux montent, la dette américaine devient moins attirante en comparaison pour les
investisseurs étrangers. Les deux sources sont publiques, gratuites et sans clé.
"""

from __future__ import annotations

import io
import re

import httpx
import pandas as pd

from tresor.config import HISTORY_START
from tresor.http import fetch_text
from tresor.sources.raw import RawDataError

# Bundesbank : rendement des emprunts fédéraux à 10 ans de durée résiduelle (courbe de Svensson),
# donnée quotidienne. Le CSV commence par quelques lignes de description, puis date,valeur.
BUND_URL = (
    "https://api.statistiken.bundesbank.de/rest/data/BBSIS/"
    "D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A"
)
# Ministère des Finances du Japon : tout l'historique jusqu'au mois dernier, puis le mois en cours.
JGB_HISTORY_URL = (
    "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"
)
JGB_CURRENT_URL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv"

COLUMNS = ("date", "yield_10y")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_bund(text: str) -> pd.DataFrame:
    """Garde les lignes « AAAA-MM-JJ,valeur » ; « . » (week-end, jour férié) reste tel quel."""
    rows = []
    for line in text.lstrip("﻿").splitlines():
        cells = [cell.strip().strip('"') for cell in line.split(",")]
        if len(cells) >= 2 and ISO_DATE.match(cells[0]):
            rows.append((cells[0], cells[1]))
    frame = pd.DataFrame(rows, columns=list(COLUMNS), dtype=str)
    if frame.empty:
        raise RawDataError("Bundesbank : aucune ligne de données dans la réponse")
    return frame[frame["date"] >= HISTORY_START.isoformat()].reset_index(drop=True)


def parse_jgb(text: str) -> pd.DataFrame:
    """Lit un CSV du ministère : une ligne de titre, puis l'en-tête (Date, 1Y, ..., 10Y, ...).

    Les dates sont écrites « 2026/11/2 » ; on les remet en AAAA-MM-JJ. « - » veut dire absent.
    """
    lines = text.lstrip("﻿").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("Date,")), None)
    if start is None:
        raise RawDataError("ministère des Finances du Japon : en-tête « Date, ... » introuvable")
    table = pd.read_csv(io.StringIO("\n".join(lines[start:])), dtype=str, keep_default_na=False)
    if "10Y" not in table.columns:
        raise RawDataError("ministère des Finances du Japon : colonne 10Y absente")
    dates = pd.to_datetime(table["Date"], format="%Y/%m/%d", errors="coerce")
    frame = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"), "yield_10y": table["10Y"]})
    frame = frame.dropna(subset=["date"])
    return frame[frame["date"] >= HISTORY_START.isoformat()].reset_index(drop=True)


def fetch_bund(client: httpx.Client) -> pd.DataFrame:
    """Tout l'historique quotidien du taux allemand à 10 ans depuis 2000."""
    params = {"format": "csv", "lang": "en", "startPeriod": HISTORY_START.isoformat()}
    return parse_bund(fetch_text(BUND_URL, params=params, client=client))


def fetch_jgb(client: httpx.Client) -> pd.DataFrame:
    """Historique et mois en cours du taux japonais à 10 ans ; le fichier du mois l'emporte."""
    history = parse_jgb(fetch_text(JGB_HISTORY_URL, client=client))
    current = parse_jgb(fetch_text(JGB_CURRENT_URL, client=client))
    frame = pd.concat([history, current], ignore_index=True)
    return frame.drop_duplicates("date", keep="last").sort_values("date", ignore_index=True)
