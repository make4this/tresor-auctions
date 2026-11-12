"""New York Fed : positions nettes hebdomadaires des primary dealers en Treasuries."""

from __future__ import annotations

import httpx
import pandas as pd

from tresor.http import fetch_text
from tresor.sources.raw import (
    RawDataError,
    business_day_on_or_after,
    csv_to_raw_frame,
    to_iso_text,
)

BASE_URL = "https://markets.newyorkfed.org/api/pd/get"

DATE_COLUMN = "As Of Date"
SERIES_COLUMN = "Time Series"
VALUE_COLUMN = "Value (millions)"

# Clés en vigueur depuis le 3 avril 2013. Au-delà de 11 ans, une seule tranche jusqu'en 2021
# (G11), puis deux à partir de janvier 2022 (G11L21 et G21).
CURRENT_SERIES = (
    "PDPOSGS-B",  # bills
    "PDPOSGSC-L2",  # coupons, jusqu'à 2 ans
    "PDPOSGSC-G2L3",  # 2 à 3 ans
    "PDPOSGSC-G3L6",  # 3 à 6 ans
    "PDPOSGSC-G6L7",  # 6 à 7 ans
    "PDPOSGSC-G7L11",  # 7 à 11 ans
    "PDPOSGSC-G11",  # plus de 11 ans, jusqu'en 2021
    "PDPOSGSC-G11L21",  # 11 à 21 ans, depuis 2022
    "PDPOSGSC-G21",  # plus de 21 ans, depuis 2022
    "PDPOSGST-TOT",  # total hors TIPS
)

# Clés en vigueur jusqu'au 27 mars 2013 : tranches plus larges, raccordables aux actuelles.
LEGACY_SERIES = (
    "PDPUSGTBNOP",  # bills
    "PDPUSGCS3LNOP",  # coupons, jusqu'à 3 ans
    "PDPUSGCS36NOP",  # 3 à 6 ans
    "PDPUSGCS611NOP",  # 6 à 11 ans
    "PDPUSGCSM11NOP",  # plus de 11 ans
    "PDPUSGTNOP",  # total hors TIPS
)

# Page officielle : « Data are updated on Thursdays ». Observé le lundi 9/11/2026 : dernier
# relevé disponible, celui du mercredi 28/10. Le relevé du mercredi sort donc le jeudi suivant.
PUBLICATION_LAG_DAYS = 8

REQUIRED_COLUMNS = (DATE_COLUMN, SERIES_COLUMN, VALUE_COLUMN, "available_at")


def _fetch_series(client: httpx.Client, series: str) -> pd.DataFrame:
    """Télécharge une série ; une série vide signale un changement côté New York Fed."""
    frame = csv_to_raw_frame(fetch_text(f"{BASE_URL}/{series}.csv", client=client))
    if frame.empty:
        raise RawDataError(f"positions des dealers : série vide : {series}")
    return frame


def fetch_dealer_positions(client: httpx.Client) -> pd.DataFrame:
    """Télécharge les positions, en valeurs brutes (« * » = valeur masquée par la Fed)."""
    # Une requête par série : l'API répond 422 aux appels groupés dès 6 clés (vérifié).
    frames = [_fetch_series(client, key) for key in (*LEGACY_SERIES, *CURRENT_SERIES)]
    frame = pd.concat(frames, ignore_index=True)

    as_of_dates = pd.to_datetime(frame[DATE_COLUMN], format="%Y-%m-%d")
    thursdays = as_of_dates + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    # Jeudi férié (Thanksgiving par exemple) : on suppose la publication au jour ouvré suivant.
    return frame.assign(available_at=to_iso_text(business_day_on_or_after(thursdays)))
