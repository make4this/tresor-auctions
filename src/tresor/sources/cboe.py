"""CBOE : historique quotidien du VIX."""

from __future__ import annotations

import httpx
import pandas as pd

from tresor.http import fetch_text
from tresor.sources.raw import csv_to_raw_frame

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def fetch_vix(client: httpx.Client) -> pd.DataFrame:
    """Télécharge tout l'historique du VIX."""
    return csv_to_raw_frame(fetch_text(VIX_URL, client=client))
