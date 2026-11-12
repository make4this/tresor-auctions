"""CFTC : positions des asset managers et des leveraged funds sur les futures de Treasuries."""

from __future__ import annotations

import httpx
import numpy as np
import pandas as pd

from tresor.http import fetch_json
from tresor.sources.raw import records_to_raw_frame, to_iso_text, us_holidays

# Jeu « TFF - Futures Only » (Traders in Financial Futures, futures seuls).
DATASET_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
PAGE_SIZE = 10_000

DATE_COLUMN = "report_date_as_yyyy_mm_dd"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
CODE_COLUMN = "cftc_contract_market_code"

# On filtre par code : les noms des contrats ont changé en février 2022, pas leurs codes.
CONTRACT_CODES = {
    "042601": "2 ans",
    "044601": "5 ans",
    "043602": "10 ans",
    "043607": "Ultra 10 ans",
    "020601": "T-Bond",
    "020604": "Ultra Bond",
}

REQUIRED_COLUMNS = (
    DATE_COLUMN,
    CODE_COLUMN,
    "market_and_exchange_names",
    "open_interest_all",
    "asset_mgr_positions_long",
    "asset_mgr_positions_short",
    "lev_money_positions_long",
    "lev_money_positions_short",
    "available_at",
)

FRIDAY = 4  # lundi = 0

# Annonce officielle du 3/12/2010 : l'historique TFF depuis juin 2006 est publié ce jour-là.
TFF_HISTORY_PUBLISHED = pd.Timestamp("2010-12-03")

# Rapports publiés en retard, d'après les annonces officielles de la CFTC (page « COT Historical
# Special Announcements ») : date du rapport -> date de publication.
DELAYED_RELEASES = {
    # Fermeture d'octobre 2013 : communiqué 6745-13 (23/10/2013). Premier rapport le 25/10, deux
    # la semaine du 28/10, deux autres la semaine du 4/11 : on retient le vendredi de la semaine.
    "2013-10-01": "2013-10-25",
    "2013-10-08": "2013-11-01",
    "2013-10-15": "2013-11-01",
    "2013-10-22": "2013-11-08",
    "2013-10-29": "2013-11-08",
    # Fermeture de l'administration 2018-2019 : calendrier du communiqué 7864-19 (29/01/2019).
    "2018-12-24": "2019-02-01",
    "2018-12-31": "2019-02-05",
    "2019-01-08": "2019-02-08",
    "2019-01-15": "2019-02-12",
    "2019-01-22": "2019-02-15",
    "2019-01-29": "2019-02-19",
    "2019-02-05": "2019-02-22",
    "2019-02-12": "2019-02-26",
    "2019-02-19": "2019-03-01",
    "2019-02-26": "2019-03-05",
    # Incident informatique chez ION : annonces du 24/02 au 21/03/2023.
    "2023-01-31": "2023-02-24",
    "2023-02-07": "2023-03-03",
    "2023-02-14": "2023-03-08",
    "2023-02-21": "2023-03-10",
    "2023-02-28": "2023-03-14",
    "2023-03-07": "2023-03-16",
    "2023-03-14": "2023-03-21",
    # Fermeture de l'administration 2025 : table officielle du 9/12/2025.
    "2025-09-30": "2025-11-19",
    "2025-10-07": "2025-11-21",
    "2025-10-14": "2025-11-25",
    "2025-10-21": "2025-12-02",
    "2025-10-28": "2025-12-05",
    "2025-11-04": "2025-12-09",
    "2025-11-10": "2025-12-10",
    "2025-11-18": "2025-12-12",
    "2025-11-25": "2025-12-15",
    "2025-12-02": "2025-12-17",
    "2025-12-09": "2025-12-19",
    "2025-12-16": "2025-12-23",
}


def _days(dates: pd.Series) -> np.ndarray:
    return dates.to_numpy(dtype="datetime64[D]")


def release_dates(report_dates: pd.Series) -> pd.Series:
    """Date de publication de chaque rapport.

    Règle normale : le vendredi de la semaine du rapport. Un jour férié entre le jour du rapport
    (exclu) et ce vendredi (inclus) reporte la publication au jour ouvré suivant, comme dans le
    calendrier officiel (« Delayed release date due to a federal holiday »).
    """
    fridays = report_dates + pd.to_timedelta(FRIDAY - report_dates.dt.day_of_week, unit="D")
    holidays = us_holidays(report_dates.min(), fridays.max() + pd.Timedelta(days=10))
    holidays_in_week = np.searchsorted(holidays, _days(fridays), side="right") - np.searchsorted(
        holidays, _days(report_dates), side="right"
    )
    after_friday = np.busday_offset(
        _days(fridays) + np.timedelta64(1, "D"), 0, roll="forward", holidays=holidays
    )
    normal = pd.Series(
        np.where(holidays_in_week > 0, after_friday, _days(fridays)), index=report_dates.index
    ).astype("datetime64[ns]")

    official = pd.to_datetime(report_dates.dt.strftime("%Y-%m-%d").map(DELAYED_RELEASES))
    return official.fillna(normal).clip(lower=TFF_HISTORY_PUBLISHED)


def fetch_positions(client: httpx.Client) -> pd.DataFrame:
    """Télécharge les positions, avec la date de publication réelle de chaque rapport."""
    codes = ",".join(f"'{code}'" for code in CONTRACT_CODES)
    records: list[dict[str, str]] = []
    while True:
        params = {
            "$where": f"{CODE_COLUMN} in ({codes})",
            "$order": f"{DATE_COLUMN},{CODE_COLUMN}",
            "$limit": PAGE_SIZE,
            "$offset": len(records),
        }
        page = fetch_json(DATASET_URL, params=params, client=client)
        records.extend(page)
        if len(page) < PAGE_SIZE:
            break

    frame = records_to_raw_frame(records)
    # Le rapport est presque toujours un mardi, parfois un lundi ou un mercredi (jours fériés).
    report_dates = pd.to_datetime(frame[DATE_COLUMN], format=DATE_FORMAT)
    return frame.assign(available_at=to_iso_text(release_dates(report_dates)))
