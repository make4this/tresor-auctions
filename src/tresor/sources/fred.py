"""FRED : séries macro-financières, avec la date à laquelle chaque valeur a été publiée."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

from tresor.config import HISTORY_START, get_fred_api_key
from tresor.http import fetch_json
from tresor.sources.raw import RawDataError, add_business_days, records_to_raw_frame, to_iso_text

API_URL = "https://api.stlouisfed.org/fred"
# Limite lue dans le message d'erreur de l'API : 2 000 versions au plus par requête.
MAX_VINTAGES_PER_REQUEST = 2000
ALL_VINTAGES = 1  # output_type=1 : chaque valeur et la période où elle était en ligne
MISSING_VALUE = "."
OBSERVATIONS_LIMIT = 100_000

DateRule = Callable[[pd.Series], pd.Series]


def next_business_day(dates: pd.Series) -> pd.Series:
    """Reverse repo : publié le jour même, le jour ouvré suivant suffit à plus de 99,9 %."""
    return add_business_days(dates, 1)


def two_business_days(dates: pd.Series) -> pd.Series:
    """Spread Baa, inflation anticipée : +1 jour ouvré était trop optimiste, +2 suffit (mesuré)."""
    return add_business_days(dates, 2)


def friday_after(dates: pd.Series) -> pd.Series:
    """Niveaux du mercredi (rapport H.4.1) : publiés le jeudi, le vendredi en cas de jour férié."""
    return dates + pd.Timedelta(days=2)


def wednesday_of_next_week(dates: pd.Series) -> pd.Series:
    """Ancien indice dollar (rapport H.10) : publié le lundi suivant, parfois le mercredi après un
    lundi férié. Mercredi : prudent dans 99,86 % des cas mesurés (mardi : 98,91 %)."""
    return dates + pd.to_timedelta(9 - dates.dt.day_of_week, unit="D")


@dataclass(frozen=True)
class FredSeries:
    """Une série FRED et sa règle de publication, utilisée quand ALFRED n'a pas d'archive.

    rule=None : la série n'existait pas avant le début de son archive (indice dollar créé en
    février 2019). Son historique antérieur n'est donc devenu public qu'à cette date.
    """

    series_id: str
    description: str
    rule: DateRule | None
    # Délai minimal entre la date d'une valeur et sa publication : plancher appliqué même aux
    # dates ALFRED (une seule anomalie trouvée : DTWEXB du 14/05/2018, daté du jour même).
    minimum_lag_days: int = 0


SERIES = (
    FredSeries("DTWEXBGS", "indice dollar large", None, minimum_lag_days=3),
    # Ancien indice dollar (biens), publié en temps réel jusqu'à fin 2019 : il couvre 2009-2019,
    # quand l'indice actuel n'existait pas encore. Archive ALFRED à partir du 18/03/2014.
    FredSeries(
        "DTWEXB",
        "ancien indice dollar (biens), arrêté fin 2019",
        wednesday_of_next_week,
        minimum_lag_days=3,
    ),
    FredSeries("WALCL", "bilan de la Fed (mercredi)", friday_after, minimum_lag_days=1),
    FredSeries("RRPONTSYD", "reverse repo de la Fed", next_business_day),
    FredSeries("WTREGEN", "compte du Trésor à la Fed (mercredi)", friday_after, minimum_lag_days=1),
    # Remplace BAMLH0A0HYM2 (spread high yield) : FRED n'en garde qu'une fenêtre glissante de
    # 3 ans (début au 13/11/2023 le jour de la vérification).
    FredSeries("BAA10Y", "spread Baa de Moody's contre le 10 ans", two_business_days),
    FredSeries("T10YIE", "inflation anticipée à 10 ans", two_business_days),
    # Titres du Trésor gardés par la Fed pour les banques centrales étrangères (rapport H.4.1,
    # niveau du mercredi, publié le jeudi) : un recul signale moins d'achats étrangers.
    FredSeries(
        "WMTSECL1",
        "dette américaine gardée pour les banques centrales étrangères (mercredi)",
        friday_after,
        minimum_lag_days=1,
    ),
)

# Publications dont la date est fixée à l'avance (calendrier annuel du BLS) : inflation (CPI)
# et emploi (Employment Situation). FRED garde la date réelle de chaque publication passée et
# les dates prévues à venir.
RELEASES = {"cpi": 10, "jobs": 50}
RELEASE_COLUMNS = ("release", "date")

# Délai minimal par série du marché nettoyé (« fred_walcl »...), réutilisé par le contrôle
# bloquant de features.py.
MINIMUM_LAG_DAYS = {
    f"fred_{s.series_id.lower()}": s.minimum_lag_days for s in SERIES if s.minimum_lag_days
}

REQUIRED_COLUMNS = (
    "series_id",
    "date",
    "value",
    "first_release",
    "available_at",
    "available_at_source",
)


def _get(client: httpx.Client, endpoint: str, api_key: str, **params: Any) -> dict[str, Any]:
    """Appel à l'API ; la clé passe en paramètre, jamais dans l'URL ni dans les messages."""
    query = {"api_key": api_key, "file_type": "json", **params}
    return fetch_json(f"{API_URL}/{endpoint}", params=query, client=client)


def vintage_windows(vintage_dates: list[str]) -> list[tuple[str, str]]:
    """Découpe la liste des versions en fenêtres que l'API accepte."""
    chunks = [
        vintage_dates[start : start + MAX_VINTAGES_PER_REQUEST]
        for start in range(0, len(vintage_dates), MAX_VINTAGES_PER_REQUEST)
    ]
    return [(chunk[0], chunk[-1]) for chunk in chunks]


def first_release_dates(
    client: httpx.Client, series_id: str, api_key: str
) -> tuple[pd.Series, str]:
    """Date de la première publication chiffrée de chaque valeur, et date d'ouverture de l'archive.

    On lit toutes les versions et l'on garde, pour chaque date, la première où la valeur est un
    nombre : une valeur publiée vide (« . ») puis remplie plus tard prend la date du chiffre.
    """
    payload = _get(client, "series/vintagedates", api_key, series_id=series_id, limit=10_000)
    vintages = payload["vintage_dates"]
    if len(vintages) < payload["count"]:
        raise RawDataError(f"{series_id} : {len(vintages)} versions reçues sur {payload['count']}")

    rows: list[dict[str, str]] = []
    for realtime_start, realtime_end in vintage_windows(vintages):
        rows += _get(
            client,
            "series/observations",
            api_key,
            series_id=series_id,
            observation_start=HISTORY_START.isoformat(),
            realtime_start=realtime_start,
            realtime_end=realtime_end,
            output_type=ALL_VINTAGES,
            limit=OBSERVATIONS_LIMIT,
        )["observations"]

    releases = pd.DataFrame(rows, columns=["date", "value", "realtime_start"])
    numbers = releases[releases["value"] != MISSING_VALUE]
    # Dans une fenêtre, une valeur publiée plus tôt apparaît datée du début de la fenêtre : le
    # minimum sur toutes les fenêtres retrouve sa vraie date.
    first = numbers.groupby("date")["realtime_start"].min()
    # Le jour où l'archive commence, tout l'historique y entre d'un coup : ce n'est pas une vraie
    # date de publication pour les valeurs plus anciennes.
    return first[first > vintages[0]], vintages[0]


def fetch_series(client: httpx.Client, series: FredSeries, api_key: str) -> pd.DataFrame:
    """Valeurs actuelles d'une série, avec la date à laquelle chacune était connue."""
    payload = _get(
        client,
        "series/observations",
        api_key,
        series_id=series.series_id,
        observation_start=HISTORY_START.isoformat(),
        limit=OBSERVATIONS_LIMIT,
    )
    if len(payload["observations"]) < payload["count"]:
        raise RawDataError(f"{series.series_id} : historique tronqué")
    frame = records_to_raw_frame(payload["observations"])[["date", "value"]]

    published, archive_start = first_release_dates(client, series.series_id, api_key)
    first_release = frame["date"].map(published)
    known = pd.to_datetime(first_release, format="%Y-%m-%d")
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d")
    if series.rule is None:
        # Rien n'était public avant l'ouverture de l'archive ; après, on garde la date elle-même.
        by_rule = dates.clip(lower=pd.Timestamp(archive_start))
    else:
        by_rule = series.rule(dates)
    # Garde-fou : une valeur n'est jamais connue avant sa date plus le délai minimal de la série.
    floor = dates + pd.Timedelta(days=series.minimum_lag_days)
    available_at = known.fillna(by_rule).clip(lower=floor)
    return frame.assign(
        series_id=series.series_id,
        first_release=first_release.fillna(""),
        available_at=to_iso_text(available_at),
        available_at_source=known.notna().map({True: "alfred", False: "regle"}),
    )[list(REQUIRED_COLUMNS)]


def fetch_release_dates(client: httpx.Client) -> pd.DataFrame:
    """Dates de publication de l'inflation et de l'emploi depuis 2000, passées et prévues."""
    api_key = get_fred_api_key()
    frames = []
    for name, release_id in RELEASES.items():
        payload = _get(
            client,
            "release/dates",
            api_key,
            release_id=release_id,
            # Toute la période réelle : sans elle, seules les dates de l'année en cours viennent.
            realtime_start="1776-07-04",
            realtime_end="9999-12-31",
            include_release_dates_with_no_data="true",
            sort_order="asc",
            limit=10_000,
        )
        dates = [row["date"] for row in payload["release_dates"]]
        if len(dates) < payload["count"]:
            raise RawDataError(f"dates de publication {name} : liste tronquée")
        frames.append(pd.DataFrame({"release": name, "date": dates}, dtype=str))
    frame = pd.concat(frames, ignore_index=True)
    return frame[frame["date"] >= HISTORY_START.isoformat()].reset_index(drop=True)


def fetch_all(client: httpx.Client) -> pd.DataFrame:
    """Télécharge toutes les séries FRED du projet, empilées dans une seule table."""
    api_key = get_fred_api_key()
    return pd.concat([fetch_series(client, s, api_key) for s in SERIES], ignore_index=True)
