"""Calendrier des réunions de la Fed (FOMC) : `python -m tresor.sources.fed_calendar`.

Les réunions sont fixées et publiées environ un an à l'avance : on peut donc savoir, la veille
d'une enchère, dans combien de jours tombe la prochaine décision de la Fed. Le calendrier est
lu une fois sur le site de la Fed et gardé dans `data/static/fomc_meetings.csv` (une page par
année depuis 2000) ; on relance la commande une fois par an, quand la Fed publie l'année suivante.

Seules les réunions prévues au calendrier comptent. Les réunions d'urgence (« unscheduled »),
les conférences téléphoniques et les votes par écrit (« notation vote ») ne pouvaient pas être
connus à l'avance : les garder ferait entrer le futur dans les variables. Une réunion prévue
puis annulée (mars 2020) reste, puisqu'elle figurait au calendrier la veille. Une réunion de
stratégie sans décision de politique monétaire (« Monetary Policy Strategy Meeting », octobre
2027) n'est pas une décision et ne compte pas.

Limite connue, sans effet : le 16 mai 2012, la Fed a décalé d'un jour trois réunions de 2012
(1er août, 13 septembre, 12 décembre). Avant cette date, la prochaine réunion était toujours
celle du 20 juin au plus tard : aucune enchère n'a utilisé les dates d'avant le décalage.
"""

from __future__ import annotations

import html
import logging
import re
import sys
import uuid
from pathlib import Path

import httpx
import pandas as pd

from tresor.config import HISTORY_START, STATIC_DIR
from tresor.http import build_client, fetch_text
from tresor.sources.raw import RawDataError

logger = logging.getLogger(__name__)

CURRENT_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
CALENDAR_FILE = STATIC_DIR / "fomc_meetings.csv"
COLUMNS = ("decision_date", "first_day", "label")

# « January 30-31 », « April/May 30-1 », « July 31-August 1 », « March 17-18* ».
WHEN = re.compile(
    r"^(?P<first_month>[A-Za-z]+)(?:/(?P<second_month>[A-Za-z]+))?\s+(?P<first_day>\d{1,2})"
    r"(?:-(?:(?P<end_month>[A-Za-z]+)\s+)?(?P<last_day>\d{1,2}))?"
)
NOT_SCHEDULED = ("unscheduled", "conference call", "notation vote")
STRATEGY = "monetary policy strategy meeting"  # réunion sans décision
EXPECTED_PER_YEAR = 8  # réunions prévues par an depuis 1981, réunions de stratégie comprises


def _day(month: str, number: str, year: int) -> pd.Timestamp:
    # « April/May » ou « Apr/May » selon les années : les trois premières lettres suffisent.
    return pd.to_datetime(f"{month[:3].title()} {number} {year}", format="%b %d %Y")


def _meeting(text: str, year: int) -> dict[str, object] | None:
    """Une réunion prévue au calendrier, ou None (réunion d'urgence, appel, vote écrit).

    `decision` vaut False pour une réunion de stratégie, qui ne donne pas de décision.
    """
    label = " ".join(html.unescape(text).split())
    if any(word in label.lower() for word in NOT_SCHEDULED):
        return None
    match = WHEN.match(label)
    if match is None:
        return None
    end_month = match["end_month"] or match["second_month"] or match["first_month"]
    first = _day(match["first_month"], match["first_day"], year)
    last = _day(end_month, match["last_day"] or match["first_day"], year)
    return {
        "decision_date": last,
        "first_day": first,
        "label": label,
        "decision": STRATEGY not in label.lower(),
    }


def parse_historical(page: str, year: int) -> pd.DataFrame:
    """Page d'archives d'une année : un titre <h5> par réunion, « March 21 Meeting - 2000 »."""
    meetings = []
    for title in re.findall(r"<h5[^>]*>([^<]*)</h5>", page):
        if "Meeting" not in title:
            continue  # « January 3 Conference Call - 2001 », « March 19 (notation vote) - 2020 »
        meeting = _meeting(title.split(" - ")[0], year)
        if meeting is not None:
            meetings.append(meeting)
    return pd.DataFrame(meetings, columns=[*COLUMNS, "decision"])


def parse_current(page: str) -> pd.DataFrame:
    """Page des cinq dernières années et des suivantes : une section « 2026 FOMC Meetings »,
    puis un bloc par réunion (mois, jours et, pour une réunion de stratégie, sa mention)."""
    headers = list(re.finditer(r"(\d{4}) FOMC Meetings", page))
    meetings = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(page)
        section = page[header.end() : end]
        blocks = re.split(r'<div class="[^"]*\bfomc-meeting"', section)[1:]
        for block in blocks:
            month = re.search(r"fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>", block)
            days = re.search(r"fomc-meeting__date[^>]*>([^<]+)<", block)
            if month is None or days is None:
                raise RawDataError(f"calendrier de la Fed {header[1]} : réunion illisible")
            text = f"{month[1].strip()} {days[1].strip()}"
            if STRATEGY in html.unescape(block).lower():
                text += " (Monetary Policy Strategy Meeting)"
            meeting = _meeting(text, int(header[1]))
            if meeting is not None:
                meetings.append(meeting)
    return pd.DataFrame(meetings, columns=[*COLUMNS, "decision"])


def build_calendar(client: httpx.Client) -> pd.DataFrame:
    """Toutes les réunions prévues depuis 2000 : archives, puis page des années récentes."""
    current = parse_current(fetch_text(CURRENT_URL, client=client))
    if current.empty:
        raise RawDataError("calendrier de la Fed : aucune réunion lue sur la page actuelle")
    first_current_year = int(current["decision_date"].dt.year.min())
    parts = [
        parse_historical(fetch_text(HISTORICAL_URL.format(year=year), client=client), year)
        for year in range(HISTORY_START.year, first_current_year)
    ]
    calendar = pd.concat([*parts, current], ignore_index=True)
    calendar = calendar.drop_duplicates("decision_date").sort_values("decision_date")
    # Deux entrées à moins de 4 jours (15 et 16 septembre 2003) : une seule réunion, la décision
    # tombe le second jour.
    following = calendar["decision_date"].shift(-1)
    calendar = calendar[~((following - calendar["decision_date"]).dt.days < 4)]
    by_year = calendar["decision_date"].dt.year.value_counts()
    # La Fed tient 8 réunions prévues par an depuis 1981 : un autre nombre trahit une page mal lue.
    odd = by_year[by_year != EXPECTED_PER_YEAR]
    if not odd.empty:
        raise RawDataError(f"calendrier de la Fed : nombre de réunions inattendu {odd.to_dict()}")
    # Seules les réunions qui donnent une décision entrent dans le calendrier.
    calendar = calendar[calendar["decision"].astype(bool)]
    return calendar[list(COLUMNS)].reset_index(drop=True)


def read_calendar(path: Path = CALENDAR_FILE) -> pd.DataFrame:
    """Le calendrier enregistré, avec des dates (vide si le fichier n'existe pas)."""
    if not path.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    calendar = pd.read_csv(path, dtype=str)
    for column in ("decision_date", "first_day"):
        calendar[column] = pd.to_datetime(calendar[column], format="%Y-%m-%d")
    return calendar


def write_calendar(calendar: pd.DataFrame, path: Path = CALENDAR_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = calendar.assign(
        decision_date=calendar["decision_date"].dt.strftime("%Y-%m-%d"),
        first_day=calendar["first_day"].dt.strftime("%Y-%m-%d"),
    )
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
    out.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with build_client() as client:
        calendar = build_calendar(client)
    write_calendar(calendar)
    logger.info(
        "Calendrier de la Fed : %d réunions prévues, du %s au %s, écrites dans %s",
        len(calendar),
        calendar["decision_date"].min().date(),
        calendar["decision_date"].max().date(),
        CALENDAR_FILE,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
