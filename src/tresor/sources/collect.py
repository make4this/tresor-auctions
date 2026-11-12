"""Collecte de toutes les sources : téléchargement, contrôle minimal, sauvegarde en Parquet."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from tresor.config import RAW_DIR
from tresor.http import FetchError, build_client
from tresor.sources import cboe, cftc, fiscaldata, fred, nyfed, treasury_rates, treasurydirect
from tresor.sources.raw import RawDataError

logger = logging.getLogger(__name__)

PREVIEW_ROWS = 3
TREASURY_DIRECT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
US_DATE_FORMAT = "%m/%d/%Y"


@dataclass(frozen=True)
class RawSource:
    """Décrit une table brute : comment la télécharger et comment la contrôler."""

    name: str  # nom du fichier Parquet
    fetch: Callable[[httpx.Client], pd.DataFrame]
    date_column: str
    date_format: str
    required_columns: tuple[str, ...]
    preview_columns: tuple[str, ...]
    # Clé de cumul : si renseignée, chaque collecte s'ajoute à la précédente au lieu de la
    # remplacer (la version la plus récente d'une ligne gagne).
    accumulate_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class RawSummary:
    """Ce qui a été collecté pour une table, à afficher et à contrôler."""

    name: str
    rows: int
    first_date: date
    last_date: date
    preview: pd.DataFrame


RAW_SOURCES: tuple[RawSource, ...] = (
    RawSource(
        name="fiscaldata_auctions",
        fetch=fiscaldata.fetch_auctions,
        date_column="auction_date",
        date_format="%Y-%m-%d",
        required_columns=fiscaldata.AUCTION_COLUMNS,
        preview_columns=(
            "cusip",
            "auction_date",
            "security_type",
            "security_term",
            "offering_amt",
            "bid_to_cover_ratio",
        ),
    ),
    RawSource(
        name="treasurydirect_recent",
        fetch=treasurydirect.fetch_recent,
        date_column="competitiveAuctionCloseDate",
        date_format=TREASURY_DIRECT_DATE_FORMAT,
        required_columns=treasurydirect.RECENT_COLUMNS,
        preview_columns=(
            "cusip",
            "competitiveAuctionCloseDate",
            "securityType",
            "securityTerm",
            "offeringAmount",
            "bidToCoverRatio",
        ),
        # L'API oublie les titres échus : sans cumul, les bills courts disparaîtraient.
        accumulate_on=treasurydirect.KEY_COLUMNS,
    ),
    RawSource(
        name="treasurydirect_upcoming",
        fetch=treasurydirect.fetch_upcoming,
        date_column="competitiveAuctionCloseDate",
        date_format=TREASURY_DIRECT_DATE_FORMAT,
        required_columns=treasurydirect.UPCOMING_COLUMNS,
        preview_columns=(
            "cusip",
            "competitiveAuctionCloseDate",
            "securityType",
            "securityTerm",
            "offeringAmount",
        ),
    ),
    RawSource(
        name="treasury_yield_curve",
        fetch=treasury_rates.fetch_yield_curve,
        date_column="Date",
        date_format=US_DATE_FORMAT,
        required_columns=("Date", "2 Yr", "5 Yr", "10 Yr"),
        preview_columns=("Date", "3 Mo", "2 Yr", "10 Yr", "30 Yr"),
    ),
    RawSource(
        name="treasury_bill_rates",
        fetch=treasury_rates.fetch_bill_rates,
        date_column="Date",
        date_format=US_DATE_FORMAT,
        required_columns=("Date", "13 WEEKS COUPON EQUIVALENT", "26 WEEKS COUPON EQUIVALENT"),
        preview_columns=("Date", "13 WEEKS BANK DISCOUNT", "13 WEEKS COUPON EQUIVALENT"),
    ),
    RawSource(
        name="cboe_vix",
        fetch=cboe.fetch_vix,
        date_column="DATE",
        date_format=US_DATE_FORMAT,
        required_columns=("DATE", "OPEN", "HIGH", "LOW", "CLOSE"),
        preview_columns=("DATE", "OPEN", "HIGH", "LOW", "CLOSE"),
    ),
    RawSource(
        name="fiscaldata_debt",
        fetch=fiscaldata.fetch_debt,
        date_column="record_date",
        date_format="%Y-%m-%d",
        required_columns=fiscaldata.DEBT_COLUMNS,
        preview_columns=("record_date", "tot_pub_debt_out_amt", "available_at"),
    ),
    RawSource(
        name="fiscaldata_cash_balance",
        fetch=fiscaldata.fetch_cash_balance,
        date_column="record_date",
        date_format="%Y-%m-%d",
        required_columns=fiscaldata.CASH_COLUMNS,
        preview_columns=fiscaldata.CASH_COLUMNS,
    ),
    RawSource(
        name="nyfed_dealer_positions",
        fetch=nyfed.fetch_dealer_positions,
        date_column=nyfed.DATE_COLUMN,
        date_format="%Y-%m-%d",
        required_columns=nyfed.REQUIRED_COLUMNS,
        preview_columns=nyfed.REQUIRED_COLUMNS,
    ),
    RawSource(
        name="cftc_positions",
        fetch=cftc.fetch_positions,
        date_column=cftc.DATE_COLUMN,
        date_format=cftc.DATE_FORMAT,
        required_columns=cftc.REQUIRED_COLUMNS,
        preview_columns=(
            cftc.DATE_COLUMN,
            cftc.CODE_COLUMN,
            "asset_mgr_positions_long",
            "lev_money_positions_short",
            "available_at",
        ),
    ),
    RawSource(
        name="fred_series",
        fetch=fred.fetch_all,
        date_column="date",
        date_format="%Y-%m-%d",
        required_columns=fred.REQUIRED_COLUMNS,
        preview_columns=("series_id", "date", "value", "available_at", "available_at_source"),
    ),
)


def check_raw(frame: pd.DataFrame, source: RawSource) -> None:
    """Refuse une réponse vide ou dont le format a changé, avant d'écraser l'ancien fichier."""
    if frame.empty:
        raise RawDataError(f"{source.name} : table vide")
    missing_columns = sorted(set(source.required_columns) - set(frame.columns))
    if missing_columns:
        raise RawDataError(f"{source.name} : colonnes absentes : {missing_columns}")


def summarize(frame: pd.DataFrame, source: RawSource) -> RawSummary:
    """Calcule la plage de dates ; une date illisible ou manquante déclenche une erreur."""
    dates = pd.to_datetime(frame[source.date_column], format=source.date_format)
    if dates.isna().any():
        raise RawDataError(f"{source.name} : {int(dates.isna().sum())} dates manquantes")
    latest_rows = dates.sort_values().index[-PREVIEW_ROWS:]
    return RawSummary(
        name=source.name,
        rows=len(frame),
        first_date=dates.min().date(),
        last_date=dates.max().date(),
        preview=frame.loc[latest_rows, list(source.preview_columns)],
    )


def save_raw(frame: pd.DataFrame, name: str, raw_dir: Path) -> Path:
    """Écrit le Parquet en passant par un fichier temporaire, pour ne jamais le laisser à moitié."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{name}.parquet"
    temporary_path = path.with_name(f"{path.name}.part")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)
    return path


def accumulate(previous: pd.DataFrame, new: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    """Ajoute les nouvelles lignes aux anciennes ; pour une même clé, la plus récente gagne."""
    combined = pd.concat([previous, new], ignore_index=True)
    return combined.drop_duplicates(list(keys), keep="last", ignore_index=True)


def _collect_one(source: RawSource, client: httpx.Client, raw_dir: Path) -> RawSummary:
    """Télécharge, contrôle et sauvegarde une source."""
    frame = source.fetch(client)
    check_raw(frame, source)
    previous_path = raw_dir / f"{source.name}.parquet"
    if source.accumulate_on and previous_path.exists():
        frame = accumulate(pd.read_parquet(previous_path), frame, source.accumulate_on)
    summary = summarize(frame, source)
    save_raw(frame, source.name, raw_dir)
    logger.info(
        "%s : %d lignes, du %s au %s\n%s",
        summary.name,
        summary.rows,
        summary.first_date,
        summary.last_date,
        summary.preview.to_string(index=False),
    )
    return summary


def collect_all(
    sources: Sequence[RawSource] = RAW_SOURCES,
    client: httpx.Client | None = None,
    raw_dir: Path = RAW_DIR,
) -> list[RawSummary]:
    """Collecte chaque source. Une panne n'empêche pas les autres et garde l'ancien fichier."""
    own_client = client if client is not None else build_client()
    summaries: list[RawSummary] = []
    failed_sources: list[str] = []
    try:
        for source in sources:
            try:
                summaries.append(_collect_one(source, own_client, raw_dir))
            # Toute erreur (réseau, clé FRED absente, format imprévu) reste limitée à sa source.
            except Exception as error:
                kind = type(error).__name__
                logger.error("%s : échec de la collecte (%s) : %s", source.name, kind, error)
                failed_sources.append(source.name)
    finally:
        if client is None:
            own_client.close()
    if failed_sources:
        raise FetchError(f"Sources en échec : {', '.join(failed_sources)}")
    return summaries
