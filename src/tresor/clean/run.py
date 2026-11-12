"""Étape --clean : lit les tables brutes, nettoie, contrôle et écrit les comptes rendus."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tresor.clean.auctions import clean_auctions, families_by_year
from tresor.clean.checks import run_checks
from tresor.clean.market import build_market
from tresor.config import CLEAN_DIR, NEW_YORK_TZ, OUTPUT_DIR, RAW_DIR
from tresor.sources.collect import RAW_SOURCES

logger = logging.getLogger(__name__)


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.part")
    temporary_path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary_path.replace(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.part")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def _read_previous(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_cleaning(
    raw_dir: Path = RAW_DIR,
    clean_dir: Path = CLEAN_DIR,
    output_dir: Path = OUTPUT_DIR,
    today: date | None = None,
) -> dict[str, Any]:
    """Nettoie tout et renvoie le rapport de qualité."""
    today = today or datetime.now(NEW_YORK_TZ).date()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = {
        source.name: pd.read_parquet(raw_dir / f"{source.name}.parquet") for source in RAW_SOURCES
    }

    auctions, auctions_report = clean_auctions(
        raw["fiscaldata_auctions"],
        raw["treasurydirect_recent"],
        raw["treasurydirect_upcoming"],
        today,
    )
    market, market_report = build_market(raw)
    _write_parquet(auctions, clean_dir / "auctions.parquet")
    _write_parquet(market, clean_dir / "market.parquet")

    by_year = families_by_year(auctions)
    by_year.to_csv(output_dir / "auctions_by_family_year.csv")
    _write_json(
        output_dir / "cleaning_report.json",
        {"today_new_york": today.isoformat(), "auctions": auctions_report, "market": market_report},
    )

    quality_path = output_dir / "data_quality.json"
    quality = run_checks(
        raw,
        RAW_SOURCES,
        auctions,
        market,
        _read_previous(quality_path),
        today,
        auctions_report["recent_without_results"],
    )
    _write_json(quality_path, quality)

    logger.info(
        "Enchères gardées : %d ; séries de marché : %d", len(auctions), market["series"].nunique()
    )
    logger.info("Enchères par année et par famille :\n%s", by_year.to_string())
    for alert in quality["alerts"]:
        logger.warning("Qualité : %s", alert)
    for warning in quality["warnings"]:
        logger.info("Qualité, à vérifier : %s", warning)
    return quality
