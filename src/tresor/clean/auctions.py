"""Nettoyage des adjudications : fusion des sources, typage, classification et exclusions."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from tresor.config import BILL_MATURITIES_WEEKS, COUPON_MATURITIES_YEARS, MIN_OFFERING_AMOUNT_USD

# Façons d'écrire « manquant » selon les sources : "null" (Fiscal Data), "" (TreasuryDirect).
MISSING_MARKERS = ["null", "", "."]

DATE_COLUMNS = ("auction_date", "announcemt_date", "issue_date", "maturity_date")
TEXT_COLUMNS = ("cusip", "security_type", "security_term", "original_security_term")
FLAG_COLUMNS = (
    "reopening",
    "inflation_index_security",
    "floating_rate",
    "cash_management_bill_cmb",
)
BIDDER_COLUMNS = ("primary_dealer_accepted", "direct_bidder_accepted", "indirect_bidder_accepted")
NUMERIC_COLUMNS = (
    "offering_amt",
    "high_yield",
    "avg_med_yield",
    "low_yield",
    "high_discnt_rate",
    "low_discnt_rate",
    "avg_med_discnt_rate",
    "high_investment_rate",
    "bid_to_cover_ratio",
    "comp_accepted",
    "comp_tendered",
    "total_accepted",
    "total_tendered",
    *BIDDER_COLUMNS,
    "soma_accepted",
)
FISCALDATA_COLUMNS = (*TEXT_COLUMNS, *DATE_COLUMNS, *FLAG_COLUMNS, *NUMERIC_COLUMNS)

# Correspondances vérifiées en phase 2 sur 99 enchères communes (voir journal_llm.md).
TREASURYDIRECT_TO_FISCALDATA = {
    "cusip": "cusip",
    "announcementDate": "announcemt_date",
    "issueDate": "issue_date",
    "maturityDate": "maturity_date",
    "securityType": "security_type",
    "securityTerm": "security_term",
    "originalSecurityTerm": "original_security_term",
    "reopening": "reopening",
    "offeringAmount": "offering_amt",
    "highYield": "high_yield",
    "lowYield": "low_yield",
    "medianYield": "avg_med_yield",
    "highRate": "high_discnt_rate",
    "lowRate": "low_discnt_rate",
    "medianRate": "avg_med_discnt_rate",
    "investmentRate": "high_investment_rate",
    "bidToCoverRatio": "bid_to_cover_ratio",
    "subtotalCompetitiveParRecognizedInsideOfOfferingAmount": "comp_tendered",
    # Vérifié : égal à la somme des trois catégories d'enchérisseurs (137 lignes sur 137) et à
    # comp_accepted de Fiscal Data (103 enchères communes) ; lu tel quel, il permet de contrôler
    # que les trois parts somment bien à 1.
    "subtotalCompetitiveParAwardedInsideOfOfferingAmount": "comp_accepted",
    "grandTotalParAwarded": "total_accepted",
    "grandTotalParRecognized": "total_tendered",
    "primaryDealersCompetitiveParAwarded": "primary_dealer_accepted",
    "directBiddersCompetitiveParAwarded": "direct_bidder_accepted",
    "indirectBiddersCompetitiveParAwarded": "indirect_bidder_accepted",
    "somaNoncompetitiveParAwarded": "soma_accepted",
    "inflationProtected": "inflation_index_security",
    "floatingRate": "floating_rate",
    "cashManagementBill": "cash_management_bill_cmb",
}

# En cas de doublon (cusip, date) : d'abord la ligne avec résultats, puis la source la plus fraîche.
SOURCE_PRIORITY = {"treasurydirect": 2, "treasurydirect_upcoming": 1, "fiscaldata": 0}

TERM_PATTERN = re.compile(r"(?:(\d+)-Year)?\s*(?:(\d+)-Month)?\s*(?:(\d+)-Week)?\s*(?:(\d+)-Day)?")
DAYS_PER_YEAR = 365.25
BILL_FAMILIES = ("bill", "cmb")
# Au-delà de cet écart relatif, une maturité n'est rattachée à aucune maturité standard : c'est
# le cas de CMB de 1 à 35 jours, qui ne doivent pas passer pour des bills à 4 semaines.
FAR_FROM_STANDARD = 0.2
RECENT_DAYS = 30  # une enchère passée depuis moins longtemps sans résultat déclenche une alerte
# Au-delà de ce délai sans enchère, une série recommence : les réouvertures hors calendrier
# d'octobre 2008 (anciens 10 ans à 6-7 ans de l'échéance) ne servent pas d'historique au 7 ans
# relancé en février 2009, ni le 20 ans de 1986 à celui de 2020.
RUN_GAP_DAYS = 120

OUTPUT_COLUMNS = (
    "cusip",
    "auction_date",
    "announcemt_date",
    "issue_date",
    "maturity_date",
    "security_type",
    "security_term",
    "original_security_term",
    "reopening",
    "family",
    "term_years",
    "maturity_bucket",
    "bucket_years",
    "series_key",
    "series_run",
    "offering_amt",
    "auction_rate",
    "high_yield",
    "high_discnt_rate",
    "high_investment_rate",
    "btc",
    "dealer_share",
    "direct_share",
    "indirect_share",
    "comp_accepted",
    "comp_tendered",
    "total_accepted",
    "total_tendered",
    *BIDDER_COLUMNS,
    "soma_accepted",
    "status",
    "source",
)


def type_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Convertit le texte brut en dates, nombres et booléens ; un format inconnu lève une erreur."""
    typed = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        values = frame[column].where(~frame[column].isin(MISSING_MARKERS))
        if column in DATE_COLUMNS:
            typed[column] = pd.to_datetime(values, format="%Y-%m-%d")
        elif column in FLAG_COLUMNS:
            unexpected = set(values.dropna()) - {"Yes", "No"}
            if unexpected:
                raise ValueError(f"{column} : valeurs inattendues {sorted(unexpected)[:5]}")
            typed[column] = values.map({"Yes": True, "No": False}).astype("boolean")
        elif column in NUMERIC_COLUMNS:
            typed[column] = pd.to_numeric(values)
        else:
            typed[column] = values
    return typed


def harmonize_fiscaldata(raw: pd.DataFrame) -> pd.DataFrame:
    """Garde les colonnes utiles de Fiscal Data et les type."""
    return type_columns(raw[list(FISCALDATA_COLUMNS)]).assign(source="fiscaldata")


def harmonize_treasurydirect(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """Renomme les champs TreasuryDirect avec les noms Fiscal Data, puis les type."""
    renamed = raw.rename(columns=TREASURYDIRECT_TO_FISCALDATA)
    renamed["auction_date"] = raw["competitiveAuctionCloseDate"]
    for column in DATE_COLUMNS:
        renamed[column] = renamed[column].str[:10]  # "2026-11-16T00:00:00" -> "2026-11-16"
    return type_columns(renamed[list(FISCALDATA_COLUMNS)]).assign(source=source)


def combine_sources(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Empile les sources et ne garde qu'une ligne par (cusip, date d'enchère)."""
    combined = pd.concat(frames, ignore_index=True)
    ranked = combined.assign(
        _has_results=combined["bid_to_cover_ratio"].notna(),
        _priority=combined["source"].map(SOURCE_PRIORITY),
    ).sort_values(["_has_results", "_priority"], ascending=False, kind="stable")
    unique = ranked.drop_duplicates(["cusip", "auction_date"]).drop(
        columns=["_has_results", "_priority"]
    )
    unique = unique.sort_values(["auction_date", "cusip"], kind="stable").reset_index(drop=True)
    return unique, len(combined) - len(unique)


def classify_family(frame: pd.DataFrame) -> pd.Series:
    """Famille de titre. TIPS et FRN sont classés « Note » ou « Bond » : on lit leurs drapeaux."""

    def flag(column: str) -> np.ndarray:
        return frame[column].fillna(False).to_numpy(dtype=bool)

    security_type = frame["security_type"].fillna("")
    conditions = [
        flag("floating_rate"),
        flag("inflation_index_security"),
        flag("cash_management_bill_cmb"),
        security_type.eq("Bill").to_numpy(dtype=bool),
        security_type.isin(["Note", "Bond"]).to_numpy(dtype=bool),
    ]
    choices = ["frn", "tips", "cmb", "bill", "nominal_coupon"]
    return pd.Series(np.select(conditions, choices, default="unknown"), index=frame.index)


def term_in_years(term: str) -> float:
    """Durée résiduelle d'un libellé comme « 9-Year 10-Month », « 13-Week » ou « 42-Day »."""
    if not isinstance(term, str):
        raise ValueError(f"durée illisible : {term!r}")
    match = TERM_PATTERN.fullmatch(term.strip())
    if match is None or not any(match.groups()):
        raise ValueError(f"durée illisible : {term!r}")
    years, months, weeks, days = (int(part) if part else 0 for part in match.groups())
    return years + months / 12 + (weeks * 7 + days) / DAYS_PER_YEAR


def nearest(value: float, choices: tuple[int, ...]) -> int:
    """Valeur standard la plus proche ; en cas d'égalité, la plus courte."""
    return min(choices, key=lambda choice: (abs(value - choice), choice))


def maturity_bucket(term_years: float, family: str) -> tuple[str | None, float, float]:
    """Maturité standard la plus proche : libellé, durée en années et écart relatif.

    Si l'écart dépasse FAR_FROM_STANDARD, aucune maturité standard n'est attribuée (None).
    """
    if family in BILL_FAMILIES:
        weeks = term_years * DAYS_PER_YEAR / 7
        standard = nearest(weeks, BILL_MATURITIES_WEEKS)
        label, years, gap = f"{standard}W", standard * 7 / DAYS_PER_YEAR, abs(weeks - standard)
    else:
        standard = nearest(term_years, COUPON_MATURITIES_YEARS)
        label, years, gap = f"{standard}Y", float(standard), abs(term_years - standard)
    relative_gap = gap / standard
    if relative_gap > FAR_FROM_STANDARD:
        return None, float("nan"), relative_gap
    return label, years, relative_gap


def add_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Série régulière de chaque enchère : même famille, même maturité standard, sans longue pause.

    series_key : famille et maturité (« nominal_coupon_10Y »). series_run : la même clé, numérotée
    à chaque reprise après plus de RUN_GAP_DAYS jours sans enchère (« nominal_coupon_7Y#2 »).
    """
    key = frame["family"] + "_" + frame["maturity_bucket"]
    ordered = frame.assign(series_key=key).sort_values(
        ["series_key", "auction_date"], kind="stable"
    )
    has_key = ordered["series_key"].notna()
    keyed = ordered[has_key]
    gaps = keyed.groupby("series_key")["auction_date"].diff()
    new_run = gaps.isna() | (gaps > pd.Timedelta(days=RUN_GAP_DAYS))
    run_number = new_run.astype(int).groupby(keyed["series_key"]).cumsum()
    ordered["series_run"] = pd.Series(pd.NA, index=ordered.index, dtype="string")
    ordered.loc[has_key, "series_run"] = keyed["series_key"] + "#" + run_number.astype(str)
    return ordered.sort_index()


def auction_status(frame: pd.DataFrame, today: date) -> pd.Series:
    """completed : résultats publiés ; upcoming : aujourd'hui ou après ; sinon missing_result."""
    has_results = frame["bid_to_cover_ratio"].notna().to_numpy(dtype=bool)
    not_past = (frame["auction_date"] >= pd.Timestamp(today)).to_numpy(dtype=bool)
    statuses = np.select([has_results, not_past], ["completed", "upcoming"], "missing_result")
    return pd.Series(statuses, index=frame.index)


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Cibles et taux d'enchère. Bills : taux équivalent obligataire, comparable au marché."""
    competitive = frame["comp_accepted"].where(frame["comp_accepted"] > 0)
    is_bill = frame["family"].isin(BILL_FAMILIES)
    return frame.assign(
        btc=frame["bid_to_cover_ratio"],
        dealer_share=frame["primary_dealer_accepted"] / competitive,
        direct_share=frame["direct_bidder_accepted"] / competitive,
        indirect_share=frame["indirect_bidder_accepted"] / competitive,
        auction_rate=frame["high_investment_rate"].where(is_bill, frame["high_yield"]),
    )


def exclusion_reasons(frame: pd.DataFrame) -> pd.Series:
    """Raison d'exclusion de chaque ligne (vide si la ligne est gardée)."""
    rules = [
        ("type_inconnu", frame["family"].eq("unknown")),
        ("petite_emission_hors_calendrier", frame["offering_amt"] < MIN_OFFERING_AMOUNT_USD),
        ("passee_sans_resultat", frame["status"].eq("missing_result")),
        # Une date future présente seulement dans Fiscal Data n'est pas confirmée par le calendrier.
        (
            "future_hors_calendrier",
            frame["status"].eq("upcoming") & frame["source"].eq("fiscaldata"),
        ),
    ]
    reasons = pd.Series(pd.NA, index=frame.index, dtype="string")
    for reason, mask in rules:
        reasons = reasons.mask(reasons.isna() & mask.fillna(False), reason)
    return reasons


def _examples(frame: pd.DataFrame, limit: int = 50) -> list[dict[str, Any]]:
    columns = ["cusip", "auction_date", "security_type", "security_term", "offering_amt"]
    sample = frame[columns].head(limit).copy()
    sample["auction_date"] = sample["auction_date"].dt.strftime("%Y-%m-%d")
    return sample.astype(object).where(sample.notna(), None).to_dict("records")


def _series_restarts(auctions: pd.DataFrame, since: str) -> dict[str, str]:
    """Début de chaque série (coupons classiques et bills) qui (re)commence après une date."""
    modeled = auctions[auctions["family"].isin(["nominal_coupon", "bill"])]
    starts = modeled.groupby("series_run")["auction_date"].min()
    recent = starts[starts >= pd.Timestamp(since)].sort_values()
    return {run: day.strftime("%Y-%m-%d") for run, day in recent.items()}


def families_by_year(auctions: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'enchères par année et par famille."""
    table = pd.crosstab(auctions["auction_date"].dt.year, auctions["family"])
    table.index.name = "annee"
    return table


def clean_auctions(
    fiscal_raw: pd.DataFrame,
    recent_raw: pd.DataFrame,
    upcoming_raw: pd.DataFrame,
    today: date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Table propre des adjudications, et compte rendu de chaque étape."""
    sources = [
        harmonize_fiscaldata(fiscal_raw),
        harmonize_treasurydirect(recent_raw, "treasurydirect"),
        harmonize_treasurydirect(upcoming_raw, "treasurydirect_upcoming"),
    ]
    combined, duplicates = combine_sources(sources)
    combined["family"] = classify_family(combined)
    combined["term_years"] = combined["security_term"].map(term_in_years)
    buckets = [
        maturity_bucket(t, f)
        for t, f in zip(combined["term_years"], combined["family"], strict=True)
    ]
    combined["maturity_bucket"] = pd.Series(
        [label for label, _, _ in buckets], index=combined.index, dtype="string"
    )
    combined["bucket_years"] = [years for _, years, _ in buckets]
    combined["bucket_gap"] = [gap for _, _, gap in buckets]
    combined["status"] = auction_status(combined, today)
    combined = add_targets(combined)

    reasons = exclusion_reasons(combined)
    # Les séries se calculent sur les enchères gardées : une ligne exclue ne coupe ni ne relie rien.
    kept = add_series(combined[reasons.isna()])
    excluded = combined[reasons.notna()].assign(reason=reasons)

    # Une enchère récente sans résultat dans aucune des deux sources mérite une alerte.
    recent_missing = excluded[
        excluded["reason"].eq("passee_sans_resultat")
        & (excluded["auction_date"] >= pd.Timestamp(today) - pd.Timedelta(days=RECENT_DAYS))
    ]

    shares = kept[["dealer_share", "direct_share", "indirect_share"]]
    share_sum = shares.sum(axis=1, min_count=3)
    far = kept[kept["bucket_gap"] > FAR_FROM_STANDARD]

    report: dict[str, Any] = {
        "rows_in": {frame["source"].iloc[0]: len(frame) for frame in sources if len(frame)},
        "duplicates_removed": duplicates,
        "rows_after_merge": len(combined),
        "excluded": {
            reason: {"rows": len(part), "examples": _examples(part)}
            for reason, part in excluded.groupby("reason")
        },
        "recent_without_results": _examples(recent_missing),
        "rows_out": len(kept),
        "rows_out_by_source": kept["source"].value_counts().to_dict(),
        "rows_out_by_status": kept["status"].value_counts().to_dict(),
        "rows_out_by_family": kept["family"].value_counts().to_dict(),
        "share_sum_checked": int(share_sum.notna().sum()),
        "share_sum_outside_0_99_1_01": int(((share_sum < 0.99) | (share_sum > 1.01)).sum()),
        "far_from_standard_maturity_by_family": far["family"].value_counts().to_dict(),
        "series_restarts_since_2009": _series_restarts(kept, since="2009-01-01"),
    }
    return kept[list(OUTPUT_COLUMNS)].reset_index(drop=True), report
