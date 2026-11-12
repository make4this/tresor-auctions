"""Séries de marché en format long, chacune avec la date à laquelle elle était publique."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

MARKET_COLUMNS = ("source", "series", "date", "value", "available_at")

# « * » : valeur masquée par la New York Fed ; « . » : valeur absente sur FRED.
MISSING_MARKERS = ["", ".", "null", "*", "N/A"]
US_DATE_FORMAT = "%m/%d/%Y"
ISO_DATE_FORMAT = "%Y-%m-%d"

YIELD_COLUMNS = {
    "1 Mo": "yield_1m",
    "1.5 Month": "yield_6w",
    "2 Mo": "yield_2m",
    "3 Mo": "yield_3m",
    "4 Mo": "yield_4m",
    "6 Mo": "yield_6m",
    "1 Yr": "yield_1y",
    "2 Yr": "yield_2y",
    "3 Yr": "yield_3y",
    "5 Yr": "yield_5y",
    "7 Yr": "yield_7y",
    "10 Yr": "yield_10y",
    "20 Yr": "yield_20y",
    "30 Yr": "yield_30y",
}
BILL_COUPON_EQUIVALENT = re.compile(r"(\d+) WEEKS COUPON EQUIVALENT")
BILL_BANK_DISCOUNT = re.compile(r"(\d+) WEEKS BANK DISCOUNT")

# Solde du compte du Trésor à la Fed : trois présentations successives. Raccords vérifiés au
# million près (30/9/2021 -> 1/10/2021 et 15/4/2022 -> 18/4/2022). Depuis avril 2022, le solde
# de fin de journée est rangé dans open_today_bal.
TGA_ACCOUNTS = (
    ("Federal Reserve Account", "close_today_bal"),
    ("Treasury General Account (TGA)", "close_today_bal"),
    ("Treasury General Account (TGA) Closing Balance", "open_today_bal"),
)

# Positions nettes des dealers : tranches communes aux anciennes clés (jusqu'en mars 2013) et aux
# actuelles. Chaque tranche liste ses groupes de clés possibles ; un groupe se somme.
DEALER_BUCKETS = {
    "dealer_bills": (("PDPUSGTBNOP",), ("PDPOSGS-B",)),
    "dealer_coupons_le3y": (("PDPUSGCS3LNOP",), ("PDPOSGSC-L2", "PDPOSGSC-G2L3")),
    "dealer_coupons_3to6y": (("PDPUSGCS36NOP",), ("PDPOSGSC-G3L6",)),
    "dealer_coupons_6to11y": (("PDPUSGCS611NOP",), ("PDPOSGSC-G6L7", "PDPOSGSC-G7L11")),
    "dealer_coupons_gt11y": (
        ("PDPUSGCSM11NOP",),
        ("PDPOSGSC-G11",),
        ("PDPOSGSC-G11L21", "PDPOSGSC-G21"),
    ),
}

CFTC_CONTRACTS = {
    "042601": "2y",
    "044601": "5y",
    "043602": "10y",
    "043607": "ultra10y",
    "020601": "bond",
    "020604": "ultrabond",
}
CFTC_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def _numbers(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.where(~values.isin(MISSING_MARKERS)))


def _series(
    source: str, name: str, dates: pd.Series, values: pd.Series, available_at: pd.Series
) -> pd.DataFrame:
    """Une série au format long, sans ses valeurs manquantes."""
    frame = pd.DataFrame(
        {
            "source": source,
            "series": name,
            "date": dates.to_numpy(dtype="datetime64[ns]"),
            "value": values.to_numpy(dtype=float),
            "available_at": available_at.to_numpy(dtype="datetime64[ns]"),
        }
    )
    return frame.dropna(subset=["value"])


def yield_curve_long(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Courbe des taux. Taux de clôture publiés le soir même : available_at = date.

    La jointure exige available_at < date de l'enchère : une enchère du jour t n'utilise donc
    que des clôtures de t-1 ou avant.
    """
    dates = pd.to_datetime(raw["Date"], format=US_DATE_FORMAT)
    frames = [
        _series("treasury_yield_curve", name, dates, _numbers(raw[column]), dates)
        for column, name in YIELD_COLUMNS.items()
        if column in raw
    ]
    unknown = sorted(set(raw.columns) - set(YIELD_COLUMNS) - {"Date"})
    return pd.concat(frames, ignore_index=True), unknown


def bill_rates_long(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Taux des bills en équivalent obligataire, comparable au taux d'enchère des bills."""
    dates = pd.to_datetime(raw["Date"], format=US_DATE_FORMAT)
    frames, unknown = [], []
    for column in raw.columns.drop("Date"):
        match = BILL_COUPON_EQUIVALENT.fullmatch(column)
        if match:
            name = f"bill_{match.group(1)}w"
            frames.append(_series("treasury_bill_rates", name, dates, _numbers(raw[column]), dates))
        elif not BILL_BANK_DISCOUNT.fullmatch(column):
            unknown.append(column)
    return pd.concat(frames, ignore_index=True), unknown


def vix_long(raw: pd.DataFrame) -> pd.DataFrame:
    """Clôture du VIX, publique le soir même."""
    dates = pd.to_datetime(raw["DATE"], format=US_DATE_FORMAT)
    return _series("cboe_vix", "vix_close", dates, _numbers(raw["CLOSE"]), dates)


def _with_available_at(raw: pd.DataFrame, date_column: str, date_format: str) -> tuple:
    dates = pd.to_datetime(raw[date_column], format=date_format)
    return dates, pd.to_datetime(raw["available_at"], format=ISO_DATE_FORMAT)


def fred_long(raw: pd.DataFrame) -> pd.DataFrame:
    """Séries FRED ; available_at vient de la collecte (vraie date ALFRED ou règle)."""
    dates, available_at = _with_available_at(raw, "date", ISO_DATE_FORMAT)
    names = "fred_" + raw["series_id"].str.lower()
    frame = pd.DataFrame(
        {
            "source": "fred_series",
            "series": names.to_numpy(),
            "date": dates.to_numpy(dtype="datetime64[ns]"),
            "value": _numbers(raw["value"]).to_numpy(dtype=float),
            "available_at": available_at.to_numpy(dtype="datetime64[ns]"),
        }
    )
    return frame.dropna(subset=["value"])


def debt_long(raw: pd.DataFrame) -> pd.DataFrame:
    """Dette totale et dette détenue par le public, en dollars."""
    dates, available_at = _with_available_at(raw, "record_date", ISO_DATE_FORMAT)
    return pd.concat(
        [
            _series(
                "fiscaldata_debt",
                "debt_total",
                dates,
                _numbers(raw["tot_pub_debt_out_amt"]),
                available_at,
            ),
            _series(
                "fiscaldata_debt",
                "debt_public",
                dates,
                _numbers(raw["debt_held_public_amt"]),
                available_at,
            ),
        ],
        ignore_index=True,
    )


def tga_long(raw: pd.DataFrame) -> pd.DataFrame:
    """Solde de fin de journée du compte du Trésor à la Fed, en millions de dollars."""
    parts = []
    for account, column in TGA_ACCOUNTS:
        rows = raw[raw["account_type"] == account]
        dates, available_at = _with_available_at(rows, "record_date", ISO_DATE_FORMAT)
        parts.append(
            _series(
                "fiscaldata_cash_balance", "tga_close", dates, _numbers(rows[column]), available_at
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    if frame["date"].duplicated().any():
        raise ValueError("solde du Trésor : deux présentations se chevauchent sur une même date")
    return frame


def dealers_long(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Positions nettes par tranche, raccordées entre anciennes et nouvelles clés (millions $)."""
    dates = pd.to_datetime(raw["As Of Date"], format=ISO_DATE_FORMAT)
    values = _numbers(raw["Value (millions)"])
    wide = pd.DataFrame({"date": dates, "key": raw["Time Series"], "value": values}).pivot(
        index="date", columns="key", values="value"
    )
    available_at = (
        pd.DataFrame({"date": dates, "available_at": pd.to_datetime(raw["available_at"])})
        .groupby("date")["available_at"]
        .max()
    )
    buckets, overlaps = {}, 0
    for name, groups in DEALER_BUCKETS.items():
        sums = [
            wide[list(group)].sum(axis=1, min_count=len(group))
            for group in groups
            if set(group) <= set(wide.columns)
        ]
        overlaps += int((pd.concat(sums, axis=1).notna().sum(axis=1) > 1).sum())
        combined = sums[0]
        for other in sums[1:]:
            combined = combined.combine_first(other)
        buckets[name] = combined
    buckets["dealer_total"] = pd.DataFrame(buckets).sum(axis=1, min_count=len(DEALER_BUCKETS))
    frames = [
        _series(
            "nyfed_dealer_positions",
            name,
            pd.Series(series.index),
            series.reset_index(drop=True),
            pd.Series(available_at.reindex(series.index).to_numpy()),
        )
        for name, series in buckets.items()
    ]
    return pd.concat(frames, ignore_index=True), overlaps


def cftc_long(raw: pd.DataFrame) -> pd.DataFrame:
    """Positions nettes (long - short) des asset managers et des leveraged funds, par contrat."""
    dates, available_at = _with_available_at(raw, "report_date_as_yyyy_mm_dd", CFTC_DATE_FORMAT)
    number = {
        column: _numbers(raw[column])
        for column in raw.columns
        if "positions" in column or column == "open_interest_all"
    }
    measures = {
        "am_net": number["asset_mgr_positions_long"] - number["asset_mgr_positions_short"],
        "lev_net": number["lev_money_positions_long"] - number["lev_money_positions_short"],
        "oi": number["open_interest_all"],
    }
    frames = []
    for code, label in CFTC_CONTRACTS.items():
        rows = raw["cftc_contract_market_code"].eq(code)
        for measure, values in measures.items():
            frames.append(
                _series(
                    "cftc_positions",
                    f"cftc_{label}_{measure}",
                    dates[rows],
                    values[rows],
                    available_at[rows],
                )
            )
    return pd.concat(frames, ignore_index=True)


def build_market(raw_tables: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble toutes les séries de marché et rend compte de ce qui a été fait."""
    curve, unknown_curve = yield_curve_long(raw_tables["treasury_yield_curve"])
    bills, unknown_bills = bill_rates_long(raw_tables["treasury_bill_rates"])
    dealers, dealer_overlaps = dealers_long(raw_tables["nyfed_dealer_positions"])
    market = pd.concat(
        [
            curve,
            bills,
            vix_long(raw_tables["cboe_vix"]),
            fred_long(raw_tables["fred_series"]),
            debt_long(raw_tables["fiscaldata_debt"]),
            tga_long(raw_tables["fiscaldata_cash_balance"]),
            dealers,
            cftc_long(raw_tables["cftc_positions"]),
        ],
        ignore_index=True,
    ).sort_values(["series", "date"], kind="stable", ignore_index=True)

    summary = market.groupby("series").agg(
        source=("source", "first"),
        rows=("value", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    report = {
        "rows": len(market),
        "series": {
            name: {
                "source": row.source,
                "rows": int(row.rows),
                "first_date": row.first_date.strftime(ISO_DATE_FORMAT),
                "last_date": row.last_date.strftime(ISO_DATE_FORMAT),
            }
            for name, row in summary.iterrows()
        },
        "unknown_columns": {
            "treasury_yield_curve": unknown_curve,
            "treasury_bill_rates": unknown_bills,
        },
        "dealer_key_overlaps": dealer_overlaps,
        "available_at_before_date": int((market["available_at"] < market["date"]).sum()),
    }
    return market[list(MARKET_COLUMNS)], report


def known_values(series_frame: pd.DataFrame) -> pd.DataFrame:
    """À chaque date de publication, garde l'observation la plus récente connue à ce moment.

    Une ancienne observation publiée en retard (panne, révision) ne remplace pas une plus récente.
    """
    ordered = series_frame.sort_values(["available_at", "date"], kind="stable")
    return ordered[ordered["date"].eq(ordered["date"].cummax())]


def point_in_time(dates: pd.Series, series_frame: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque date t, la dernière valeur publiée strictement avant t (jointure sans fuite)."""
    known = known_values(series_frame)[["available_at", "date", "value"]].rename(
        columns={"date": "observed_on"}
    )
    known["available_at"] = known["available_at"].astype("datetime64[ns]")
    left = pd.DataFrame(
        {"t": dates.to_numpy(dtype="datetime64[ns]"), "position": np.arange(len(dates))}
    )
    merged = pd.merge_asof(
        left.sort_values("t", kind="stable"),
        known,
        left_on="t",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.sort_values("position").set_index(dates.index)
    return merged[["observed_on", "value", "available_at"]]
