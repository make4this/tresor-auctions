"""Variables explicatives point-in-time (section 7 du plan) : seulement ce qui est connu avant t.

Règle : pour une enchère du jour t, une variable de marché ou d'historique n'utilise que des
informations publiées strictement avant t. Les caractéristiques de l'enchère elle-même (montant,
maturité, réouverture) sont connues à son annonce, toujours antérieure ou égale à t.
"""

from __future__ import annotations

import logging
import operator
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from tresor.clean.market import point_in_time
from tresor.config import CLEAN_DIR, MODEL_SAMPLE_START, OUTPUT_DIR, WALK_FORWARD_START
from tresor.sources.fred import MINIMUM_LAG_DAYS
from tresor.sources.raw import add_business_days

logger = logging.getLogger(__name__)

MODELED_FAMILIES = ("nominal_coupon", "bill")
TEST_START = pd.Timestamp(WALK_FORWARD_START)
TARGETS = ("btc", "dealer_share", "indirect_share", "tail_bp")
HISTORY_LAGS = (1, 2, 3)

REL_BTC_WINDOW, REL_BTC_MIN_PERIODS = 12, 6  # bid-to-cover relatif : sur 12 enchères passées
RECENT_DEMAND_AUCTIONS, RECENT_DEMAND_MIN_PERIODS = 5, 3
REFUNDING_MONTHS = (2, 5, 8, 11)  # refinancement trimestriel du Trésor
QUARTER_END_DAYS = 10

# Âge maximal d'une valeur de marché : au-delà, elle est jugée périmée (série interrompue).
DAILY_MAX_AGE = pd.Timedelta(days=10)
WEEKLY_MAX_AGE = pd.Timedelta(days=21)  # séries publiées une fois par semaine
# « fred_dtwexb » couvre l'ancien indice dollar (DTWEXB) et l'actuel (DTWEXBGS).
WEEKLY_BASES = ("dealer_", "cftc_", "fred_walcl", "fred_wtregen", "fred_dtwexb", "fred_wmtsecl1")
ZSCORE_WEEKS = 52  # écart normalisé sur un an pour les positions, qui ont une tendance de fond

EPOCH = pd.Timestamp("1970-01-01")
# Une valeur publiée plus de 30 jours après sa date (complétée des années plus tard, comme
# l'ancien indice dollar de 2000-2008 en 2017 ou le reverse repo de 2020 en 2025) manquait en
# temps réel : elle n'entre pas dans les calculs sur fenêtre (variations, volatilité, écarts).
LATE_FILL = pd.Timedelta(days=30)

RATE_SERIES = {
    "2Y": "yield_2y",
    "3Y": "yield_3y",
    "5Y": "yield_5y",
    "7Y": "yield_7y",
    "10Y": "yield_10y",
    "20Y": "yield_20y",
    "30Y": "yield_30y",
    # Bills : taux « coupon equivalent », sur la même base que le taux d'enchère retenu.
    "4W": "bill_4w",
    "6W": "bill_6w",
    "8W": "bill_8w",
    "13W": "bill_13w",
    "17W": "bill_17w",
    "26W": "bill_26w",
    "52W": "bill_52w",
}
BILL_BUCKETS = ("4W", "6W", "8W", "13W", "17W", "26W", "52W")
DEALER_SERIES = {
    "2Y": "dealer_coupons_le3y",
    "3Y": "dealer_coupons_le3y",
    "5Y": "dealer_coupons_3to6y",
    "7Y": "dealer_coupons_6to11y",
    "10Y": "dealer_coupons_6to11y",
    "20Y": "dealer_coupons_gt11y",
    "30Y": "dealer_coupons_gt11y",
    **{bucket: "dealer_bills" for bucket in BILL_BUCKETS},
}
# Contrat future le plus proche en maturité. Le T-Bond ne livre que des titres de 15 à 25 ans
# depuis 2011 : le 30 ans prend l'Ultra Bond. Le 3 ans et les bills prennent le 2 ans, faute de
# contrat équivalent (choix arbitraire, documenté).
CFTC_CONTRACTS = {
    "2Y": "2y",
    "3Y": "2y",
    "5Y": "5y",
    "7Y": "10y",
    "10Y": "10y",
    "20Y": "bond",
    "30Y": "ultrabond",
    **{bucket: "2y" for bucket in BILL_BUCKETS},
}

# Variable -> (groupe, séries dérivées communes à toutes les enchères). Si plusieurs séries sont
# données, la première connue à la date de l'enchère est utilisée. Les niveaux qui ont une
# tendance de fond (bilan de la Fed, solde du Trésor, reverse repo, positions) sont pris en
# variation ou en écart normalisé, pour rester comparables d'une période à l'autre.
FIXED_MARKET_FEATURES = {
    "slope_10y_2y": ("market", "slope_10y_2y"),
    "rate_3m": ("market", "yield_3m"),
    "vix": ("market", "vix_close"),
    "vix_chg_5d": ("market", "vix_close__chg5"),
    # Indice actuel (depuis février 2019), sinon l'ancien, publié en temps réel jusqu'à fin 2019.
    "dollar_chg_20d": ("market", ("fred_dtwexbgs__pct20", "fred_dtwexb__pct20")),
    "credit_spread_baa": ("market", "fred_baa10y"),
    "breakeven_10y": ("market", "fred_t10yie"),
    "debt_chg_30d": ("liquidity", "debt_total__pct21"),
    "tga_chg_4w_bn": ("liquidity", "tga_bn__chg20"),
    "rrp_chg_4w_bn": ("liquidity", "fred_rrpontsyd__chg20"),
    "fed_assets_chg_13w": ("liquidity", "fred_walcl__pct13"),
    "dealer_total_z52": ("positioning", "dealer_total_bn__z52"),
}
# Variable -> (groupe, série de base selon la maturité, suffixe de la série dérivée).
MATURITY_MARKET_FEATURES = {
    "rate_level": ("market", RATE_SERIES, ""),
    "rate_chg_5d": ("market", RATE_SERIES, "__chg5"),
    "rate_chg_20d": ("market", RATE_SERIES, "__chg20"),
    "rate_vol_20d": ("market", RATE_SERIES, "__vol20"),
    "dealer_net_z52": ("positioning", DEALER_SERIES, "_bn__z52"),
    "dealer_net_chg_4w_bn": ("positioning", DEALER_SERIES, "_bn__chg4"),
    "cftc_am_net_z52": (
        "positioning",
        {k: f"cftc_{v}" for k, v in CFTC_CONTRACTS.items()},
        "_am_ratio__z52",
    ),
    "cftc_lev_net_z52": (
        "positioning",
        {k: f"cftc_{v}" for k, v in CFTC_CONTRACTS.items()},
        "_lev_ratio__z52",
    ),
}

AUCTION_FEATURES = (
    "maturity_years",
    "is_reopening",
    "log_offering",
    "offering_change",
    "days_since_previous",
    "weekday",
    "refunding_month",
    "days_to_quarter_end",
    "is_quarter_end",
)
HISTORY_FEATURES = tuple(
    f"{target}_{kind}"
    for target in TARGETS
    for kind in (*(f"lag{k}" for k in HISTORY_LAGS), "mean3")
)
DEMAND_FEATURES = ("recent_relative_btc",)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "auction": AUCTION_FEATURES,
    "history": (*HISTORY_FEATURES, *DEMAND_FEATURES),
    **{
        group: tuple(
            name
            for name, spec in {**FIXED_MARKET_FEATURES, **MATURITY_MARKET_FEATURES}.items()
            if spec[0] == group
        )
        for group in ("market", "liquidity", "positioning")
    },
}
FEATURE_COLUMNS = tuple(name for names in FEATURE_GROUPS.values() for name in names)
# Les « nouvelles sources » de l'expérience clé du plan : dealers, CFTC et liquidité.
NEW_SOURCE_GROUPS = ("liquidity", "positioning")

# --- Variables candidates (novembre 2026) ----------------------------------------------------
# Calculées et contrôlées comme les autres, mais hors du modèle du site : elles n'y entrent que
# si le test walk-forward « avec et sans » (tresor.new_data) montre qu'elles aident.
CANDIDATE_MARKET_FEATURES = {
    # Dette américaine gardée par la Fed pour les banques centrales étrangères.
    "foreign_custody_chg_4w": ("foreign_official", "fred_wmtsecl1__pct4"),
    "foreign_custody_chg_13w": ("foreign_official", "fred_wmtsecl1__pct13"),
    # Taux étrangers : variation sur un mois et écart de taux entre États-Unis et Allemagne.
    "bund_10y_chg_20d": ("foreign_rates", "bund_10y__chg20"),
    "jgb_10y_chg_20d": ("foreign_rates", "jgb_10y__chg20"),
    "us_bund_spread_10y": ("foreign_rates", "us_bund_spread_10y"),
}
# Jours jusqu'à la prochaine annonce (0 : le jour même) et depuis la dernière décision de la Fed,
# plafonnés : au-delà de 45 jours, l'annonce est trop loin pour compter.
CALENDAR_FEATURES = {
    "days_to_fomc": ("fomc", "to"),
    "days_since_fomc": ("fomc", "since"),
    "days_to_cpi": ("cpi", "to"),
    "days_to_jobs": ("jobs", "to"),
}
CALENDAR_CAP_DAYS = 45
CANDIDATE_GROUPS: dict[str, tuple[str, ...]] = {
    "foreign_official": tuple(
        n for n, spec in CANDIDATE_MARKET_FEATURES.items() if spec[0] == "foreign_official"
    ),
    "calendar": tuple(CALENDAR_FEATURES),
    "foreign_rates": tuple(
        n for n, spec in CANDIDATE_MARKET_FEATURES.items() if spec[0] == "foreign_rates"
    ),
}
CANDIDATE_COLUMNS = tuple(name for names in CANDIDATE_GROUPS.values() for name in names)
ALL_FEATURE_COLUMNS = (*FEATURE_COLUMNS, *CANDIDATE_COLUMNS)

ID_COLUMNS = (
    "cusip",
    "auction_date",
    "family",
    "maturity_bucket",
    "series_run",
    "status",
    "reopening",
    "offering_amt",
    "offering_estimated",  # 1 si le montant n'est pas encore annoncé (prévision en direct)
)

DerivedSeries = dict[str, tuple[pd.DataFrame, pd.Timedelta]]


# --- Séries dérivées : chaque valeur n'est connue que quand toute sa fenêtre l'était ------------


def _latest(dates: pd.Series, size: int) -> pd.Series:
    """Date la plus tardive sur une fenêtre glissante (calculée en jours, sans perte)."""
    days = (dates - EPOCH).dt.days.astype(float)
    return EPOCH + pd.to_timedelta(days.rolling(size, min_periods=size).max(), unit="D")


def _windowed(frame: pd.DataFrame, values: pd.Series, size: int) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": frame["date"],
            "value": values,
            "available_at": _latest(frame["available_at"], size),
        }
    )
    return out.dropna(subset=["value", "available_at"]).reset_index(drop=True)


def _timely(frame: pd.DataFrame) -> pd.DataFrame:
    """Les observations telles qu'elles existaient en temps réel, sans les compléments tardifs."""
    return frame[frame["available_at"] - frame["date"] <= LATE_FILL].reset_index(drop=True)


def change(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Variation sur `periods` observations."""
    frame = _timely(frame)
    return _windowed(frame, frame["value"] - frame["value"].shift(periods), periods + 1)


def pct_change(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Variation relative sur `periods` observations."""
    frame = _timely(frame)
    return _windowed(frame, frame["value"] / frame["value"].shift(periods) - 1, periods + 1)


def volatility(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Écart-type des variations quotidiennes sur `periods` observations."""
    frame = _timely(frame)
    return _windowed(frame, frame["value"].diff().rolling(periods).std(), periods + 1)


def zscore(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Écart à la moyenne des `periods` dernières observations, en nombre d'écarts-types."""
    frame = _timely(frame)
    window = frame["value"].rolling(periods, min_periods=periods)
    return _windowed(frame, (frame["value"] - window.mean()) / window.std(), periods)


def scaled(frame: pd.DataFrame, factor: float) -> pd.DataFrame:
    return frame.assign(value=frame["value"] * factor)


def combine(
    left: pd.DataFrame, right: pd.DataFrame, op: Callable[[pd.Series, pd.Series], pd.Series]
) -> pd.DataFrame:
    """Combine deux séries à date égale ; le résultat est connu quand les deux le sont."""
    merged = left.merge(right, on="date", suffixes=("_left", "_right"))
    return pd.DataFrame(
        {
            "date": merged["date"],
            "value": op(merged["value_left"], merged["value_right"]),
            "available_at": merged[["available_at_left", "available_at_right"]].max(axis=1),
        }
    )


def _max_age(base: str) -> pd.Timedelta:
    return WEEKLY_MAX_AGE if base.startswith(WEEKLY_BASES) else DAILY_MAX_AGE


def derived_series(market: pd.DataFrame) -> DerivedSeries:
    """Toutes les séries utilisées par les variables, avec leur âge maximal.

    Une série de base absente du marché est simplement ignorée : la variable sera vide, et le
    tableau des valeurs manquantes le montrera.
    """
    base = {
        name: part[["date", "value", "available_at"]].sort_values("date").reset_index(drop=True)
        for name, part in market.groupby("series")
    }
    out: DerivedSeries = {}

    def add(name: str, origin: str, build: Callable[[], pd.DataFrame]) -> None:
        if all(part in base for part in origin.split("+")):
            out[name] = (build(), _max_age(origin))

    for rate in RATE_SERIES.values():
        add(rate, rate, lambda r=rate: base[r])
        add(f"{rate}__chg5", rate, lambda r=rate: change(base[r], 5))
        add(f"{rate}__chg20", rate, lambda r=rate: change(base[r], 20))
        add(f"{rate}__vol20", rate, lambda r=rate: volatility(base[r], 20))
    add(
        "slope_10y_2y",
        "yield_10y+yield_2y",
        lambda: combine(base["yield_10y"], base["yield_2y"], operator.sub),
    )
    add("yield_3m", "yield_3m", lambda: base["yield_3m"])
    add("vix_close", "vix_close", lambda: base["vix_close"])
    add("vix_close__chg5", "vix_close", lambda: change(base["vix_close"], 5))
    add("fred_dtwexbgs__pct20", "fred_dtwexbgs", lambda: pct_change(base["fred_dtwexbgs"], 20))
    add("fred_dtwexb__pct20", "fred_dtwexb", lambda: pct_change(base["fred_dtwexb"], 20))
    add("fred_baa10y", "fred_baa10y", lambda: base["fred_baa10y"])
    add("fred_t10yie", "fred_t10yie", lambda: base["fred_t10yie"])
    add("debt_total__pct21", "debt_total", lambda: pct_change(base["debt_total"], 21))
    add("tga_bn__chg20", "tga_close", lambda: change(scaled(base["tga_close"], 1e-3), 20))
    add("fred_rrpontsyd__chg20", "fred_rrpontsyd", lambda: change(base["fred_rrpontsyd"], 20))
    add("fred_walcl__pct13", "fred_walcl", lambda: pct_change(base["fred_walcl"], 13))
    add("fred_wmtsecl1__pct4", "fred_wmtsecl1", lambda: pct_change(base["fred_wmtsecl1"], 4))
    add("fred_wmtsecl1__pct13", "fred_wmtsecl1", lambda: pct_change(base["fred_wmtsecl1"], 13))
    add("bund_10y__chg20", "bund_10y", lambda: change(base["bund_10y"], 20))
    add("jgb_10y__chg20", "jgb_10y", lambda: change(base["jgb_10y"], 20))
    add(
        "us_bund_spread_10y",
        "yield_10y+bund_10y",
        lambda: combine(base["yield_10y"], base["bund_10y"], operator.sub),
    )
    for dealer in {*DEALER_SERIES.values(), "dealer_total"}:
        add(f"{dealer}_bn__z52", dealer, lambda d=dealer: zscore(base[d], ZSCORE_WEEKS))
        add(f"{dealer}_bn__chg4", dealer, lambda d=dealer: change(scaled(base[d], 1e-3), 4))
    for contract in set(CFTC_CONTRACTS.values()):
        prefix = f"cftc_{contract}"
        for side in ("am", "lev"):
            add(
                f"{prefix}_{side}_ratio__z52",
                f"{prefix}_{side}_net+{prefix}_oi",
                lambda p=prefix, s=side: zscore(
                    combine(base[f"{p}_{s}_net"], base[f"{p}_oi"], operator.truediv), ZSCORE_WEEKS
                ),
            )
    return out


# --- Variables -----------------------------------------------------------------------------


def _point_in_time(
    dates: pd.Series, derived: DerivedSeries, names: str | tuple[str, ...]
) -> tuple[pd.Series, pd.Series]:
    """Valeur connue avant chaque date, et sa date de publication (vide si série absente).

    Avec plusieurs séries, la première connue à la date est retenue (raccord de deux séries).
    """
    value = pd.Series(np.nan, index=dates.index)
    seen = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    for name in (names,) if isinstance(names, str) else names:
        if name not in derived:
            continue
        frame, max_age = derived[name]
        known = point_in_time(dates, frame, max_age=max_age)
        missing = value.isna() & known["value"].notna()
        value = value.mask(missing, known["value"].astype(float))
        seen = seen.mask(missing, known["available_at"])
    return value, seen


def market_features(
    rows: pd.DataFrame, derived: DerivedSeries
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Variables de marché, de liquidité et de positionnement à t-1."""
    values, known = {}, {}
    for name, (_, series) in {**FIXED_MARKET_FEATURES, **CANDIDATE_MARKET_FEATURES}.items():
        values[name], known[name] = _point_in_time(rows["auction_date"], derived, series)
    for name, (_, by_maturity, suffix) in MATURITY_MARKET_FEATURES.items():
        value = pd.Series(np.nan, index=rows.index)
        seen = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns]")
        for maturity, index in rows.groupby("maturity_bucket").groups.items():
            series = by_maturity[maturity] + suffix
            value.loc[index], seen.loc[index] = _point_in_time(
                rows.loc[index, "auction_date"], derived, series
            )
        values[name], known[name] = value, seen
    return pd.DataFrame(values), pd.DataFrame(known)


def calendar_features(
    rows: pd.DataFrame, events: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jours jusqu'à la prochaine annonce et depuis la dernière décision de la Fed.

    Une annonce à venir ne compte que si sa date était déjà publique avant l'enchère
    (`known_at`, le 1er décembre de l'année d'avant) ; sinon la variable reste vide.
    """
    dates = rows["auction_date"].to_numpy(dtype="datetime64[ns]")
    values, known = {}, {}
    for name, (event, direction) in CALENDAR_FEATURES.items():
        value = np.full(len(rows), np.nan)
        seen = np.full(len(rows), np.datetime64("NaT"), dtype="datetime64[ns]")
        if events is not None and not events.empty:
            part = events[events["event"] == event].sort_values("date")
            event_dates = part["date"].to_numpy(dtype="datetime64[ns]")
            known_at = part["known_at"].to_numpy(dtype="datetime64[ns]")
            if direction == "to":  # prochaine annonce, le jour même compris
                index = np.searchsorted(event_dates, dates, side="left")
                found = index < len(event_dates)
                safe = np.minimum(index, len(event_dates) - 1)
                found &= known_at[safe] < dates  # date déjà publique avant l'enchère
                days = (event_dates[safe] - dates) / np.timedelta64(1, "D")
                seen = np.where(found, known_at[safe], seen)
            else:  # dernière décision, strictement avant le jour de l'enchère
                index = np.searchsorted(event_dates, dates, side="left") - 1
                found = index >= 0
                safe = np.maximum(index, 0)
                days = (dates - event_dates[safe]) / np.timedelta64(1, "D")
                seen = np.where(found, event_dates[safe], seen)
            value = np.where(found, np.minimum(days, CALENDAR_CAP_DAYS), np.nan)
        values[name] = pd.Series(value, index=rows.index)
        known[name] = pd.Series(seen, index=rows.index)
    return pd.DataFrame(values), pd.DataFrame(known)


def _previous_in_run(rows: pd.DataFrame, history: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pour chaque enchère, la ligne de `history` de la même série, à la date strictement antérieure
    la plus récente. `history` a une ligne par (series_run, previous_date)."""
    left = rows[["auction_date", "series_run"]].reset_index(names="row").sort_values("auction_date")
    right = history.sort_values("previous_date")[["series_run", "previous_date", *columns]]
    merged = pd.merge_asof(
        left,
        right,
        left_on="auction_date",
        right_on="previous_date",
        by="series_run",
        direction="backward",
        allow_exact_matches=False,  # une enchère du même jour n'est pas « précédente »
    )
    return merged.set_index("row").reindex(rows.index)


def history_features(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cibles des 1, 2 et 3 enchères précédentes de la même série, et leur moyenne."""
    values, known = {}, {}
    for target in TARGETS:
        done = rows[rows[target].notna()]
        daily = (
            done.groupby(["series_run", "auction_date"], as_index=False)[target]
            .mean()
            .sort_values(["series_run", "auction_date"])
        )
        by_run = daily.groupby("series_run")[target]
        daily["lag1"] = daily[target]
        daily["lag2"] = by_run.shift(1)
        daily["lag3"] = by_run.shift(2)
        daily = daily.rename(columns={"auction_date": "previous_date"})
        previous = _previous_in_run(rows, daily, ["lag1", "lag2", "lag3"])
        for lag in HISTORY_LAGS:
            values[f"{target}_lag{lag}"] = previous[f"lag{lag}"]
        values[f"{target}_mean3"] = previous[["lag1", "lag2", "lag3"]].mean(axis=1)
        for kind in (*(f"lag{k}" for k in HISTORY_LAGS), "mean3"):
            known[f"{target}_{kind}"] = previous["previous_date"]
    return pd.DataFrame(values), pd.DataFrame(known)


def recent_demand(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Moyenne du bid-to-cover relatif des 5 dernières enchères de la famille, avant t.

    Bid-to-cover relatif : bid-to-cover divisé par sa moyenne sur les 12 enchères précédentes
    de la même série.
    """
    done = rows[rows["btc"].notna()].sort_values(["series_run", "auction_date"], kind="stable")
    past_mean = done.groupby("series_run")["btc"].transform(
        lambda s: s.shift(1).rolling(REL_BTC_WINDOW, min_periods=REL_BTC_MIN_PERIODS).mean()
    )
    by_date = done.assign(relative_btc=done["btc"] / past_mean).sort_values(
        "auction_date", kind="stable"
    )
    by_date["recent_relative_btc"] = by_date.groupby("family")["relative_btc"].transform(
        lambda s: s.rolling(RECENT_DEMAND_AUCTIONS, min_periods=RECENT_DEMAND_MIN_PERIODS).mean()
    )
    right = by_date[["family", "auction_date", "recent_relative_btc"]].rename(
        columns={"auction_date": "demand_date"}
    )
    left = rows[["auction_date", "family"]].reset_index(names="row").sort_values("auction_date")
    merged = (
        pd.merge_asof(
            left,
            right,
            left_on="auction_date",
            right_on="demand_date",
            by="family",
            direction="backward",
            allow_exact_matches=False,
        )
        .set_index("row")
        .reindex(rows.index)
    )
    values = merged[["recent_relative_btc"]]
    known = merged[["demand_date"]].rename(columns={"demand_date": "recent_relative_btc"})
    return values, known


def _with_estimated_offering(rows: pd.DataFrame) -> pd.DataFrame:
    """Montant pas encore annoncé : celui de la dernière enchère de la même série, marqué estimé."""
    announced = rows[rows["offering_amt"].notna()]
    history = (
        announced.groupby(["series_run", "auction_date"], as_index=False)["offering_amt"]
        .mean()
        .rename(columns={"auction_date": "previous_date", "offering_amt": "previous_offering"})
    )
    previous = _previous_in_run(rows, history, ["previous_offering"])
    estimated = rows["offering_amt"].isna()
    return rows.assign(
        offering_amt=rows["offering_amt"].fillna(previous["previous_offering"]),
        offering_estimated=estimated.astype(int),
        offering_known_at=rows["announcemt_date"].where(~estimated, previous["previous_date"]),
    )


def auction_features(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Caractéristiques de l'enchère, connues à son annonce."""
    history = (
        rows.groupby(["series_run", "auction_date"], as_index=False)["offering_amt"]
        .mean()
        .rename(columns={"auction_date": "previous_date", "offering_amt": "previous_offering"})
    )
    previous = _previous_in_run(rows, history, ["previous_offering"])
    dates = rows["auction_date"]
    quarter_end = dates.dt.to_period("Q").dt.end_time.dt.normalize()
    days_to_quarter_end = (quarter_end - dates).dt.days
    values = pd.DataFrame(
        {
            "maturity_years": rows["bucket_years"],
            "is_reopening": rows["reopening"].fillna(False).astype(int),
            "log_offering": np.log(rows["offering_amt"]),
            "offering_change": np.log(rows["offering_amt"] / previous["previous_offering"]),
            "days_since_previous": (dates - previous["previous_date"]).dt.days,
            "weekday": dates.dt.day_of_week,
            "refunding_month": dates.dt.month.isin(REFUNDING_MONTHS).astype(int),
            "days_to_quarter_end": days_to_quarter_end,
            "is_quarter_end": (days_to_quarter_end <= QUARTER_END_DAYS).astype(int),
        }
    )
    known = pd.DataFrame({name: rows["offering_known_at"] for name in AUCTION_FEATURES})
    # Les variables fondées sur l'enchère précédente sont connues à la date de celle-ci.
    for name in ("offering_change", "days_since_previous"):
        known[name] = known[name].where(
            known[name] > previous["previous_date"], previous["previous_date"]
        )
    return values, known


def build_features(
    auctions: pd.DataFrame, market: pd.DataFrame, events: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Table des variables et des cibles, et date de publication de chaque information utilisée."""
    rows = auctions[auctions["family"].isin(MODELED_FAMILIES)].sort_values(
        ["auction_date", "cusip"], kind="stable"
    )
    rows = _with_estimated_offering(rows.reset_index(drop=True))
    derived = derived_series(market)

    market_values, market_known = market_features(rows, derived)
    # Tail : taux obtenu moins taux de marché de la veille (clôture de t-1), en points de base.
    rows["tail_bp"] = (rows["auction_rate"] - market_values["rate_level"]) * 100

    history_values, history_known = history_features(rows)
    demand_values, demand_known = recent_demand(rows)
    auction_values, auction_known = auction_features(rows)
    calendar_values, calendar_known = calendar_features(rows, events)

    features = pd.concat(
        [
            rows[list(ID_COLUMNS)],
            auction_values,
            history_values,
            demand_values,
            market_values,
            calendar_values,
            rows[list(TARGETS)],
        ],
        axis=1,
    )
    provenance = pd.concat(
        [auction_known, history_known, demand_known, market_known, calendar_known], axis=1
    )
    return features, provenance[list(ALL_FEATURE_COLUMNS)]


# --- Contrôles et sorties ------------------------------------------------------------------


def look_ahead_violations(features: pd.DataFrame, provenance: pd.DataFrame) -> pd.Series:
    """Nombre de valeurs, par variable, fondées sur une information publiée trop tard.

    Caractéristiques de l'enchère : connues au plus tard le jour de l'enchère (annonce).
    Toutes les autres : publiées strictement avant ce jour.
    """
    dates = features["auction_date"]
    names = [name for name in ALL_FEATURE_COLUMNS if name in provenance.columns]
    present = features[names].notna()
    counts = {}
    for name in names:
        known = provenance[name]
        if name in AUCTION_FEATURES:
            too_late = known > dates
        else:
            too_late = known >= dates
        counts[name] = int((present[name] & (too_late | known.isna())).sum())
    return pd.Series(counts)


def minimum_publication_dates(market: pd.DataFrame) -> pd.Series:
    """Date de publication la plus précoce possible de chaque valeur, d'après le calendrier
    connu de chaque source. Contrôle indépendant de available_at, qui ne se vérifie pas lui-même.
    """
    dates = market["date"]
    minimum = dates.copy()  # clôtures du jour (taux, bills, VIX) et FRED daté par ALFRED : >= date
    rules = {
        "fiscaldata_cash_balance": lambda d: add_business_days(d, 1),
        "fiscaldata_debt": lambda d: add_business_days(d, 2),
        "nyfed_dealer_positions": lambda d: d + pd.Timedelta(days=8),
        "cftc_positions": lambda d: d + pd.to_timedelta(4 - d.dt.day_of_week, unit="D"),
        "bundesbank_bund_yield": lambda d: add_business_days(d, 1),
        # Publié à 9 h 30 à Tokyo le jour ouvré japonais suivant : au plus tôt le soir même
        # à New York, jamais avant.
        "mof_jgb_yield": lambda d: d,
    }
    for source, rule in rules.items():
        rows = market["source"].eq(source)
        if rows.any():
            minimum.loc[rows] = rule(dates[rows]).to_numpy()
    for series, days in MINIMUM_LAG_DAYS.items():
        rows = market["series"].eq(series)
        minimum.loc[rows] = (dates[rows] + pd.Timedelta(days=days)).to_numpy()
    return minimum


def publication_lag_violations(market: pd.DataFrame) -> pd.Series:
    """Nombre de valeurs, par série, dont la date de publication précède le délai minimal."""
    too_early = market["available_at"].isna() | (
        market["available_at"] < minimum_publication_dates(market)
    )
    return too_early.groupby(market["series"]).sum().astype(int)


def model_sample(features: pd.DataFrame) -> pd.DataFrame:
    """Enchères réalisées depuis le début de l'échantillon des modèles (2009)."""
    in_sample = (features["status"] == "completed") & (
        features["auction_date"] >= pd.Timestamp(MODEL_SAMPLE_START)
    )
    return features[in_sample]


def missing_rates(features: pd.DataFrame) -> pd.DataFrame:
    """Part de valeurs manquantes (%) par variable et par famille : sur tout l'échantillon des
    modèles (depuis 2009) et sur la période de test du walk-forward (depuis 2014)."""
    sample = model_sample(features)
    columns = [*ALL_FEATURE_COLUMNS, *TARGETS]
    parts = {}
    for label, part in {
        "2009+": sample,
        "2014+": sample[sample["auction_date"] >= TEST_START],
    }.items():
        rates = part.groupby("family")[columns].agg(lambda s: s.isna().mean() * 100).T
        parts.update({f"{family} {label}": rates[family] for family in rates.columns})
    table = pd.DataFrame(parts)
    table.insert(0, "groupe", [_group_of(name) for name in table.index])
    return table.round(1)


def _group_of(name: str) -> str:
    for group, names in {**FEATURE_GROUPS, **CANDIDATE_GROUPS}.items():
        if name in names:
            return group
    return "cible"


def run_features(clean_dir: Path = CLEAN_DIR, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Construit les variables, refuse toute fuite d'information, écrit la table et le bilan."""
    auctions = pd.read_parquet(clean_dir / "auctions.parquet")
    market = pd.read_parquet(clean_dir / "market.parquet")
    events_path = clean_dir / "events.parquet"
    events = pd.read_parquet(events_path) if events_path.exists() else None
    features, provenance = build_features(auctions, market, events)

    # Deux contrôles bloquants : les dates de publication du marché respectent les délais
    # minimaux de chaque source, et aucune variable n'utilise une information publiée trop tard.
    lags = publication_lag_violations(market)
    if lags.sum():
        raise ValueError(f"dates de publication trop précoces : {lags[lags > 0].to_dict()}")
    violations = look_ahead_violations(features, provenance)
    if violations.sum():
        raise ValueError(f"fuite d'information : {violations[violations > 0].to_dict()}")

    clean_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Écriture atomique : l'application ne lit jamais un fichier à moitié écrit.
    path = clean_dir / "features.parquet"
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
    features.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)
    rates = missing_rates(features)
    path = output_dir / "features_missing.csv"
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
    rates.to_csv(temporary_path, index_label="variable")
    temporary_path.replace(path)
    logger.info(
        "Variables : %d enchères, %d variables, aucune fuite d'information détectée",
        len(features),
        len(FEATURE_COLUMNS),
    )
    logger.info("Valeurs manquantes (%%) sur l'échantillon des modèles :\n%s", rates.to_string())
    return features


def feature_groups_without(
    groups: Mapping[str, tuple[str, ...]], excluded: tuple[str, ...]
) -> list[str]:
    """Variables de tous les groupes sauf ceux exclus (expérience « sans nouvelles sources »)."""
    return [name for group, names in groups.items() if group not in excluded for name in names]
