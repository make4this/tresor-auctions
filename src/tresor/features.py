"""Variables explicatives point-in-time (section 7 du plan) : seulement ce qui est connu avant t.

Règle : pour une enchère du jour t, une variable de marché ou d'historique n'utilise que des
informations publiées strictement avant t. Les caractéristiques de l'enchère elle-même (montant,
maturité, réouverture) sont connues à son annonce, toujours antérieure ou égale à t.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from tresor.clean.market import point_in_time
from tresor.config import CLEAN_DIR, MODEL_SAMPLE_START, OUTPUT_DIR

logger = logging.getLogger(__name__)

MODELED_FAMILIES = ("nominal_coupon", "bill")
TARGETS = ("btc", "dealer_share", "indirect_share", "tail_bp")
HISTORY_LAGS = (1, 2, 3)

REL_BTC_WINDOW, REL_BTC_MIN_PERIODS = 12, 6  # bid-to-cover relatif : sur 12 enchères passées
RECENT_DEMAND_AUCTIONS, RECENT_DEMAND_MIN_PERIODS = 5, 3
REFUNDING_MONTHS = (2, 5, 8, 11)  # refinancement trimestriel du Trésor
QUARTER_END_DAYS = 10

# Âge maximal d'une valeur de marché : au-delà, elle est jugée périmée (série interrompue).
DAILY_MAX_AGE = pd.Timedelta(days=10)
WEEKLY_MAX_AGE = pd.Timedelta(days=21)  # séries publiées une fois par semaine
WEEKLY_BASES = ("dealer_", "cftc_", "fred_walcl", "fred_wtregen", "fred_dtwexbgs")

EPOCH = pd.Timestamp("1970-01-01")

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
# Contrat future le plus proche en maturité ; les bills prennent le 2 ans, faute de mieux.
CFTC_CONTRACTS = {
    "2Y": "2y",
    "3Y": "2y",
    "5Y": "5y",
    "7Y": "10y",
    "10Y": "10y",
    "20Y": "bond",
    "30Y": "bond",
    **{bucket: "2y" for bucket in BILL_BUCKETS},
}

# Variable -> (groupe, série dérivée commune à toutes les enchères).
FIXED_MARKET_FEATURES = {
    "slope_10y_2y": ("market", "slope_10y_2y"),
    "rate_3m": ("market", "yield_3m"),
    "vix": ("market", "vix_close"),
    "vix_chg_5d": ("market", "vix_close__chg5"),
    "dollar_chg_20d": ("market", "fred_dtwexbgs__pct20"),
    "credit_spread_baa": ("market", "fred_baa10y"),
    "breakeven_10y": ("market", "fred_t10yie"),
    "debt_chg_30d": ("liquidity", "debt_total__pct21"),
    "tga_bn": ("liquidity", "tga_bn"),
    "rrp_bn": ("liquidity", "fred_rrpontsyd"),
    "fed_assets_bn": ("liquidity", "fred_walcl_bn"),
    "dealer_total_bn": ("positioning", "dealer_total_bn"),
}
# Variable -> (groupe, série de base selon la maturité, suffixe de la série dérivée).
MATURITY_MARKET_FEATURES = {
    "rate_level": ("market", RATE_SERIES, ""),
    "rate_chg_5d": ("market", RATE_SERIES, "__chg5"),
    "rate_chg_20d": ("market", RATE_SERIES, "__chg20"),
    "rate_vol_20d": ("market", RATE_SERIES, "__vol20"),
    "dealer_net_bn": ("positioning", DEALER_SERIES, "_bn"),
    "dealer_net_chg_4w_bn": ("positioning", DEALER_SERIES, "_bn__chg4"),
    "cftc_am_net_ratio": (
        "positioning",
        {k: f"cftc_{v}" for k, v in CFTC_CONTRACTS.items()},
        "_am_ratio",
    ),
    "cftc_lev_net_ratio": (
        "positioning",
        {k: f"cftc_{v}" for k, v in CFTC_CONTRACTS.items()},
        "_lev_ratio",
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
    "offering_estimated",
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

ID_COLUMNS = (
    "cusip",
    "auction_date",
    "family",
    "maturity_bucket",
    "series_run",
    "status",
    "reopening",
    "offering_amt",
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


def change(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Variation sur `periods` observations."""
    return _windowed(frame, frame["value"] - frame["value"].shift(periods), periods + 1)


def pct_change(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Variation relative sur `periods` observations."""
    return _windowed(frame, frame["value"] / frame["value"].shift(periods) - 1, periods + 1)


def volatility(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Écart-type des variations quotidiennes sur `periods` observations."""
    return _windowed(frame, frame["value"].diff().rolling(periods).std(), periods + 1)


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
    add("fred_baa10y", "fred_baa10y", lambda: base["fred_baa10y"])
    add("fred_t10yie", "fred_t10yie", lambda: base["fred_t10yie"])
    add("debt_total__pct21", "debt_total", lambda: pct_change(base["debt_total"], 21))
    add("tga_bn", "tga_close", lambda: scaled(base["tga_close"], 1e-3))
    add("fred_rrpontsyd", "fred_rrpontsyd", lambda: base["fred_rrpontsyd"])
    add("fred_walcl_bn", "fred_walcl", lambda: scaled(base["fred_walcl"], 1e-3))
    for dealer in {*DEALER_SERIES.values(), "dealer_total"}:
        add(f"{dealer}_bn", dealer, lambda d=dealer: scaled(base[d], 1e-3))
        add(f"{dealer}_bn__chg4", dealer, lambda d=dealer: change(scaled(base[d], 1e-3), 4))
    for contract in set(CFTC_CONTRACTS.values()):
        prefix = f"cftc_{contract}"
        for side in ("am", "lev"):
            add(
                f"{prefix}_{side}_ratio",
                f"{prefix}_{side}_net+{prefix}_oi",
                lambda p=prefix, s=side: combine(
                    base[f"{p}_{s}_net"], base[f"{p}_oi"], operator.truediv
                ),
            )
    return out


# --- Variables -----------------------------------------------------------------------------


def _point_in_time(
    dates: pd.Series, derived: DerivedSeries, name: str
) -> tuple[pd.Series, pd.Series]:
    """Valeur connue avant chaque date, et sa date de publication (vide si série absente)."""
    if name not in derived:
        return pd.Series(np.nan, index=dates.index), pd.Series(pd.NaT, index=dates.index)
    frame, max_age = derived[name]
    known = point_in_time(dates, frame, max_age=max_age)
    return known["value"].astype(float), known["available_at"]


def market_features(
    rows: pd.DataFrame, derived: DerivedSeries
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Variables de marché, de liquidité et de positionnement à t-1."""
    values, known = {}, {}
    for name, (_, series) in FIXED_MARKET_FEATURES.items():
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
            "offering_estimated": rows["offering_estimated"],
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
    auctions: pd.DataFrame, market: pd.DataFrame
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

    features = pd.concat(
        [
            rows[list(ID_COLUMNS)],
            auction_values,
            history_values,
            demand_values,
            market_values,
            rows[list(TARGETS)],
        ],
        axis=1,
    )
    provenance = pd.concat([auction_known, history_known, demand_known, market_known], axis=1)
    return features, provenance[list(FEATURE_COLUMNS)]


# --- Contrôles et sorties ------------------------------------------------------------------


def look_ahead_violations(features: pd.DataFrame, provenance: pd.DataFrame) -> pd.Series:
    """Nombre de valeurs, par variable, fondées sur une information publiée trop tard.

    Caractéristiques de l'enchère : connues au plus tard le jour de l'enchère (annonce).
    Toutes les autres : publiées strictement avant ce jour.
    """
    dates = features["auction_date"]
    present = features[list(FEATURE_COLUMNS)].notna()
    counts = {}
    for name in FEATURE_COLUMNS:
        known = provenance[name]
        if name in AUCTION_FEATURES:
            too_late = known > dates
        else:
            too_late = known >= dates
        counts[name] = int((present[name] & (too_late | known.isna())).sum())
    return pd.Series(counts)


def model_sample(features: pd.DataFrame) -> pd.DataFrame:
    """Enchères réalisées depuis le début de l'échantillon des modèles (2009)."""
    in_sample = (features["status"] == "completed") & (
        features["auction_date"] >= pd.Timestamp(MODEL_SAMPLE_START)
    )
    return features[in_sample]


def missing_rates(features: pd.DataFrame) -> pd.DataFrame:
    """Part de valeurs manquantes (%) par variable et par famille, sur l'échantillon des modèles."""
    sample = model_sample(features)
    columns = [*FEATURE_COLUMNS, *TARGETS]
    table = sample.groupby("family")[columns].agg(lambda s: s.isna().mean() * 100).T
    table.insert(0, "groupe", [_group_of(name) for name in table.index])
    return table.round(1)


def _group_of(name: str) -> str:
    for group, names in FEATURE_GROUPS.items():
        if name in names:
            return group
    return "cible"


def run_features(clean_dir: Path = CLEAN_DIR, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Construit les variables, refuse toute fuite d'information, écrit la table et le bilan."""
    auctions = pd.read_parquet(clean_dir / "auctions.parquet")
    market = pd.read_parquet(clean_dir / "market.parquet")
    features, provenance = build_features(auctions, market)

    violations = look_ahead_violations(features, provenance)
    if violations.sum():
        raise ValueError(f"fuite d'information : {violations[violations > 0].to_dict()}")

    clean_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(clean_dir / "features.parquet", index=False)
    rates = missing_rates(features)
    rates.to_csv(output_dir / "features_missing.csv", index_label="variable")
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
