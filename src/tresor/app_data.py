"""Données de l'application, préparées sans Streamlit (donc testables).

Chaque fonction prend des tables déjà lues et renvoie une table prête à afficher. Les textes
affichés sont en anglais (application en ligne) ; les commentaires restent en français.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tresor import models as md
from tresor.config import MODEL_SAMPLE_START, SIGNAL_THRESHOLD
from tresor.formats import en

SAMPLE_START = pd.Timestamp(MODEL_SAMPLE_START)

MATURITY_LABELS = {
    "2Y": "2-year",
    "3Y": "3-year",
    "5Y": "5-year",
    "7Y": "7-year",
    "10Y": "10-year",
    "20Y": "20-year",
    "30Y": "30-year",
    "4W": "4-week",
    "6W": "6-week",
    "8W": "8-week",
    "13W": "13-week",
    "17W": "17-week",
    "26W": "26-week",
    "52W": "52-week",
}
MATURITY_ORDER = list(MATURITY_LABELS)
FAMILY_LABELS = {"nominal_coupon": "Notes & bonds", "bill": "Bills"}
TARGET_LABELS = {
    "btc": "Bid-to-cover",
    "dealer_share": "Dealer share",
    "indirect_share": "Indirect share",
    "tail_bp": "Tail",
}
SHORT_TARGETS = {
    "btc": "bid-to-cover",
    "dealer_share": "dealer share",
    "indirect_share": "indirect share",
    "tail_bp": "tail",
}
WEAK, STRONG, NORMAL = "weaker", "stronger", "normal"
SOURCE_LABELS = {
    "fiscaldata_auctions": "Fiscal Data: auction results",
    "treasurydirect_recent": "TreasuryDirect: recent results",
    "treasurydirect_upcoming": "TreasuryDirect: auction calendar",
    "treasury_yield_curve": "Treasury.gov: yield curve",
    "treasury_bill_rates": "Treasury.gov: bill rates",
    "cboe_vix": "Cboe: VIX",
    "fiscaldata_debt": "Fiscal Data: public debt",
    "fiscaldata_cash_balance": "Fiscal Data: Treasury cash balance",
    "nyfed_dealer_positions": "New York Fed: dealer positions",
    "cftc_positions": "CFTC: futures positioning",
    "fred_series": "FRED: macro series",
}
# Noms des lignes reçues dans le rapport de nettoyage des enchères.
INPUT_LABELS = {
    "fiscaldata": "Fiscal Data",
    "treasurydirect": "TreasuryDirect (recent)",
    "treasurydirect_upcoming": "TreasuryDirect (calendar)",
}
EXCLUSION_LABELS = {
    "type_inconnu": "Unknown security type",
    "petite_emission_hors_calendrier": "Small off-calendar issue (under $1bn)",
    "passee_sans_resultat": "Past date without a result",
    "future_hors_calendrier": "Future date missing from the calendar",
}
MODEL_LABELS = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "ElasticNet",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "moyenne_5": "5-model average",
    "ridge_ecart": "Ridge (gap)",
    "lasso_ecart": "Lasso (gap)",
    "elasticnet_ecart": "ElasticNet (gap)",
    "constante": "Constant",
    "naive": "Naive (last auction)",
}
FEATURE_LABELS = {
    "maturity_years": "Maturity (years)",
    "is_reopening": "Reopening",
    "log_offering": "Offering size (log)",
    "offering_change": "Change in offering size",
    "days_since_previous": "Days since previous auction",
    "weekday": "Day of the week",
    "refunding_month": "Refunding month",
    "days_to_quarter_end": "Days to quarter end",
    "is_quarter_end": "Quarter end",
    "recent_relative_btc": "Recent demand (same family)",
    "slope_10y_2y": "10y − 2y slope",
    "rate_3m": "3-month rate",
    "vix": "VIX",
    "vix_chg_5d": "VIX, 5-day change",
    "dollar_chg_20d": "Dollar, 20-day change",
    "credit_spread_baa": "Baa credit spread",
    "breakeven_10y": "10-year breakeven inflation",
    "rate_level": "Yield of the maturity",
    "rate_chg_5d": "Yield, 5-day change",
    "rate_chg_20d": "Yield, 20-day change",
    "rate_vol_20d": "Yield volatility, 20 days",
    "debt_chg_30d": "Public debt, 30-day change",
    "tga_chg_4w_bn": "Treasury cash, 4-week change",
    "rrp_chg_4w_bn": "Reverse repo, 4-week change",
    "fed_assets_chg_13w": "Fed balance sheet, 13-week change",
    "dealer_total_z52": "Dealer positions, total (normalised)",
    "dealer_net_z52": "Dealer positions, same maturity (normalised)",
    "dealer_net_chg_4w_bn": "Dealer positions, same maturity, 4-week change",
    "cftc_am_net_z52": "Asset managers (CFTC)",
    "cftc_lev_net_z52": "Leveraged funds (CFTC)",
}
LAG_LABELS = {
    "lag1": "last auction",
    "lag2": "second-to-last auction",
    "lag3": "third-to-last auction",
    "mean3": "average of last 3",
}
RULE_LABELS = {
    "strategie": "Strategy (5-model average)",
    "toujours_vendeur": "Always short",
    "information_parfaite": "Perfect foresight (impossible in practice)",
}
SURPRISE_LABELS = {
    "part_des_dealers": "Dealer share surprise (actual − average of last 3)",
    "bid_to_cover": "Bid-to-cover surprise (actual − average of last 3)",
}
# Crises du plafond de la dette, visibles sur les bills courts.
DEBT_CEILING = {"Oct 2015": "2015-10-15", "Oct 2021": "2021-10-01", "May 2023": "2023-05-15"}


def feature_label(name: str) -> str:
    """Nom lisible d'une variable, en anglais."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    target, _, kind = name.rpartition("_")
    if target in TARGET_LABELS and kind in LAG_LABELS:
        return f"{TARGET_LABELS[target]} ({LAG_LABELS[kind]})"
    return name


def format_target(value: float, target: str, sign: bool = False) -> str:
    """Valeur d'une cible dans son unité : bid-to-cover à 2 décimales, parts en %, tail en pb."""
    if target in ("dealer_share", "indirect_share"):
        return en(value * 100, 1, sign=sign, suffix="%")
    if target == "tail_bp":
        return en(value, 1, sign=True, suffix=" bp")
    return en(value, 2, sign=sign)


def amount(value: float, estimated: bool = False) -> str:
    """Montant d'une enchère, en milliards de dollars : « $95bn »."""
    text = f"${en(value / 1e9, 0)}bn"
    return text + (" (est.)" if estimated else "")


def auction_title(bucket: str, family: str, reopening: bool) -> str:
    """« 10-year (reopening) ». Pour les bills, la réouverture n'apporte rien : non affichée."""
    label = MATURITY_LABELS.get(bucket, bucket)
    if family == "nominal_coupon" and bool(reopening):
        return f"{label} (reopening)"
    return label


def signal(gap: float, threshold: float = SIGNAL_THRESHOLD) -> str:
    """Signal sur l'écart non arrondi entre le bid-to-cover prévu et mean3.

    L'écart est d'abord arrondi à 10⁻⁹ : la règle « ≤ −0,08 » reste exacte malgré les erreurs
    d'arrondi des nombres à virgule (2,53 − 2,61 donne −0,08000000000000007).
    """
    if pd.isna(gap):
        return "–"
    gap = round(float(gap), 9)
    if gap <= -threshold:
        return WEAK
    if gap >= threshold:
        return STRONG
    return NORMAL


def _maturity_rank(buckets: pd.Series) -> pd.Series:
    order = {bucket: rank for rank, bucket in enumerate(MATURITY_ORDER)}
    return buckets.map(order).fillna(len(order))


def upcoming_table(
    log: pd.DataFrame,
    auctions: pd.DataFrame,
    today: pd.Timestamp,
    model: str = md.ENSEMBLE,
) -> pd.DataFrame:
    """Une ligne par enchère du jour ou à venir : prévisions de la moyenne des 5 et signal.

    Les valeurs restent numériques ; le formatage est fait à l'affichage.
    """
    rows = log[(log["model"] == model) & (log["auction_date"] >= today)]
    if rows.empty:
        return pd.DataFrame()
    keys = {"cusip": "string", "auction_date": "datetime64[us]"}
    rows = rows.astype(keys)
    wide = rows.pivot_table(
        index=["cusip", "auction_date", "family", "maturity_bucket"],
        columns="target",
        values=["pred", "bench_mean3"],
        aggfunc="first",
        dropna=False,  # garder une colonne même vide (première enchère d'une série sans mean3)
    )
    wide.columns = [f"{value}_{target}" for value, target in wide.columns]
    expected = [
        f"{value}_{target}" for value in ("pred", "bench_mean3") for target in TARGET_LABELS
    ]
    wide = wide.reindex(columns=expected).reset_index()
    wide = wide.dropna(subset=[f"pred_{target}" for target in TARGET_LABELS], how="all")
    extra = rows.groupby(["cusip", "auction_date"], as_index=False).agg(
        offering_amt=("offering_amt", "first"),
        offering_estimated=("offering_estimated", "max"),
        predicted_at=("predicted_at", "max"),
    )
    info = (
        auctions[["cusip", "auction_date", "reopening", "security_term"]]
        .astype(keys)
        .drop_duplicates(["cusip", "auction_date"])
    )
    table = wide.astype(keys).merge(extra, on=["cusip", "auction_date"])
    table = table.merge(info, on=["cusip", "auction_date"], how="left")
    table["title"] = [
        auction_title(bucket, family, reopening)
        for bucket, family, reopening in zip(
            table["maturity_bucket"], table["family"], table["reopening"].fillna(False), strict=True
        )
    ]
    table["gap_btc"] = table["pred_btc"] - table["bench_mean3_btc"]
    table["signal"] = table["gap_btc"].map(signal)
    table["rank"] = _maturity_rank(table["maturity_bucket"])
    return table.sort_values(["auction_date", "rank"], ignore_index=True).drop(columns="rank")


def signal_track_record(
    backtest: pd.DataFrame, threshold: float = SIGNAL_THRESHOLD
) -> pd.DataFrame:
    """Fiabilité passée du signal dans le backtest (moyenne des 5, bid-to-cover, 2014-2026).

    Pour chaque famille : à quelle fréquence chaque signal se déclenche, et dans quelle part des
    cas le bid-to-cover réel est ensuite du même côté de mean3 (d'un écart quelconque).
    """
    data = backtest[
        (backtest["model"] == md.ENSEMBLE)
        & (backtest["target"] == "btc")
        & (backtest["feature_set"] == "toutes")
    ].dropna(subset=["actual", "prediction", "mean3"])
    rows = []
    for family, part in data.groupby("family"):
        gap = (part["prediction"] - part["mean3"]).map(signal, threshold=threshold)
        below, above = part["actual"] < part["mean3"], part["actual"] > part["mean3"]
        weak, strong = gap == WEAK, gap == STRONG
        rows.append(
            {
                "family": family,
                "auctions": len(part),
                "weak_count": int(weak.sum()),
                "share_weak": weak.mean(),
                "weak_then_below": below[weak].mean() if weak.any() else np.nan,
                "strong_count": int(strong.sum()),
                "share_strong": strong.mean(),
                "strong_then_above": above[strong].mean() if strong.any() else np.nan,
                "base_rate_below": below.mean(),
                "base_rate_above": above.mean(),
            }
        )
    return pd.DataFrame(rows)


def live_summary(tracking: pd.DataFrame, model: str = md.ENSEMBLE) -> dict[str, int]:
    """Nombre d'enchères suivies, par statut (une enchère compte une fois)."""
    auctions = tracking[tracking["model"] == model].drop_duplicates(["cusip", "auction_date"])
    counts = auctions["status"].value_counts()
    return {
        "suivies": int(len(auctions)),
        "realisees": int(counts.get("realisee", 0)),
        "en_attente": int(counts.get("en_attente", 0)),
        "sans_resultat": int(counts.get("sans_resultat", 0)),
    }


def live_results(tracking: pd.DataFrame, target: str, model: str = md.ENSEMBLE) -> pd.DataFrame:
    """Prévisions réalisées pour une cible, avec le gain cumulé contre mean3.

    `gain` : somme cumulée de (réel − mean3)² − (réel − prévu)², rapportée à l'erreur totale
    de mean3, comme dans le backtest : la courbe finit au R² du suivi en direct.
    """
    done = tracking[
        (tracking["model"] == model)
        & (tracking["target"] == target)
        & (tracking["status"] == "realisee")
    ].dropna(subset=["actual", "bench_error", "error"])
    done = done.sort_values(["auction_date", "cusip"], kind="stable").reset_index(drop=True)
    reference = done["bench_error"] ** 2
    total = reference.sum()
    done["gain"] = (reference - done["error"] ** 2).cumsum() / total if total > 0 else np.nan
    return done


def cumulative_gain(
    backtest: pd.DataFrame, family: str, target: str, model: str = md.ENSEMBLE
) -> pd.DataFrame:
    """Gain cumulé du backtest contre mean3, en part de l'erreur totale de mean3."""
    data = backtest[
        (backtest["family"] == family)
        & (backtest["target"] == target)
        & (backtest["model"] == model)
        & (backtest["feature_set"] == "toutes")
    ].dropna(subset=["actual", "prediction", "mean3"])
    data = data.sort_values("auction_date", kind="stable")
    reference = (data["actual"] - data["mean3"]) ** 2
    gain = (reference - (data["actual"] - data["prediction"]) ** 2).cumsum() / reference.sum()
    return pd.DataFrame({"auction_date": data["auction_date"].to_numpy(), "gain": gain.to_numpy()})


def explorer_series(
    features: pd.DataFrame,
    backtest: pd.DataFrame,
    family: str,
    bucket: str,
    target: str,
    start: pd.Timestamp = SAMPLE_START,
) -> pd.DataFrame:
    """Historique d'une maturité depuis 2009 (l'échantillon des modèles) : valeur réelle,
    mean3 et prévision du backtest (moyenne des 5)."""
    history = features[
        (features["family"] == family)
        & (features["maturity_bucket"] == bucket)
        & (features["status"] == "completed")
        & (features["auction_date"] >= start)
    ][["cusip", "auction_date", target, f"{target}_mean3"]]
    history = history.rename(columns={target: "actual", f"{target}_mean3": "mean3"})
    history = history.dropna(subset=["actual"])
    predicted = backtest[
        (backtest["family"] == family)
        & (backtest["maturity_bucket"] == bucket)
        & (backtest["target"] == target)
        & (backtest["model"] == md.ENSEMBLE)
        & (backtest["feature_set"] == "toutes")
    ][["cusip", "auction_date", "prediction"]]
    keys = {"cusip": "string", "auction_date": "datetime64[us]"}
    return (
        history.astype(keys)
        .merge(predicted.astype(keys), on=["cusip", "auction_date"], how="left")
        .sort_values("auction_date", ignore_index=True)
    )


def quality_table(quality: dict[str, object]) -> pd.DataFrame:
    """Une ligne par source brute : statut, lignes, dernières dates, retard, alertes."""
    rows = []
    for name, source in dict(quality.get("sources", {})).items():  # type: ignore[call-overload]
        rows.append(
            {
                "source": name,
                "status": source.get("status"),
                "rows": source.get("rows"),
                "last_date": source.get("last_date"),
                "last_published": source.get("last_published"),
                "business_days_since_last": source.get("business_days_since_last"),
                "alerts": "; ".join(source.get("alerts", [])) or "none",
            }
        )
    return pd.DataFrame(rows)


def fetch_table(status: dict[str, object]) -> pd.DataFrame:
    """Résultat de chaque source au dernier téléchargement, les échecs en premier."""
    sources = status.get("sources", {})
    rows = [
        {
            "source": name,
            "ok": bool(info.get("ok")),
            "last_success_at": info.get("last_success_at"),
            "error": info.get("error") or "",
        }
        for name, info in (sources.items() if isinstance(sources, dict) else [])
        if isinstance(info, dict)
    ]
    columns = ["source", "ok", "last_success_at", "error"]
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values(["ok", "source"], kind="stable").reset_index(drop=True)


def failed_sources(status: dict[str, object]) -> list[str]:
    """Noms lisibles des sources en échec au dernier téléchargement."""
    table = fetch_table(status)
    return [SOURCE_LABELS.get(name, name) for name in table.loc[~table["ok"], "source"]]


def backtest_r2(metrics: pd.DataFrame, period: str = "2020-2026") -> pd.DataFrame:
    """R² de la moyenne des 5 contre mean3 (avec intervalle) et contre la constante."""
    data = metrics[
        (metrics["model"] == md.ENSEMBLE)
        & (metrics["feature_set"] == "toutes")
        & (metrics["period"] == period)
    ]
    columns = ["r2_vs_mean3", "r2_ci_low", "r2_ci_high", "r2_vs_constant", "direction"]
    return data[["family", "target", *columns]]


# --- Page d'accueil : les mêmes informations, en mots simples ---------------------------------

BADGES = {WEAK: ("Weaker demand", "tr-weak"), STRONG: ("Stronger demand", "tr-strong"),
          NORMAL: ("Normal demand", "tr-normal")}  # fmt: skip


def home_cards(table: pd.DataFrame) -> list[dict[str, str]]:
    """Une carte par enchère à venir : date, titre, montant, signal et une phrase simple.

    `table` : le résultat de `upcoming_table`. Les dates restent des Timestamp : la page les
    met en forme.
    """
    cards = []
    for row in table.itertuples():
        badge, css = BADGES.get(row.signal, ("No comparison yet", "tr-normal"))
        predicted = en(row.pred_btc, 2)
        if pd.isna(row.bench_mean3_btc):
            sentence = f"Expected bid-to-cover: {predicted}. First auction of its series, so no"
            sentence += " recent average to compare with."
        else:
            sentence = (
                f"Expected bid-to-cover: {predicted}, versus {en(row.bench_mean3_btc, 2)} on"
                " average over the last 3 auctions of the same maturity."
            )
        cards.append(
            {
                "date": row.auction_date,
                "title": row.title,
                "family": FAMILY_LABELS.get(row.family, row.family),
                "amount": amount(row.offering_amt, bool(row.offering_estimated)),
                "badge": badge,
                "css": css,
                "sentence": sentence,
            }
        )
    return cards


def key_facts(
    metrics: pd.DataFrame, summary: pd.DataFrame, quintiles: pd.DataFrame
) -> dict[str, float]:
    """Les trois chiffres de la page d'accueil, lus dans les fichiers du backtest et de la
    stratégie (les mêmes que dans reports/chiffres_cles.md)."""
    r2 = backtest_r2(metrics).set_index(["family", "target"])["r2_vs_mean3"]
    strategy = summary[(summary["period"] == "2020-2026") & (summary["rule"] == "strategie")]
    weakest = quintiles[
        (quintiles["period"] == "2009-2026")
        & (quintiles["surprise"] == "bid_to_cover")
        & (quintiles["quintile"] == 5)
    ]
    if len(strategy) != 1 or len(weakest) != 1:
        raise ValueError("fichiers de stratégie ou de quintiles inattendus")
    return {
        "r2_coupons": float(r2.loc[("nominal_coupon", "btc")]),
        "r2_bills": float(r2.loc[("bill", "btc")]),
        "weak_move_bp": float(weakest["rate_change_mean_bp"].iloc[0]),
        "strategy_bp": float(strategy["mean_bp"].iloc[0]),
    }
