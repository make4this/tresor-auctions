"""Données de l'application, préparées sans Streamlit (donc testables).

Chaque fonction prend des tables déjà lues et renvoie une table prête à afficher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tresor import models as md
from tresor.config import SIGNAL_THRESHOLD

MATURITY_LABELS = {
    "2Y": "2 ans",
    "3Y": "3 ans",
    "5Y": "5 ans",
    "7Y": "7 ans",
    "10Y": "10 ans",
    "20Y": "20 ans",
    "30Y": "30 ans",
    "4W": "4 semaines",
    "6W": "6 semaines",
    "8W": "8 semaines",
    "13W": "13 semaines",
    "17W": "17 semaines",
    "26W": "26 semaines",
    "52W": "52 semaines",
}
MATURITY_ORDER = list(MATURITY_LABELS)
FAMILY_LABELS = {"nominal_coupon": "Notes et bonds", "bill": "Bills"}
TARGET_LABELS = {
    "btc": "Bid-to-cover",
    "dealer_share": "Part des dealers",
    "indirect_share": "Part des indirects",
    "tail_bp": "Tail (écart au taux de la veille, pb)",
}
WEAK, STRONG, NORMAL = "plus faible", "plus forte", "dans la norme"
SOURCE_LABELS = {
    "fiscaldata_auctions": "Fiscal Data : adjudications",
    "treasurydirect_recent": "TreasuryDirect : résultats récents",
    "treasurydirect_upcoming": "TreasuryDirect : calendrier",
    "treasury_yield_curve": "Treasury.gov : courbe des taux",
    "treasury_bill_rates": "Treasury.gov : taux des bills",
    "cboe_vix": "CBOE : VIX",
    "fiscaldata_debt": "Fiscal Data : dette",
    "fiscaldata_cash_balance": "Fiscal Data : trésorerie",
    "nyfed_dealer_positions": "New York Fed : positions des dealers",
    "cftc_positions": "CFTC : positions sur les futures",
    "fred_series": "FRED : séries macroéconomiques",
}
EXCLUSION_LABELS = {
    "type_inconnu": "Type de titre inconnu",
    "petite_emission_hors_calendrier": "Petite émission hors calendrier (moins de 1 Md$)",
    "passee_sans_resultat": "Date passée sans résultat",
    "future_hors_calendrier": "Date future absente du calendrier",
}


def fr(value: float, decimals: int = 2, sign: bool = False, suffix: str = "") -> str:
    """Nombre à la française : virgule décimale, espace des milliers, vrai signe moins."""
    if value is None or pd.isna(value):
        return "–"
    text = f"{value:{'+' if sign else ''},.{decimals}f}"
    text = text.replace(",", " ").replace(".", ",").replace("-", "−")
    return text + suffix


def auction_title(bucket: str, family: str, reopening: bool) -> str:
    """« 10 ans, réouverture ». Pour les bills, la réouverture n'apporte rien : non affichée."""
    label = MATURITY_LABELS.get(bucket, bucket)
    if family == "nominal_coupon" and bool(reopening):
        return f"{label}, réouverture"
    return label


def signal(gap: float, threshold: float = SIGNAL_THRESHOLD) -> str:
    """Signal sur l'écart non arrondi entre le bid-to-cover prévu et mean3."""
    if pd.isna(gap):
        return "–"
    if gap <= -threshold:
        return WEAK
    if gap >= threshold:
        return STRONG
    return NORMAL


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
    wide = rows.pivot_table(
        index=["cusip", "auction_date", "family", "maturity_bucket"],
        columns="target",
        values=["pred", "bench_mean3"],
        aggfunc="first",
    )
    wide.columns = [f"{value}_{target}" for value, target in wide.columns]
    wide = wide.reset_index()
    extra = rows.groupby(["cusip", "auction_date"], as_index=False).agg(
        offering_amt=("offering_amt", "first"),
        offering_estimated=("offering_estimated", "max"),
        predicted_at=("predicted_at", "max"),
    )
    info = auctions[["cusip", "auction_date", "reopening", "security_term"]].astype(
        {"cusip": "string", "auction_date": "datetime64[us]"}
    )
    wide = wide.astype({"cusip": "string", "auction_date": "datetime64[us]"})
    extra = extra.astype({"cusip": "string", "auction_date": "datetime64[us]"})
    table = wide.merge(extra, on=["cusip", "auction_date"]).merge(
        info, on=["cusip", "auction_date"], how="left"
    )
    table["title"] = [
        auction_title(bucket, family, reopening)
        for bucket, family, reopening in zip(
            table["maturity_bucket"], table["family"], table["reopening"].fillna(False), strict=True
        )
    ]
    table["gap_btc"] = table["pred_btc"] - table["bench_mean3_btc"]
    table["signal"] = table["gap_btc"].map(signal)
    return table.sort_values(["auction_date", "family", "maturity_bucket"], ignore_index=True)


def signal_track_record(
    backtest: pd.DataFrame, threshold: float = SIGNAL_THRESHOLD
) -> pd.DataFrame:
    """Fiabilité passée du signal dans le backtest (moyenne des 5, bid-to-cover, 2014-2026).

    Pour chaque famille : à quelle fréquence le signal se déclenche, et dans quelle part des
    cas le bid-to-cover réel est ensuite du même côté de mean3.
    """
    data = backtest[
        (backtest["model"] == md.ENSEMBLE)
        & (backtest["target"] == "btc")
        & (backtest["feature_set"] == "toutes")
    ].dropna(subset=["actual", "prediction", "mean3"])
    rows = []
    for family, part in data.groupby("family"):
        gap = part["prediction"] - part["mean3"]
        below = part["actual"] < part["mean3"]
        weak, strong = gap <= -threshold, gap >= threshold
        rows.append(
            {
                "family": family,
                "auctions": len(part),
                "share_weak": weak.mean(),
                "weak_then_below": below[weak].mean() if weak.any() else np.nan,
                "share_strong": strong.mean(),
                "strong_then_above": (part["actual"] > part["mean3"])[strong].mean()
                if strong.any()
                else np.nan,
                "base_rate_below": below.mean(),
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
    """Prévisions réalisées pour une cible, avec le gain cumulé contre mean3."""
    done = tracking[
        (tracking["model"] == model)
        & (tracking["target"] == target)
        & (tracking["status"] == "realisee")
    ].dropna(subset=["actual"])
    done = done.sort_values(["auction_date", "cusip"], kind="stable").reset_index(drop=True)
    done["gain"] = (done["bench_error"] ** 2 - done["error"] ** 2).cumsum()
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
    features: pd.DataFrame, backtest: pd.DataFrame, family: str, bucket: str, target: str
) -> pd.DataFrame:
    """Historique d'une maturité : valeur réelle, mean3 et prévision du backtest (moyenne des 5)."""
    history = features[
        (features["family"] == family)
        & (features["maturity_bucket"] == bucket)
        & (features["status"] == "completed")
    ][["cusip", "auction_date", target, f"{target}_mean3"]]
    history = history.rename(columns={target: "actual", f"{target}_mean3": "mean3"})
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
    """Une ligne par source brute : statut, lignes, dernière date, retard, alertes."""
    rows = []
    for name, source in quality["sources"].items():  # type: ignore[union-attr]
        rows.append(
            {
                "source": name,
                "status": source["status"],
                "rows": source["rows"],
                "last_date": source.get("last_date"),
                "business_days_since_last": source.get("business_days_since_last"),
                "alerts": "; ".join(source.get("alerts", [])) or "aucune",
            }
        )
    return pd.DataFrame(rows)


def backtest_r2(metrics: pd.DataFrame, period: str = "2020-2026") -> pd.DataFrame:
    """R² de la moyenne des 5 contre mean3 (et intervalle) par famille et cible."""
    data = metrics[
        (metrics["model"] == md.ENSEMBLE)
        & (metrics["feature_set"] == "toutes")
        & (metrics["period"] == period)
    ]
    return data[["family", "target", "r2_vs_mean3", "r2_ci_low", "r2_ci_high", "direction"]]
