"""Données de l'application, préparées sans Streamlit (donc testables).

Chaque fonction prend des tables déjà lues et renvoie une table prête à afficher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tresor import models as md
from tresor.config import MODEL_SAMPLE_START, SIGNAL_THRESHOLD

SAMPLE_START = pd.Timestamp(MODEL_SAMPLE_START)

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
SHORT_TARGETS = {
    "btc": "bid-to-cover",
    "dealer_share": "part des dealers",
    "indirect_share": "part des indirects",
    "tail_bp": "tail",
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
# Noms des lignes reçues dans le rapport de nettoyage des enchères.
INPUT_LABELS = {
    "fiscaldata": "Fiscal Data",
    "treasurydirect": "TreasuryDirect (récent)",
    "treasurydirect_upcoming": "TreasuryDirect (calendrier)",
}
EXCLUSION_LABELS = {
    "type_inconnu": "Type de titre inconnu",
    "petite_emission_hors_calendrier": "Petite émission hors calendrier (moins de 1 Md$)",
    "passee_sans_resultat": "Date passée sans résultat",
    "future_hors_calendrier": "Date future absente du calendrier",
}


def fr(value: float, decimals: int = 2, sign: bool = False, suffix: str = "") -> str:
    """Nombre à la française : virgule décimale, espace insécable des milliers, vrai signe
    moins ; « – » si la valeur manque. Un zéro arrondi ne garde pas de signe moins."""
    if value is None or pd.isna(value):
        return "–"
    value = float(value)
    if round(value, decimals) == 0:
        value = 0.0
    text = f"{value:{'+' if sign else ''},.{decimals}f}"
    text = text.replace(",", " ").replace(".", ",").replace("-", "−")
    if suffix.startswith(" "):
        suffix = " " + suffix[1:]  # pas de retour à la ligne avant « % » ou « pb »
    return text + suffix


def format_target(value: float, target: str, sign: bool = False) -> str:
    """Valeur d'une cible dans son unité : bid-to-cover à 2 décimales, parts en %, tail en pb."""
    if target in ("dealer_share", "indirect_share"):
        return fr(value * 100, 1, sign=sign, suffix=" %")
    if target == "tail_bp":
        return fr(value, 1, sign=True, suffix=" pb")
    return fr(value, 2, sign=sign)


def auction_title(bucket: str, family: str, reopening: bool) -> str:
    """« 10 ans, réouverture ». Pour les bills, la réouverture n'apporte rien : non affichée."""
    label = MATURITY_LABELS.get(bucket, bucket)
    if family == "nominal_coupon" and bool(reopening):
        return f"{label}, réouverture"
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
                "alerts": "; ".join(source.get("alerts", [])) or "aucune",
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
