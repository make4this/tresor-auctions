"""Lien avec le marché et stratégie illustrative (section 9 du plan).

1. Analyse descriptive : les enchères faibles font-elles monter les taux le jour même ?
2. Stratégie : vendre de la duration de la clôture de t-1 à la clôture de t quand le modèle
   prévoit une enchère faible (bid-to-cover sous mean3), en acheter sinon.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tresor import features as ft
from tresor import models as md
from tresor.config import CLEAN_DIR, OUTPUT_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

COST_BP = 0.5  # coût d'un aller-retour, en points de base de taux
# Notes et bonds seulement : la duration d'un bill est trop courte pour ce pari sur les taux.
FAMILY = "nominal_coupon"
SIGNAL_MODEL = md.ENSEMBLE  # modèle de référence fixé à l'avance en phase 6
MAX_GAP = ft.DAILY_MAX_AGE  # clôture précédente au plus 10 jours avant l'enchère
QUINTILES = 5
# Surprise : écart à la moyenne des 3 dernières enchères. Le signe est choisi pour que le
# quintile 1 soit toujours la demande la plus forte et le quintile 5 la plus faible.
SURPRISES = {
    "part_des_dealers": ("dealer_share", 1),  # dealers servis plus que d'habitude : demande faible
    "bid_to_cover": ("btc", -1),  # bid-to-cover plus bas que d'habitude : demande faible
}
RULES = {
    "strategie": "Stratégie (moyenne des 5)",
    "toujours_vendeur": "Toujours vendeur",
    "information_parfaite": "Information parfaite (impossible en pratique)",
}


def rate_changes(auctions: pd.DataFrame, market: pd.DataFrame) -> pd.Series:
    """Variation du taux de la maturité de chaque enchère, de la clôture de t-1 à celle de t (pb).

    `auctions` doit avoir les colonnes `auction_date` et `maturity_bucket`. Une variation manque
    si le taux du jour ou celui de la veille (au plus 10 jours avant) n'est pas publié.
    """
    result = pd.Series(np.nan, index=auctions.index, dtype=float)
    for bucket, part in auctions.groupby("maturity_bucket"):
        series = ft.RATE_SERIES[bucket]
        rates = market.loc[market["series"] == series, ["date", "value"]].dropna()
        if rates["date"].duplicated().any():
            raise ValueError(f"{series} : plusieurs taux pour une même date")
        # Même précision des dates des deux côtés, sinon la jointure refuse.
        rates = rates.astype({"date": "datetime64[ns]"}).sort_values("date")
        left = part[["auction_date"]].astype("datetime64[ns]").reset_index()
        left = left.sort_values("auction_date")
        before = pd.merge_asof(
            left,
            rates.rename(columns={"date": "previous_date", "value": "previous"}),
            left_on="auction_date",
            right_on="previous_date",
            allow_exact_matches=False,  # strictement avant le jour de l'enchère
            tolerance=MAX_GAP,
        )
        same_day = left.merge(
            rates.rename(columns={"date": "auction_date", "value": "close"}),
            on="auction_date",
            how="left",
        )
        change = (same_day["close"].to_numpy() - before["previous"].to_numpy()) * 100
        result.loc[left["index"].to_numpy()] = change
    return result


def quintile_table(sample: pd.DataFrame) -> pd.DataFrame:
    """Variation moyenne du taux par quintile de surprise, avec intervalle à 90 %.

    Analyse descriptive, après coup : les bornes des quintiles sont calculées sur tout
    l'échantillon, ce qui est sans danger ici puisque rien n'est prévu.
    """
    rows = []
    for name, (target, sign) in SURPRISES.items():
        data = sample.assign(surprise=sample[target] - sample[f"{target}_mean3"])
        data = data.dropna(subset=["surprise", "rate_change_bp"])
        data["quintile"] = pd.qcut(sign * data["surprise"], QUINTILES, labels=False) + 1
        for quintile, part in data.groupby("quintile"):
            change = part["rate_change_bp"]
            half_width = 1.645 * change.std() / np.sqrt(len(change))
            rows.append(
                {
                    "surprise": name,
                    "quintile": int(quintile),
                    "n": len(part),
                    "surprise_mean": part["surprise"].mean(),
                    "rate_change_mean_bp": change.mean(),
                    "ci_low": change.mean() - half_width,
                    "ci_high": change.mean() + half_width,
                    "share_rate_up": (change > 0).mean(),
                }
            )
    return pd.DataFrame(rows)


def signal_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Prévisions hors échantillon du bid-to-cover des notes et bonds par la moyenne des 5."""
    return predictions[
        (predictions["family"] == FAMILY)
        & (predictions["target"] == "btc")
        & (predictions["model"] == SIGNAL_MODEL)
        & (predictions["feature_set"] == "toutes")
    ]


def strategy_trades(rows: pd.DataFrame, changes: pd.Series) -> pd.DataFrame:
    """Une ligne par enchère : position et P&L (pb) de chaque règle.

    Position +1 : vendeur de duration (on gagne si le taux monte) ; -1 : acheteur.
    La stratégie n'utilise que la prévision et mean3, tous deux connus avant l'enchère.
    """
    trades = rows.assign(rate_change_bp=changes.reindex(rows.index)).dropna(
        subset=["prediction", "mean3", "actual", "rate_change_bp"]
    )
    trades = trades.sort_values(["auction_date", "cusip"], kind="stable").reset_index(drop=True)
    move = trades["rate_change_bp"]
    position = np.where(trades["prediction"] < trades["mean3"], 1, -1)
    perfect = np.where(trades["actual"] < trades["mean3"], 1, -1)  # connaît le résultat : triche
    return pd.DataFrame(
        {
            "auction_date": trades["auction_date"],
            "cusip": trades["cusip"],
            "maturity_bucket": trades["maturity_bucket"],
            "rate_change_bp": move,
            "position": position,
            "strategie": position * move - COST_BP,
            "toujours_vendeur": move - COST_BP,
            "information_parfaite": perfect * move - COST_BP,
        }
    )


def _mean_interval(pnl: pd.Series, dates: pd.Series, draws: int = 2000) -> tuple[float, float]:
    """Intervalle à 90 % du P&L moyen par trade, par bootstrap d'années entières."""
    yearly = pd.DataFrame({"pnl": pnl, "year": dates.dt.year}).groupby("year")["pnl"]
    sums, counts = yearly.sum().to_numpy(), yearly.size().to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    draw = rng.integers(0, len(sums), size=(draws, len(sums)))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    low, high = np.quantile(means, [0.05, 0.95])
    return float(low), float(high)


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    """Nombre de trades, P&L moyen et son intervalle, taux de réussite, Sharpe annualisé.

    Le Sharpe est calculé sur le P&L total de chaque jour d'enchère (deux enchères le même jour
    forment un seul portefeuille), annualisé par la racine du nombre de jours d'enchère par an.
    """
    rows = []
    for period in md.PERIODS:
        part = md.in_period(trades, period)
        years = part["auction_date"].dt.year.nunique()
        for rule in RULES:
            daily = part.groupby("auction_date")[rule].sum()
            low, high = _mean_interval(part[rule], part["auction_date"])
            rows.append(
                {
                    "period": period,
                    "rule": rule,
                    "trades": len(part),
                    "mean_bp": part[rule].mean(),
                    "mean_ci_low": low,
                    "mean_ci_high": high,
                    "hit_rate": (part[rule] > 0).mean(),
                    "total_bp": part[rule].sum(),
                    "sharpe": daily.mean() / daily.std() * np.sqrt(len(daily) / years),
                    "share_short": (part["position"] == 1).mean() if rule == "strategie" else None,
                }
            )
    return pd.DataFrame(rows)


def run_strategy(
    clean_dir: Path = CLEAN_DIR, output_dir: Path = OUTPUT_DIR, figures: bool = True
) -> pd.DataFrame:
    """Quintiles, stratégie, tables et figures de la phase 7."""
    market = pd.read_parquet(clean_dir / "market.parquet")
    sample = ft.model_sample(pd.read_parquet(clean_dir / "features.parquet"))
    sample = sample[sample["family"] == FAMILY].copy()
    sample["rate_change_bp"] = rate_changes(sample, market)
    quintiles = quintile_table(sample)

    predictions = pd.read_parquet(output_dir / "backtest_predictions.parquet")
    rows = signal_rows(predictions)
    trades = strategy_trades(rows, rate_changes(rows, market))
    summary = summarize(trades)
    logger.info("Stratégie : %d trades, %d variations de taux manquantes", len(trades),
                int(sample["rate_change_bp"].isna().sum()))  # fmt: skip

    output_dir.mkdir(parents=True, exist_ok=True)
    quintiles.to_csv(output_dir / "quintiles.csv", index=False)
    trades.to_parquet(output_dir / "strategy_trades.parquet", index=False)
    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    if figures:
        from tresor import plots

        plots.quintiles(quintiles)
        plots.strategy_pnl(trades)
    return summary
