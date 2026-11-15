"""Lien avec le marché et stratégie illustrative (section 9 du plan).

1. Analyse descriptive : les enchères faibles s'accompagnent-elles d'une hausse des taux ?
2. Stratégie : vendre de la duration de la clôture de t-1 à la clôture de t quand le modèle
   prévoit une enchère faible (bid-to-cover sous mean3), en acheter sinon.

Chaque chiffre cité dans le journal est écrit dans un fichier de `data/output/`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tresor import features as ft
from tresor import models as md
from tresor.clean.market import _auction_days
from tresor.config import CLEAN_DIR, OUTPUT_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

COST_BP = 0.5  # coût d'un aller-retour, en points de base de taux
COSTS_BP = (0.0, 0.25, 0.5, 1.0)  # sensibilité au coût
# Notes et bonds seulement : la duration d'un bill est trop courte pour ce pari sur les taux.
FAMILY = "nominal_coupon"
SIGNAL_MODEL = md.ENSEMBLE  # modèle de référence retenu en phase 6 (moyenne des 5)
MAX_GAP = ft.DAILY_MAX_AGE  # clôture précédente au plus 10 jours avant l'enchère
QUINTILES = 5
# Surprise : écart à la moyenne des 3 dernières enchères. Le signe est choisi pour que le
# quintile 1 soit toujours la demande la plus forte et le quintile 5 la plus faible.
SURPRISES = {
    "part_des_dealers": ("dealer_share", 1),  # dealers servis plus que d'habitude : demande faible
    "bid_to_cover": ("btc", -1),  # bid-to-cover plus bas que d'habitude : demande faible
}
QUINTILE_PERIODS = {
    "2009-2026": (pd.Timestamp("2009-01-01"), pd.Timestamp("2026-12-31")),
    "2009-2013": (pd.Timestamp("2009-01-01"), pd.Timestamp("2013-12-31")),
    **{name: md.PERIODS[name] for name in ("2014-2019", "2020-2026")},
}
RULES = {
    "strategie": "Stratégie (moyenne des 5)",
    "toujours_vendeur": "Toujours vendeur",
    "information_parfaite": "Information parfaite sur le sens du bid-to-cover",
}
REFERENCE_RULE = "toujours_vendeur"  # la règle à battre, d'après le plan
# Étude d'événement, en jours de cotation : avant (t-6 à t-1), le jour même, après (t à t+3).
EVENT_OFFSETS = {"avant_bp": (-6, -1), "jour_bp": (-1, 0), "apres_bp": (0, 3)}
REGIMES = {
    "2014-2021": (pd.Timestamp("2014-01-01"), pd.Timestamp("2021-12-31")),
    "2022-2026": (pd.Timestamp("2022-01-01"), pd.Timestamp("2026-12-31")),
}


# --- Taux de marché ------------------------------------------------------------------------


def _yields(market: pd.DataFrame, series: str) -> pd.Series:
    """Taux de clôture d'une série, indexés par date (en ns), triés, une valeur par date."""
    rates = market.loc[market["series"] == series, ["date", "value"]].dropna()
    if rates["date"].duplicated().any():
        raise ValueError(f"{series} : plusieurs taux pour une même date")
    dates = pd.DatetimeIndex(rates["date"].to_numpy(dtype="datetime64[ns]"))
    return pd.Series(rates["value"].to_numpy(dtype=float), index=dates).sort_index()


def rate_changes(auctions: pd.DataFrame, market: pd.DataFrame) -> pd.Series:
    """Variation du taux de la maturité de chaque enchère, de la clôture de t-1 à celle de t (pb).

    `auctions` doit avoir les colonnes `auction_date` (jours sans heure) et `maturity_bucket`.
    La veille est la dernière clôture publiée strictement avant t, au plus 10 jours avant.
    """
    result = pd.Series(np.nan, index=auctions.index, dtype=float)
    days = pd.Series(_auction_days(auctions["auction_date"]), index=auctions.index)
    for bucket, part in days.groupby(auctions["maturity_bucket"]):
        rates = _yields(market, ft.RATE_SERIES[bucket])
        dates = part.to_numpy()
        previous = rates.index.searchsorted(dates, side="left") - 1  # strictement avant t
        found = previous >= 0
        previous_date = rates.index[previous.clip(0)]
        fresh = found & ((dates - previous_date) <= MAX_GAP)
        before = np.where(fresh, rates.to_numpy()[previous.clip(0)], np.nan)
        close = rates.reindex(dates).to_numpy()
        result.loc[part.index] = (close - before) * 100
    return result


def event_moves(auctions: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Variations du taux avant, pendant et après l'enchère, en jours de cotation (pb).

    Analyse descriptive, après coup : « après » utilise des taux postérieurs à l'enchère.
    """
    result = pd.DataFrame(np.nan, index=auctions.index, columns=list(EVENT_OFFSETS))
    days = pd.Series(_auction_days(auctions["auction_date"]), index=auctions.index)
    for bucket, part in days.groupby(auctions["maturity_bucket"]):
        rates = _yields(market, ft.RATE_SERIES[bucket])
        values = rates.to_numpy()
        position = rates.index.get_indexer(part.to_numpy())  # -1 : pas de taux le jour même
        for name, (start, end) in EVENT_OFFSETS.items():
            first, last = position + start, position + end
            valid = (position >= 0) & (first >= 0) & (last < len(values))
            move = values[last.clip(0, len(values) - 1)] - values[first.clip(0, len(values) - 1)]
            result.loc[part.index, name] = np.where(valid, move * 100, np.nan)
    return result


# --- Analyse descriptive -------------------------------------------------------------------


def _mean_interval(values: pd.Series, dates: pd.Series, draws: int = 2000) -> tuple[float, float]:
    """Intervalle à 90 % d'une moyenne, par bootstrap d'années entières (dépendance temporelle).

    Avec peu d'années (6 ou 7 par sous-période), l'intervalle est approximatif.
    """
    yearly = pd.DataFrame({"value": values, "year": dates.dt.year}).groupby("year")["value"]
    sums, counts = yearly.sum().to_numpy(), yearly.size().to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    draw = rng.integers(0, len(sums), size=(draws, len(sums)))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    low, high = np.quantile(means, [0.05, 0.95])
    return float(low), float(high)


def quintile_table(sample: pd.DataFrame) -> pd.DataFrame:
    """Variation moyenne du taux par quintile de surprise et par période, avec intervalle à 90 %.

    Analyse descriptive, après coup : les bornes des quintiles sont calculées sur toute la
    période, ce qui est sans danger ici puisque rien n'est prévu.
    """
    rows = []
    for period, (start, end) in QUINTILE_PERIODS.items():
        in_period = sample[sample["auction_date"].between(start, end)]
        for name, (target, sign) in SURPRISES.items():
            data = in_period.assign(surprise=in_period[target] - in_period[f"{target}_mean3"])
            data = data.dropna(subset=["surprise", "rate_change_bp"])
            if len(data) < QUINTILES:
                continue
            data["quintile"] = pd.qcut(sign * data["surprise"], QUINTILES, labels=False) + 1
            for quintile, part in data.groupby("quintile"):
                change = part["rate_change_bp"]
                low, high = _mean_interval(change, part["auction_date"])
                rows.append(
                    {
                        "period": period,
                        "surprise": name,
                        "quintile": int(quintile),
                        "n": len(part),
                        "surprise_mean": part["surprise"].mean(),
                        "rate_change_mean_bp": change.mean(),
                        "ci_low": low,
                        "ci_high": high,
                        "share_rate_up": (change > 0).mean(),
                    }
                )
    return pd.DataFrame(rows)


# --- Stratégie -----------------------------------------------------------------------------


def signal_rows(
    predictions: pd.DataFrame, model: str = SIGNAL_MODEL, feature_set: str = "toutes"
) -> pd.DataFrame:
    """Prévisions hors échantillon du bid-to-cover des notes et bonds par un modèle donné."""
    return predictions[
        (predictions["family"] == FAMILY)
        & (predictions["target"] == "btc")
        & (predictions["model"] == model)
        & (predictions["feature_set"] == feature_set)
    ]


def strategy_trades(rows: pd.DataFrame, changes: pd.Series) -> pd.DataFrame:
    """Une ligne par enchère : position et P&L (pb, après coût) de chaque règle.

    Position +1 : vendeur de duration (on gagne si le taux monte) ; -1 : acheteur.
    La stratégie n'utilise que la prévision et mean3, tous deux connus avant l'enchère ; le
    résultat réel ne sert qu'à la règle « information parfaite » (égalité : acheteur).
    """
    trades = rows.assign(rate_change_bp=changes.reindex(rows.index)).dropna(
        subset=["prediction", "mean3", "rate_change_bp"]
    )
    trades = trades.sort_values(["auction_date", "cusip"], kind="stable").reset_index(drop=True)
    move = trades["rate_change_bp"]
    position = np.where(trades["prediction"] < trades["mean3"], 1, -1)
    perfect = np.where(trades["actual"] < trades["mean3"], 1.0, -1.0)  # connaît le résultat
    perfect[trades["actual"].isna().to_numpy()] = np.nan
    return pd.DataFrame(
        {
            "auction_date": trades["auction_date"],
            "cusip": trades["cusip"],
            "maturity_bucket": trades["maturity_bucket"],
            "prediction": trades["prediction"],
            "mean3": trades["mean3"],
            "actual": trades["actual"],
            "rate_change_bp": move,
            "position": position,
            "strategie": position * move - COST_BP,
            "toujours_vendeur": move - COST_BP,
            "information_parfaite": perfect * move - COST_BP,
        }
    )


def _years(dates: pd.Series) -> float:
    """Durée couverte, en années (une année partielle compte pour sa part)."""
    return ((dates.max() - dates.min()).days + 1) / 365.25


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    """Par période et par règle : trades, P&L moyen (après et avant coût) et son intervalle,
    réussite, total, Sharpe annualisé, et écart avec « toujours vendeur » et son intervalle.

    Le Sharpe est calculé sur le P&L total de chaque jour d'enchère (deux enchères le même jour
    forment un seul portefeuille), annualisé par la racine du nombre de jours d'enchère par an.
    """
    rows = []
    for period in md.PERIODS:
        part = md.in_period(trades, period)
        if part.empty:
            continue
        days_per_year = part["auction_date"].nunique() / _years(part["auction_date"])
        for rule in RULES:
            daily = part.groupby("auction_date")[rule].sum()
            low, high = _mean_interval(part[rule], part["auction_date"])
            gap = part[rule] - part[REFERENCE_RULE]
            gap_low, gap_high = _mean_interval(gap, part["auction_date"])
            reference = rule == REFERENCE_RULE
            rows.append(
                {
                    "period": period,
                    "rule": rule,
                    "trades": len(part),
                    "mean_bp": part[rule].mean(),
                    "mean_ci_low": low,
                    "mean_ci_high": high,
                    "mean_gross_bp": part[rule].mean() + COST_BP,
                    "hit_rate": (part[rule] > 0).mean(),
                    "total_bp": part[rule].sum(),
                    "cost_total_bp": COST_BP * len(part),
                    "sharpe": daily.mean() / daily.std() * np.sqrt(days_per_year),
                    "share_short": (part["position"] == 1).mean()
                    if rule == "strategie"
                    else np.nan,
                    "vs_vendeur_bp": np.nan if reference else gap.mean(),
                    "vs_vendeur_ci_low": np.nan if reference else gap_low,
                    "vs_vendeur_ci_high": np.nan if reference else gap_high,
                }
            )
    return pd.DataFrame(rows)


def by_year(trades: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par année : position, mouvement moyen des taux et P&L moyen de chaque règle."""
    data = trades.assign(
        year=trades["auction_date"].dt.year,
        short=trades["position"] == 1,
        weak=trades["actual"] < trades["mean3"],
        gross=trades["strategie"] + COST_BP,
    )
    return (
        data.groupby("year")
        .agg(
            trades=("short", "size"),
            share_short=("short", "mean"),
            share_actual_weak=("weak", "mean"),
            rate_change_mean_bp=("rate_change_bp", "mean"),
            strategie_bp=("strategie", "mean"),
            strategie_gross_bp=("gross", "mean"),
            toujours_vendeur_bp=("toujours_vendeur", "mean"),
            information_parfaite_bp=("information_parfaite", "mean"),
        )
        .reset_index()
    )


def cost_sensitivity(trades: pd.DataFrame) -> pd.DataFrame:
    """P&L moyen par trade de chaque règle pour plusieurs coûts, avec intervalle à 90 %."""
    rows = []
    for cost in COSTS_BP:
        for rule in RULES:
            pnl = trades[rule] + COST_BP - cost
            low, high = _mean_interval(pnl, trades["auction_date"])
            rows.append(
                {"cost_bp": cost, "rule": rule, "mean_bp": pnl.mean(), "ci_low": low,
                 "ci_high": high}
            )  # fmt: skip
    return pd.DataFrame(rows)


def signal_robustness(predictions: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """La même stratégie avec chacun des signaux disponibles (vérification, pas sélection)."""
    rows = []
    coupons = predictions[(predictions["family"] == FAMILY) & (predictions["target"] == "btc")]
    for (model, feature_set), _ in coupons.groupby(["model", "feature_set"]):
        signal = signal_rows(predictions, model, feature_set)
        trades = strategy_trades(signal, rate_changes(signal, market))
        gap = trades["strategie"] - trades[REFERENCE_RULE]
        low, high = _mean_interval(gap, trades["auction_date"])
        recent = trades[trades["auction_date"].dt.year.between(2022, 2025)]
        predicted = trades["prediction"] - trades["mean3"]
        rows.append(
            {
                "model": model,
                "feature_set": feature_set,
                "trades": len(trades),
                "mean_bp": trades["strategie"].mean(),
                "mean_gross_bp": trades["strategie"].mean() + COST_BP,
                "vs_vendeur_bp": gap.mean(),
                "vs_vendeur_ci_low": low,
                "vs_vendeur_ci_high": high,
                "share_short_2022_2025": (recent["position"] == 1).mean(),
                "corr_predicted_move": predicted.corr(trades["rate_change_bp"]),
            }
        )
    return pd.DataFrame(rows)


def event_study(trades: pd.DataFrame) -> pd.DataFrame:
    """Mouvement moyen des taux avant, pendant et après l'enchère, par période."""
    rows = []
    for regime, (start, end) in REGIMES.items():
        part = trades[trades["auction_date"].between(start, end)]
        row: dict[str, float | str | int] = {"regime": regime, "trades": len(part)}
        for name in EVENT_OFFSETS:
            row[name] = part[name].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    """Les chiffres d'explication cités dans le journal, recalculables."""
    move = trades["rate_change_bp"]
    predicted = trades["prediction"] - trades["mean3"]
    surprise = trades["actual"] - trades["mean3"]
    year = trades["auction_date"].dt.year
    gross = trades["strategie"] + COST_BP
    inverse_low, inverse_high = _mean_interval(-gross, trades["auction_date"])
    # Si la partie prévue faisait bouger le taux autant que la surprise réelle, combien la
    # stratégie gagnerait-elle avant coût ? (pente : variation du taux par unité de surprise)
    slope = float(np.polyfit(surprise, move, 1)[0])
    within_predicted = predicted - predicted.groupby(year).transform("mean")
    within_move = move - move.groupby(year).transform("mean")
    short, long = trades["position"] == 1, trades["position"] == -1
    values = {
        "corr_surprise_move": (surprise.corr(move), "surprise réelle et variation du taux"),
        "corr_predicted_move": (predicted.corr(move), "partie prévue et variation du taux"),
        "corr_predicted_move_within_year": (
            within_predicted.corr(within_move),
            "même chose, en écart à la moyenne de chaque année",
        ),
        "corr_unexpected_move": ((surprise - predicted).corr(move), "partie non prévue et taux"),
        "slope_bp_per_btc": (slope, "variation du taux par unité de surprise de bid-to-cover"),
        "gain_if_priced_like_surprise_bp": (
            float((trades["position"] * slope * predicted).mean()),
            "gain avant coût si la partie prévue agissait comme la surprise réelle",
        ),
        "strategy_gross_bp": (gross.mean(), "stratégie, avant coût"),
        "inverse_gross_bp": (-gross.mean(), "stratégie inverse, avant coût"),
        "inverse_gross_ci_low": (inverse_low, "intervalle à 90 %, bootstrap par année"),
        "inverse_gross_ci_high": (inverse_high, "intervalle à 90 %, bootstrap par année"),
        "inverse_net_bp": (-gross.mean() - COST_BP, "stratégie inverse, après coût"),
        "share_zero_move_trades": ((move == 0).mean(), "trades dont le taux ne bouge pas"),
        "hit_rate_gross_nonzero": (
            (gross[move != 0] > 0).mean(),
            "réussite avant coût, hors trades à variation nulle",
        ),
        "pre_move_short_bp": (
            trades.loc[short, "avant_bp"].mean(),
            "variation du taux de t-6 à t-1 quand le modèle est vendeur",
        ),
        "pre_move_long_bp": (
            trades.loc[long, "avant_bp"].mean(),
            "variation du taux de t-6 à t-1 quand le modèle est acheteur",
        ),
    }
    return pd.DataFrame(
        [
            {"name": key, "value": value, "description": text}
            for key, (value, text) in values.items()
        ]
    )


def _check_in_sync(rows: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Les prévisions enregistrées doivent correspondre aux variables actuelles."""
    keys = ["cusip", "auction_date"]
    joined = rows[[*keys, "mean3", "actual"]].merge(
        sample[[*keys, "btc", "btc_mean3"]], on=keys, how="left"
    )
    same = np.isclose(joined["mean3"], joined["btc_mean3"], equal_nan=True) & np.isclose(
        joined["actual"], joined["btc"], equal_nan=True
    )
    if not same.all():
        raise ValueError("prévisions et variables désynchronisées : relancer --backtest")


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
    _check_in_sync(rows, sample)
    changes = rate_changes(rows, market)
    trades = strategy_trades(rows, changes)
    trades = pd.concat([trades, event_moves(trades, market)], axis=1)
    summary = summarize(trades)
    logger.info("Stratégie : %d trades sur %d prévisions (%d sans variation de taux)",
                len(trades), len(rows), int(changes.isna().sum()))  # fmt: skip

    output_dir.mkdir(parents=True, exist_ok=True)
    quintiles.to_csv(output_dir / "quintiles.csv", index=False)
    trades.to_parquet(output_dir / "strategy_trades.parquet", index=False)
    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    by_year(trades).to_csv(output_dir / "strategy_by_year.csv", index=False)
    cost_sensitivity(trades).to_csv(output_dir / "strategy_costs.csv", index=False)
    signal_robustness(predictions, market).to_csv(output_dir / "strategy_signals.csv", index=False)
    event_study(trades).to_csv(output_dir / "event_study.csv", index=False)
    diagnostics(trades).to_csv(output_dir / "strategy_diagnostics.csv", index=False)
    if figures:
        from tresor import plots

        plots.quintiles(quintiles[quintiles["period"] == "2009-2026"])
        plots.strategy_pnl(trades, COST_BP)
    return summary
