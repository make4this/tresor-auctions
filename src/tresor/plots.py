"""Figures pour l'oral et le rapport (reports/figures/), en français."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # aucune fenêtre : les figures sont écrites dans des fichiers

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from tresor.config import FIGURES_DIR  # noqa: E402

# Palette de référence validée (daltonisme, contraste) ; le gris sert aux éléments secondaires.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
PRIMARY = "#2a78d6"
SECOND = "#eb6834"
MUTED = "#c3c2b7"

FAMILY_LABELS = {"nominal_coupon": "Notes et bonds", "bill": "Bills"}
TARGET_LABELS = {
    "btc": "Bid-to-cover",
    "dealer_share": "Part des dealers",
    "indirect_share": "Part des indirects",
    "tail_bp": "Tail (pb)",
}
# Ordre d'affichage : les 5 modèles, la référence fixée à l'avance, les variantes, les benchmarks.
MODEL_LABELS = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "ElasticNet",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "moyenne_5": "Moyenne des 5",
    "ridge_ecart": "Ridge, en écart",
    "lasso_ecart": "Lasso, en écart",
    "elasticnet_ecart": "ElasticNet, en écart",
    "constante": "Constante",
    "naive": "Naïf (dernière)",
}
SHORT_TARGETS = {
    "btc": "Bid-to-cover",
    "dealer_share": "Part dealers",
    "indirect_share": "Part indirects",
    "tail_bp": "Tail",
}
FEATURE_LABELS = {
    "maturity_years": "Maturité (années)",
    "is_reopening": "Réouverture",
    "log_offering": "Montant (log)",
    "offering_change": "Variation du montant",
    "days_since_previous": "Jours depuis la précédente",
    "weekday": "Jour de la semaine",
    "refunding_month": "Mois de refinancement",
    "days_to_quarter_end": "Jours avant fin de trimestre",
    "is_quarter_end": "Fin de trimestre",
    "recent_relative_btc": "Demande récente (famille)",
    "slope_10y_2y": "Pente 10 ans - 2 ans",
    "rate_3m": "Taux 3 mois",
    "vix": "VIX",
    "vix_chg_5d": "VIX, variation 5 j",
    "dollar_chg_20d": "Dollar, variation 20 j",
    "credit_spread_baa": "Spread Baa",
    "breakeven_10y": "Inflation anticipée 10 ans",
    "rate_level": "Taux de la maturité",
    "rate_chg_5d": "Taux, variation 5 j",
    "rate_chg_20d": "Taux, variation 20 j",
    "rate_vol_20d": "Volatilité du taux 20 j",
    "debt_chg_30d": "Dette, variation 30 j",
    "tga_chg_4w_bn": "Solde du Trésor, var. 4 sem.",
    "rrp_chg_4w_bn": "Reverse repo, var. 4 sem.",
    "fed_assets_chg_13w": "Bilan de la Fed, var. 13 sem.",
    "dealer_total_z52": "Dealers, total (normalisé)",
    "dealer_net_z52": "Dealers, tranche (normalisé)",
    "dealer_net_chg_4w_bn": "Dealers, tranche, var. 4 sem.",
    "cftc_am_net_z52": "Asset managers (CFTC)",
    "cftc_lev_net_z52": "Leveraged funds (CFTC)",
}
LAG_LABELS = {
    "lag1": "enchère -1",
    "lag2": "enchère -2",
    "lag3": "enchère -3",
    "mean3": "moyenne des 3",
}
# Crises du plafond de la dette, visibles sur les bills courts.
DEBT_CEILING = {"oct. 2015": "2015-10-15", "oct. 2021": "2021-10-01", "mai 2023": "2023-05-15"}
AXIS_FLOOR = -0.30  # au-delà, la barre est coupée et sa valeur écrite


def feature_label(name: str) -> str:
    """Nom lisible d'une variable, en français."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    target, _, kind = name.rpartition("_")
    if target in SHORT_TARGETS and kind in LAG_LABELS:
        return f"{SHORT_TARGETS[target]} ({LAG_LABELS[kind]})"
    return name


def _style(axis: plt.Axes, grid_axis: str = "x") -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=10, length=0)
    axis.grid(axis=grid_axis, color=GRID, linewidth=1)
    axis.set_axisbelow(True)


def _percent(value: float) -> str:
    return f"{value * 100:+.1f} %".replace(".", ",")


def _whole_percent(value: float, _position: int) -> str:
    return f"{value * 100:.0f} %"


def _french_decimal(value: float, _position: int) -> str:
    """Graduations à la française : virgule décimale."""
    return f"{value:g}".replace(".", ",")


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def r2_by_model(
    metrics: pd.DataFrame,
    best: pd.DataFrame,
    family: str,
    period: str = "2020-2026",
    directory: Path = FIGURES_DIR,
) -> Path:
    """R² contre mean3 sur la période d'évaluation, avec son intervalle à 90 %.

    En bleu, le modèle choisi sur 2014-2019 (avant d'avoir vu cette période) ; en orange, la
    moyenne des 5 modèles, fixée à l'avance. Toutes les cibles partagent la même échelle.
    """
    data = metrics[
        (metrics["family"] == family)
        & (metrics["feature_set"] == "toutes")
        & (metrics["period"] == period)
    ]
    order = [name for name in MODEL_LABELS if name in set(data["model"])]
    right = max(0.1, data["r2_ci_high"].max() + 0.12)
    figure, axes = plt.subplots(1, 4, figsize=(16, 5.4), sharey=True, facecolor=SURFACE)
    for axis, (target, label) in zip(axes, TARGET_LABELS.items(), strict=True):
        part = data[data["target"] == target].set_index("model").reindex(order)
        chosen = best.set_index(["family", "target"]).loc[(family, target), "model"]
        colors = {name: MUTED for name in order} | {"moyenne_5": SECOND, chosen: PRIMARY}
        for position, name in enumerate(order):
            value, low, high = part.loc[name, ["r2_vs_mean3", "r2_ci_low", "r2_ci_high"]]
            cut = value < AXIS_FLOOR
            axis.barh(position, max(value, AXIS_FLOOR), color=colors[name], height=0.62)
            if not cut:
                axis.plot([max(low, AXIS_FLOOR), high], [position, position],
                          color=TEXT_SECONDARY, linewidth=1)  # fmt: skip
            anchor = AXIS_FLOOR + 0.01 if cut else max(0.0, value, high)  # toujours à droite de 0
            axis.text(
                anchor + 0.01,
                position,
                _percent(value) + (" (coupé)" if cut else ""),
                va="center",
                fontsize=9,
                color=TEXT if name in (chosen, "moyenne_5") else TEXT_SECONDARY,
            )
        axis.axvline(0, color=TEXT_SECONDARY, linewidth=1)
        axis.set_xlim(AXIS_FLOOR, right)
        axis.set_xticks([step / 10 for step in range(-2, int(right * 10) + 1, 2)])
        axis.xaxis.set_major_formatter(_whole_percent)
        axis.set_title(label, fontsize=11, color=TEXT, loc="left")
        _style(axis)
    axes[0].set_yticks(range(len(order)), [MODEL_LABELS[name] for name in order])
    axes[0].invert_yaxis()
    figure.legend(
        handles=[
            Patch(color=PRIMARY, label="Modèle choisi sur 2014-2019"),
            Patch(color=SECOND, label="Moyenne des 5, fixée à l'avance"),
            Patch(color=MUTED, label="Autres modèles et benchmarks"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.01, -0.06),
        ncol=3,
        frameon=False,
        fontsize=10,
        labelcolor=TEXT_SECONDARY,
    )
    figure.suptitle(
        f"{FAMILY_LABELS[family]} : R² hors échantillon contre la moyenne des 3 dernières"
        f" enchères, {period} (trait : intervalle à 90 %, bootstrap par année)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=TEXT,
    )
    figure.tight_layout(w_pad=3)
    return _save(figure, directory / f"r2_par_modele_{family}.png")


def new_sources(comparison: pd.DataFrame, directory: Path = FIGURES_DIR) -> Path:
    """Expérience clé : moyenne des 5 modèles, avec et sans dealers, CFTC et liquidité."""
    rows = comparison[comparison["model"] == "moyenne_5"].reset_index(drop=True)
    labels = [f"{FAMILY_LABELS[r.family]} - {TARGET_LABELS[r.target]}" for r in rows.itertuples()]
    figure, axis = plt.subplots(figsize=(10.5, 5.2), facecolor=SURFACE)
    positions = list(range(len(rows)))
    axis.barh([p - 0.18 for p in positions], rows["toutes"], height=0.34, color=PRIMARY,
              label="Avec les nouvelles sources")  # fmt: skip
    axis.barh([p + 0.18 for p in positions], rows["sans_nouvelles_sources"], height=0.34,
              color=SECOND, label="Sans")  # fmt: skip
    right = rows[["toutes", "sans_nouvelles_sources"]].max().max()
    for position, row in zip(positions, rows.itertuples(), strict=True):
        low, high = row.gain_ci_low * 100, row.gain_ci_high * 100
        text = f"{row.gain * 100:+.1f} pt  [{low:+.1f} ; {high:+.1f}]".replace(".", ",")
        axis.text(right + 0.02, position, text, va="center", fontsize=9, color=TEXT_SECONDARY)
    axis.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    left = min(0, rows[["toutes", "sans_nouvelles_sources"]].min().min()) - 0.02
    axis.set_xlim(left, right + 0.24)
    axis.xaxis.set_major_formatter(_whole_percent)
    axis.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.07),
                ncol=2, labelcolor=TEXT_SECONDARY)  # fmt: skip
    axis.set_title(
        "Les nouvelles sources aident-elles ? Moyenne des 5 modèles, R² contre mean3,"
        " 2014-2026.\nÀ droite : écart avec - sans, et son intervalle à 90 %"
        " (s'il contient 0, pas d'effet net).",
        fontsize=11,
        color=TEXT,
        loc="left",
    )
    _style(axis)
    figure.tight_layout()
    return _save(figure, directory / "nouvelles_sources.png")


def cumulative_gain(
    predictions: pd.DataFrame,
    family: str,
    target: str,
    model: str = "moyenne_5",
    directory: Path = FIGURES_DIR,
) -> Path:
    """Gain cumulé contre mean3, rapporté à l'erreur totale de mean3 : la courbe finit au R²."""
    data = predictions[
        (predictions["family"] == family)
        & (predictions["target"] == target)
        & (predictions["model"] == model)
        & (predictions["feature_set"] == "toutes")
    ].dropna(subset=["actual", "prediction", "mean3", "naive", "constant"])
    data = data.sort_values("auction_date")
    reference = (data["actual"] - data["mean3"]) ** 2
    gain = (reference - (data["actual"] - data["prediction"]) ** 2).cumsum() / reference.sum()
    figure, axis = plt.subplots(figsize=(10.5, 4), facecolor=SURFACE)
    axis.plot(data["auction_date"], gain, color=PRIMARY, linewidth=2, solid_capstyle="round")
    axis.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    if family == "bill":
        for label, day in DEBT_CEILING.items():
            axis.axvline(pd.Timestamp(day), color=MUTED, linewidth=1, linestyle="--")
            axis.annotate(f"plafond\n{label}", (pd.Timestamp(day), gain.min()), xytext=(5, 0),
                          textcoords="offset points", fontsize=8, color=TEXT_SECONDARY,
                          va="bottom")  # fmt: skip
    axis.annotate(
        _percent(gain.iloc[-1]),
        (data["auction_date"].iloc[-1], gain.iloc[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=10,
        color=TEXT,
    )
    axis.yaxis.set_major_formatter(_whole_percent)
    axis.set_title(
        f"{FAMILY_LABELS[family]}, {TARGET_LABELS[target].lower()} : gain cumulé de la"
        f" {MODEL_LABELS[model].lower()} contre mean3, en part de l'erreur totale de mean3\n"
        "(au-dessus de 0 : le modèle fait mieux ; la courbe finit au R² de 2014-2026)"
        + ("\nPointillés : crises du plafond de la dette." if family == "bill" else ""),
        fontsize=11,
        color=TEXT,
        loc="left",
    )
    _style(axis, grid_axis="y")
    figure.tight_layout()
    return _save(figure, directory / f"gain_cumule_{family}_{target}.png")


def importance(
    table: pd.DataFrame, family: str, target: str, directory: Path = FIGURES_DIR, top: int = 12
) -> Path:
    """Ce qui aide vraiment (permutation, hors échantillon) et ce qui pèse dans le modèle (SHAP)."""
    data = table[(table["family"] == family) & (table["target"] == target)]
    model = MODEL_LABELS[data["model"].iloc[0]]
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE)
    permutation = data.nlargest(top, "permutation_mean").iloc[::-1]
    left.barh([feature_label(n) for n in permutation["feature"]], permutation["permutation_mean"],
              xerr=permutation["permutation_std"], color=PRIMARY, height=0.6,
              error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1})  # fmt: skip
    left.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    left.xaxis.set_major_formatter(_french_decimal)
    left.set_title("Permutation : hausse de l'erreur quadratique si la\nvariable est brouillée,"
                   " 2022-2026 (ce qui aide vraiment)", fontsize=10, color=TEXT,
                   loc="left")  # fmt: skip
    shap_part = data.nlargest(top, "shap_share").iloc[::-1]
    right.barh([feature_label(n) for n in shap_part["feature"]], shap_part["shap_share"],
               color=MUTED, height=0.6)  # fmt: skip
    right.xaxis.set_major_formatter(_whole_percent)
    right.set_title("SHAP : part de chaque variable dans la formule\ndu modèle"
                    " (ce qui pèse, pas forcément ce qui aide)", fontsize=10, color=TEXT,
                    loc="left")  # fmt: skip
    for axis in (left, right):
        _style(axis)
    figure.suptitle(
        f"{FAMILY_LABELS[family]}, {TARGET_LABELS[target].lower()} : importance des variables"
        f" pour {model} (choisi sur 2014-2019), entraîné avant 2022, expliqué sur 2022-2026",
        x=0.01,
        ha="left",
        fontsize=12,
        color=TEXT,
    )
    figure.tight_layout()
    return _save(figure, directory / f"importance_{family}_{target}.png")


def all_figures(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    best: pd.DataFrame,
    importance_table: pd.DataFrame,
    directory: Path = FIGURES_DIR,
) -> list[Path]:
    """Toutes les figures de la phase 6."""
    for old in directory.glob("shap_*.png"):  # anciennes figures, remplacées par importance_*
        old.unlink()
    paths = [r2_by_model(metrics, best, family, directory=directory) for family in FAMILY_LABELS]
    paths.append(new_sources(comparison, directory))
    for family in FAMILY_LABELS:
        paths.append(cumulative_gain(predictions, family, "btc", directory=directory))
        paths.append(importance(importance_table, family, "btc", directory))
    return paths
