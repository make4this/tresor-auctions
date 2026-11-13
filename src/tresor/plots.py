"""Figures pour l'oral et le rapport (reports/figures/), en français."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # aucune fenêtre : les figures sont écrites dans des fichiers

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

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
MODEL_LABELS = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "ElasticNet",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "naive": "Naïf (dernière)",
}
AXIS_FLOOR = -0.30  # au-delà, la barre est coupée et sa valeur écrite


def _style(axis: plt.Axes) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    axis.grid(axis="x", color=GRID, linewidth=1)
    axis.set_axisbelow(True)


def _percent(value: float) -> str:
    return f"{value * 100:+.1f} %".replace(".", ",")


def _french_decimal(value: float, _position: int) -> str:
    """Graduations à la française : virgule décimale."""
    return f"{value:g}".replace(".", ",")


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def r2_by_model(metrics: pd.DataFrame, family: str, directory: Path = FIGURES_DIR) -> Path:
    """R² hors échantillon contre mean3, par cible et par modèle ; le meilleur en couleur."""
    data = metrics[(metrics["family"] == family) & (metrics["feature_set"] == "toutes")]
    order = [name for name in MODEL_LABELS if name in set(data["model"])]
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, facecolor=SURFACE)
    for axis, (target, label) in zip(axes, TARGET_LABELS.items(), strict=True):
        part = data[data["target"] == target].set_index("model").reindex(order)
        values = part["r2_vs_mean3"]
        best = values.drop("naive", errors="ignore").idxmax()
        colors = [PRIMARY if name == best else MUTED for name in order]
        shown = values.clip(lower=AXIS_FLOOR)
        axis.barh(range(len(order)), shown, color=colors, height=0.6)
        for position, (value, kept) in enumerate(zip(values, shown, strict=True)):
            inside = value < AXIS_FLOOR
            anchor = AXIS_FLOOR + 0.01 if inside else kept
            axis.text(
                anchor + (0.005 if kept >= 0 or inside else -0.005),
                position,
                _percent(value) + (" (coupé)" if inside else ""),
                va="center",
                ha="left" if kept >= 0 or inside else "right",
                fontsize=8,
                color=TEXT if order[position] == best else TEXT_SECONDARY,
            )
        axis.axvline(0, color=TEXT_SECONDARY, linewidth=1)
        axis.set_xlim(AXIS_FLOOR, max(0.05, values.max() + 0.12))
        axis.xaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f} %")
        axis.set_title(label, fontsize=10, color=TEXT, loc="left")
        _style(axis)
    axes[0].set_yticks(range(len(order)), [MODEL_LABELS[name] for name in order])
    axes[0].invert_yaxis()
    figure.suptitle(
        f"{FAMILY_LABELS[family]} : R² hors échantillon contre la moyenne des 3 dernières"
        " enchères (walk-forward 2014-2026)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=TEXT,
    )
    figure.tight_layout()
    return _save(figure, directory / f"r2_par_modele_{family}.png")


def new_sources(
    comparison: pd.DataFrame, best: pd.DataFrame, directory: Path = FIGURES_DIR
) -> Path:
    """Expérience clé : R² du meilleur modèle avec et sans dealers, CFTC et liquidité."""
    rows = best.merge(comparison, on=["family", "target", "model"])
    labels = [f"{FAMILY_LABELS[r.family]} - {TARGET_LABELS[r.target]}" for r in rows.itertuples()]
    figure, axis = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    positions = range(len(rows))
    axis.barh([p - 0.18 for p in positions], rows["toutes"], height=0.34, color=PRIMARY,
              label="Avec les nouvelles sources")  # fmt: skip
    axis.barh([p + 0.18 for p in positions], rows["sans_nouvelles_sources"], height=0.34,
              color=SECOND, label="Sans")  # fmt: skip
    right = max(rows["toutes"].max(), rows["sans_nouvelles_sources"].max())
    for position, gain in zip(positions, rows["gain"], strict=True):
        axis.text(right + 0.02, position, f"écart {gain * 100:+.1f} pt".replace(".", ","),
                  va="center", fontsize=8, color=TEXT_SECONDARY)  # fmt: skip
    axis.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    axis.set_yticks(list(positions), labels)
    axis.invert_yaxis()
    axis.set_xlim(
        min(0, rows[["toutes", "sans_nouvelles_sources"]].min().min()) - 0.02, right + 0.14
    )
    axis.xaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f} %")
    # Légende sous le graphique : à droite, elle masquait l'écart de la dernière ligne.
    axis.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        labelcolor=TEXT_SECONDARY,
    )
    axis.set_title(
        "Les nouvelles sources aident-elles ? R² contre mean3 du meilleur modèle",
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
    model: str,
    directory: Path = FIGURES_DIR,
) -> Path:
    """Gain cumulé contre mean3 : somme de (réel - mean3)² - (réel - prévu)², date par date."""
    data = predictions[
        (predictions["family"] == family)
        & (predictions["target"] == target)
        & (predictions["model"] == model)
        & (predictions["feature_set"] == "toutes")
    ].dropna(subset=["actual", "prediction", "mean3", "naive"])
    data = data.sort_values("auction_date")
    gain = (
        (data["actual"] - data["mean3"]) ** 2 - (data["actual"] - data["prediction"]) ** 2
    ).cumsum()
    figure, axis = plt.subplots(figsize=(8.5, 3.6), facecolor=SURFACE)
    axis.plot(data["auction_date"], gain, color=PRIMARY, linewidth=2, solid_capstyle="round")
    axis.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    axis.annotate(
        f"{gain.iloc[-1]:,.1f}".replace(",", " ").replace(".", ","),
        (data["auction_date"].iloc[-1], gain.iloc[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=TEXT,
    )
    axis.grid(axis="y", color=GRID, linewidth=1)
    axis.yaxis.set_major_formatter(_french_decimal)
    axis.set_title(
        f"{FAMILY_LABELS[family]}, {TARGET_LABELS[target].lower()} : gain cumulé de"
        f" {MODEL_LABELS[model]} contre mean3 (au-dessus de 0 : le modèle fait mieux)",
        fontsize=10,
        color=TEXT,
        loc="left",
    )
    _style(axis)
    axis.grid(axis="x", visible=False)
    figure.tight_layout()
    return _save(figure, directory / f"gain_cumule_{family}_{target}.png")


def shap_importance(
    importance: pd.DataFrame, family: str, target: str, directory: Path = FIGURES_DIR, top: int = 12
) -> Path:
    """Les variables qui comptent le plus pour le meilleur modèle (SHAP moyen absolu)."""
    data = importance[(importance["family"] == family) & (importance["target"] == target)]
    data = data.nlargest(top, "shap_mean_abs").iloc[::-1]
    model = MODEL_LABELS[data["model"].iloc[0]]
    figure, axis = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    axis.barh(data["feature"], data["shap_mean_abs"], color=PRIMARY, height=0.6)
    axis.xaxis.set_major_formatter(_french_decimal)
    axis.set_title(
        f"{FAMILY_LABELS[family]}, {TARGET_LABELS[target].lower()} : contribution moyenne"
        f" (|SHAP|) de chaque variable, {model}, 2022-2026",
        fontsize=10,
        color=TEXT,
        loc="left",
    )
    _style(axis)
    figure.tight_layout()
    return _save(figure, directory / f"shap_{family}_{target}.png")


def all_figures(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    best: pd.DataFrame,
    importance: pd.DataFrame,
    directory: Path = FIGURES_DIR,
) -> list[Path]:
    """Toutes les figures de la phase 6."""
    paths = [r2_by_model(metrics, family, directory) for family in FAMILY_LABELS]
    paths.append(new_sources(comparison, best, directory))
    for row in best[best["target"] == "btc"].itertuples():
        paths.append(cumulative_gain(predictions, row.family, row.target, row.model, directory))
        paths.append(shap_importance(importance, row.family, row.target, directory))
    return paths
