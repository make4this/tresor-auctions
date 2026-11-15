"""Chiffres clés du rapport et de l'oral, recalculés depuis les fichiers de data/output/.

Chaque chiffre a son fichier source et son calcul. `write` écrit `reports/chiffres_cles.md` ;
`check` recalcule tout et compare au fichier écrit. Deux natures de chiffres :

- « figé » : backtest, lien avec le marché, stratégie. Leurs fichiers ne changent que si l'on
  relance `--backtest --strategy` à la main : un écart est une erreur.
- « du jour » : données et prévisions en direct, mises à jour chaque soir par le passage
  automatique : un écart est normal, il est seulement signalé.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tresor import live
from tresor.app_data import fr
from tresor.config import OUTPUT_DIR
from tresor.features import NEW_SOURCE_GROUPS

FIXED, DAILY = "figé", "du jour"
FAMILIES = {"nominal_coupon": "notes et bonds", "bill": "bills"}
TARGETS = {
    "btc": "bid-to-cover",
    "dealer_share": "part des dealers",
    "indirect_share": "part des indirects",
    "tail_bp": "tail",
}
REFERENCE = "moyenne_5"  # modèle de référence, fixé avant de voir les résultats (phase 6)
EVALUATION = "2020-2026"  # période de jugement ; 2014-2019 sert au choix du modèle
HEADER = ["Chiffre", "Valeur", "Nature", "Fichier", "Calcul"]


@dataclass(frozen=True)
class KeyFigure:
    """Un chiffre : où il est, comment il se calcule, et s'il change chaque soir."""

    section: str
    label: str
    source: str  # nom du fichier dans data/output/
    how: str
    compute: Callable[[Path], str]
    nature: str = FIXED


# --- Lecture des fichiers --------------------------------------------------------------------


def _json(folder: Path, name: str) -> dict:
    return json.loads((folder / name).read_text(encoding="utf-8"))


def _row(table: pd.DataFrame, **conditions: str) -> pd.Series:
    """L'unique ligne qui remplit toutes les conditions (sinon erreur : pas de chiffre au
    hasard)."""
    mask = pd.Series(True, index=table.index)
    for column, value in conditions.items():
        mask &= table[column] == value
    found = table[mask]
    if len(found) != 1:
        raise ValueError(f"{len(found)} lignes pour {conditions}")
    return found.iloc[0]


def _metric(folder: Path, **conditions: str) -> pd.Series:
    table = pd.read_csv(folder / "backtest_metrics.csv")
    defaults = {"model": REFERENCE, "feature_set": "toutes", "period": EVALUATION}
    return _row(table, **{**defaults, **conditions})


def _interval(
    value: float, low: float, high: float, decimals: int, suffix: str = "", sign: bool = False
) -> str:
    """« valeur [bas ; haut] » ; `sign` ajoute le + des valeurs positives (variations, gains)."""
    return (
        f"{fr(value, decimals, sign=sign, suffix=suffix)} [{fr(low, decimals, sign=sign)} ;"
        f" {fr(high, decimals, sign=sign)}]"
    )


def _percent(value: float) -> str:
    return fr(100 * value, 0, suffix=" %")


# --- Les chiffres ----------------------------------------------------------------------------


def _r2(family: str, target: str) -> KeyFigure:
    def compute(folder: Path) -> str:
        row = _metric(folder, family=family, target=target)
        return _interval(row["r2_vs_mean3"], row["r2_ci_low"], row["r2_ci_high"], 2)

    return KeyFigure(
        "Validation walk-forward",
        f"R² contre mean3, {FAMILIES[family]}, {TARGETS[target]}, {EVALUATION}",
        "backtest_metrics.csv",
        f"model={REFERENCE}, feature_set=toutes, period={EVALUATION} ; r2_vs_mean3 [r2_ci_low ;"
        " r2_ci_high]",
        compute,
    )


def _quintile(surprise: str, quintile: int, label: str) -> KeyFigure:
    def compute(folder: Path) -> str:
        table = pd.read_csv(folder / "quintiles.csv")
        row = _row(table, period="2009-2026", surprise=surprise, quintile=quintile)
        change, low, high = row["rate_change_mean_bp"], row["ci_low"], row["ci_high"]
        return _interval(change, low, high, 1, " pb", sign=True)

    return KeyFigure(
        "Lien avec le marché (notes et bonds, 2009-2026)",
        label,
        "quintiles.csv",
        f"period=2009-2026, surprise={surprise}, quintile={quintile} ; rate_change_mean_bp"
        " [ci_low ; ci_high]",
        compute,
    )


def _strategy(period: str, rule: str, column: str, label: str, form: str) -> KeyFigure:
    def compute(folder: Path) -> str:
        row = _row(pd.read_csv(folder / "strategy_summary.csv"), period=period, rule=rule)
        if form == "pb":  # gain moyen par trade : colonnes mean_ci_low et mean_ci_high
            return _interval(row[column], row["mean_ci_low"], row["mean_ci_high"], 2, " pb", True)
        if form == "écart":
            low, high = row["vs_vendeur_ci_low"], row["vs_vendeur_ci_high"]
            return _interval(row["vs_vendeur_bp"], low, high, 2, " pb", sign=True)
        if form == "%":
            return _percent(row[column])
        if form == "entier":
            return fr(row[column], 0)
        return fr(row[column], 2, sign=True)

    how = f"period={period}, rule={rule} ; {column}"
    if form == "pb":
        how += " [mean_ci_low ; mean_ci_high]"
    elif form == "écart":
        how += " [vs_vendeur_ci_low ; vs_vendeur_ci_high]"
    return KeyFigure("Stratégie illustrative", label, "strategy_summary.csv", how, compute)


def _diagnostic(name: str, label: str) -> KeyFigure:
    def compute(folder: Path) -> str:
        row = _row(pd.read_csv(folder / "strategy_diagnostics.csv"), name=name)
        return fr(row["value"], 2, sign=True)

    return KeyFigure(
        "Stratégie illustrative", label, "strategy_diagnostics.csv", f"name={name} ; value", compute
    )


def _new_sources_gain(folder: Path) -> str:
    table = pd.read_csv(folder / "new_sources_comparison.csv")
    row = _row(table, family="nominal_coupon", target="btc", model=REFERENCE)
    return _interval(row["gain"], row["gain_ci_low"], row["gain_ci_high"], 3, sign=True)


def _cleaning(folder: Path) -> dict:
    return _json(folder, "cleaning_report.json")["auctions"]


def _features(folder: Path) -> pd.DataFrame:
    table = pd.read_csv(folder / "features_missing.csv")
    return table[table["groupe"] != "cible"]


def _log_counts(folder: Path, kind: str) -> str:
    log = live.read_log(folder / live.LOG_FILE)
    today = pd.Timestamp(_json(folder, live.STATUS_FILE)["today_new_york"])
    auctions = log.drop_duplicates(["cusip", "auction_date"])
    if kind == "lignes":
        return fr(len(log), 0)
    if kind == "figées":
        return fr(int((auctions["auction_date"] <= today).sum()), 0)
    tracking = pd.read_parquet(folder / live.TRACKING_FILE)
    done = tracking[tracking["status"] == "realisee"].drop_duplicates(["cusip", "auction_date"])
    return fr(len(done), 0)


DATA = "Données"
LIVE = "Prévisions en direct"

KEY_FIGURES: tuple[KeyFigure, ...] = (
    # Données
    KeyFigure(
        DATA,
        "Sources publiques collectées",
        "data_quality.json",
        "nombre d'entrées de sources",
        lambda f: fr(len(_json(f, "data_quality.json")["sources"]), 0),
        DAILY,
    ),
    KeyFigure(
        DATA,
        "Enchères reçues de Fiscal Data (depuis 2000)",
        "cleaning_report.json",
        "auctions.rows_in.fiscaldata",
        lambda f: fr(_cleaning(f)["rows_in"]["fiscaldata"], 0),
        DAILY,
    ),
    KeyFigure(
        DATA,
        "Enchères présentes dans Fiscal Data et TreasuryDirect, fusionnées",
        "cleaning_report.json",
        "auctions.duplicates_removed",
        lambda f: fr(_cleaning(f)["duplicates_removed"], 0),
        DAILY,
    ),
    KeyFigure(
        DATA,
        "Petites émissions hors calendrier exclues (moins de 1 Md$)",
        "cleaning_report.json",
        "auctions.excluded.petite_emission_hors_calendrier.rows",
        lambda f: fr(_cleaning(f)["excluded"]["petite_emission_hors_calendrier"]["rows"], 0),
        DAILY,
    ),
    KeyFigure(
        DATA,
        "Enchères gardées après nettoyage",
        "cleaning_report.json",
        "auctions.rows_out",
        lambda f: fr(_cleaning(f)["rows_out"], 0),
        DAILY,
    ),
    KeyFigure(
        DATA,
        "dont bills",
        "cleaning_report.json",
        "auctions.rows_out_by_family.bill",
        lambda f: fr(_cleaning(f)["rows_out_by_family"]["bill"], 0),
        DAILY,
    ),
    KeyFigure(
        DATA,
        "dont notes et bonds",
        "cleaning_report.json",
        "auctions.rows_out_by_family.nominal_coupon",
        lambda f: fr(_cleaning(f)["rows_out_by_family"]["nominal_coupon"], 0),
        DAILY,
    ),
    KeyFigure(
        DATA,
        "Variables explicatives point-in-time",
        "features_missing.csv",
        "lignes dont groupe ≠ cible",
        lambda f: fr(len(_features(f)), 0),
    ),
    KeyFigure(
        DATA,
        "dont variables des nouvelles sources (dealers, CFTC, liquidité)",
        "features_missing.csv",
        f"lignes dont groupe ∈ {sorted(NEW_SOURCE_GROUPS)}",
        lambda f: fr(int(_features(f)["groupe"].isin(NEW_SOURCE_GROUPS).sum()), 0),
    ),
    # Validation
    KeyFigure(
        "Validation walk-forward",
        "Enchères évaluées 2014-2026, bills",
        "backtest_metrics.csv",
        f"model={REFERENCE}, period=2014-2026, target=btc ; n",
        lambda f: fr(_metric(f, family="bill", target="btc", period="2014-2026")["n"], 0),
    ),
    KeyFigure(
        "Validation walk-forward",
        "Enchères évaluées 2014-2026, notes et bonds",
        "backtest_metrics.csv",
        f"model={REFERENCE}, period=2014-2026, target=btc ; n",
        lambda f: fr(_metric(f, family="nominal_coupon", target="btc", period="2014-2026")["n"], 0),
    ),
    KeyFigure(
        "Validation walk-forward",
        f"Enchères évaluées {EVALUATION}, bills",
        "backtest_metrics.csv",
        f"model={REFERENCE}, period={EVALUATION}, target=btc ; n",
        lambda f: fr(_metric(f, family="bill", target="btc")["n"], 0),
    ),
    KeyFigure(
        "Validation walk-forward",
        f"Enchères évaluées {EVALUATION}, notes et bonds",
        "backtest_metrics.csv",
        f"model={REFERENCE}, period={EVALUATION}, target=btc ; n",
        lambda f: fr(_metric(f, family="nominal_coupon", target="btc")["n"], 0),
    ),
    *(_r2(family, target) for family in FAMILIES for target in TARGETS),
    KeyFigure(
        "Validation walk-forward",
        f"Bonne direction contre mean3, notes et bonds, bid-to-cover, {EVALUATION}",
        "backtest_metrics.csv",
        f"model={REFERENCE}, period={EVALUATION} ; direction",
        lambda f: _percent(_metric(f, family="nominal_coupon", target="btc")["direction"]),
    ),
    KeyFigure(
        "Validation walk-forward",
        f"R² contre une constante, notes et bonds, tail, {EVALUATION}",
        "backtest_metrics.csv",
        f"model={REFERENCE}, period={EVALUATION} ; r2_vs_constant",
        lambda f: fr(_metric(f, family="nominal_coupon", target="tail_bp")["r2_vs_constant"], 2),
    ),
    KeyFigure(
        "Validation walk-forward",
        f"R² contre mean3 sans le meilleur trimestre, bills, bid-to-cover, {EVALUATION}",
        "backtest_metrics.csv",
        f"model={REFERENCE}, period={EVALUATION} ; r2_without_best_quarter",
        lambda f: fr(_metric(f, family="bill", target="btc")["r2_without_best_quarter"], 2),
    ),
    KeyFigure(
        "Validation walk-forward",
        f"R² du benchmark naive contre mean3, notes et bonds, bid-to-cover, {EVALUATION}",
        "backtest_metrics.csv",
        f"model=naive, period={EVALUATION} ; r2_vs_mean3",
        lambda f: fr(
            _metric(f, family="nominal_coupon", target="btc", model="naive")["r2_vs_mean3"],
            2,
            sign=True,
        ),
    ),
    KeyFigure(
        "Validation walk-forward",
        "Apport des nouvelles sources au R², notes et bonds, bid-to-cover, 2014-2026",
        "new_sources_comparison.csv",
        f"family=nominal_coupon, target=btc, model={REFERENCE} ; gain [gain_ci_low ; gain_ci_high]",
        _new_sources_gain,
    ),
    # Marché
    _quintile(
        "part_des_dealers",
        5,
        "Variation du taux, 20 % des enchères à la part des dealers la plus haute (demande faible)",
    ),
    _quintile(
        "part_des_dealers",
        1,
        "Variation du taux, 20 % des enchères à la part des dealers la plus basse (demande forte)",
    ),
    _quintile(
        "bid_to_cover",
        5,
        "Variation du taux, 20 % des enchères au bid-to-cover le plus bas par rapport à mean3",
    ),
    _quintile(
        "bid_to_cover",
        1,
        "Variation du taux, 20 % des enchères au bid-to-cover le plus haut par rapport à mean3",
    ),
    # Stratégie
    _strategy(EVALUATION, "strategie", "trades", f"Trades de la stratégie, {EVALUATION}", "entier"),
    _strategy(
        EVALUATION,
        "strategie",
        "mean_bp",
        f"Gain moyen par trade après coût, stratégie, {EVALUATION}",
        "pb",
    ),
    _strategy(
        EVALUATION,
        "toujours_vendeur",
        "mean_bp",
        f"Gain moyen par trade après coût, toujours vendeur, {EVALUATION}",
        "pb",
    ),
    _strategy(
        EVALUATION,
        "strategie",
        "vs_vendeur_bp",
        f"Écart de la stratégie avec toujours vendeur, {EVALUATION}",
        "écart",
    ),
    _strategy(
        EVALUATION, "strategie", "hit_rate", f"Taux de réussite de la stratégie, {EVALUATION}", "%"
    ),
    _strategy(
        EVALUATION,
        "strategie",
        "sharpe",
        f"Sharpe annualisé de la stratégie, {EVALUATION}",
        "nombre",
    ),
    _strategy(
        EVALUATION,
        "information_parfaite",
        "mean_bp",
        f"Gain moyen par trade avec information parfaite sur le sens du bid-to-cover, {EVALUATION}",
        "pb",
    ),
    _strategy(
        "2014-2026",
        "strategie",
        "mean_bp",
        "Gain moyen par trade après coût, stratégie, 2014-2026",
        "pb",
    ),
    _strategy(
        "2014-2026",
        "strategie",
        "vs_vendeur_bp",
        "Écart de la stratégie avec toujours vendeur, 2014-2026",
        "écart",
    ),
    _diagnostic(
        "corr_surprise_move",
        "Corrélation entre la surprise réelle de bid-to-cover et la variation du taux, 2014-2026",
    ),
    _diagnostic(
        "corr_predicted_move",
        "Corrélation entre la partie prévue du bid-to-cover et la variation du taux, 2014-2026",
    ),
    # Prévisions en direct
    KeyFigure(
        LIVE,
        "Enchères à venir prévues au dernier passage",
        live.STATUS_FILE,
        "upcoming_auctions",
        lambda f: fr(_json(f, live.STATUS_FILE)["upcoming_auctions"], 0),
        DAILY,
    ),
    KeyFigure(
        LIVE,
        "Lignes du journal figé",
        live.LOG_FILE,
        "nombre de lignes",
        lambda f: _log_counts(f, "lignes"),
        DAILY,
    ),
    KeyFigure(
        LIVE,
        "Enchères du journal déjà figées (date de l'enchère passée)",
        live.LOG_FILE,
        "enchères distinctes dont auction_date ≤ today_new_york de live_status.json",
        lambda f: _log_counts(f, "figées"),
        DAILY,
    ),
    KeyFigure(
        LIVE,
        "Enchères du journal dont le résultat est publié",
        live.TRACKING_FILE,
        "enchères distinctes dont status = realisee",
        lambda f: _log_counts(f, "réalisées"),
        DAILY,
    ),
)


# --- Écriture et vérification ----------------------------------------------------------------


def compute_all(folder: Path = OUTPUT_DIR) -> dict[str, str]:
    """Valeur de chaque chiffre, recalculée depuis les fichiers."""
    labels = [figure.label for figure in KEY_FIGURES]
    if len(set(labels)) != len(labels):
        raise ValueError("deux chiffres portent le même nom")
    return {figure.label: figure.compute(folder) for figure in KEY_FIGURES}


def render(folder: Path = OUTPUT_DIR, link_prefix: str = "../data/output/") -> str:
    """Contenu de chiffres_cles.md."""
    values = compute_all(folder)
    day = _json(folder, "data_quality.json")["today_new_york"]
    lines = [
        "# Chiffres clés",
        "",
        "Tous les chiffres du rapport et de l'oral, recalculés depuis les fichiers de"
        " `data/output/` par `python reports/make_figures.py`. La commande"
        " `python reports/make_figures.py --check` vérifie qu'ils correspondent toujours aux"
        " fichiers.",
        "",
        f"Données du {pd.Timestamp(day):%d/%m/%Y} (date de New York). Les chiffres « du jour »"
        " changent chaque soir avec le passage automatique ; les chiffres « figé » ne changent"
        " que si l'on relance `python -m tresor.pipeline --backtest --strategy`.",
        "",
        f"Modèle de référence : la moyenne des 5 modèles ({REFERENCE}), fixée avant de voir les"
        f" résultats. R² contre mean3 : 1 − erreur quadratique du modèle / erreur quadratique de"
        f" la moyenne des 3 dernières enchères de même maturité. Choix sur 2014-2019, jugement sur"
        f" {EVALUATION}. Intervalles à 90 %, par bootstrap d'années entières. pb : point de base.",
    ]
    for section in dict.fromkeys(figure.section for figure in KEY_FIGURES):
        lines += [
            "",
            f"## {section}",
            "",
            "| " + " | ".join(HEADER) + " |",
            "|" + "---|" * len(HEADER),
        ]
        for figure in KEY_FIGURES:
            if figure.section == section:
                link = f"[{figure.source}]({link_prefix}{figure.source})"
                cells = [figure.label, values[figure.label], figure.nature, link, figure.how]
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def read_values(text: str) -> dict[str, str]:
    """Chiffre → valeur, lus dans les tableaux de chiffres_cles.md."""
    values = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("| ") and len(cells) == len(HEADER) and cells[0] != HEADER[0]:
            values[cells[0]] = cells[1]
    return values


def check(path: Path, folder: Path = OUTPUT_DIR) -> tuple[list[str], list[str]]:
    """Erreurs (chiffre figé différent ou absent) et changements normaux (chiffres du jour)."""
    written = read_values(path.read_text(encoding="utf-8"))
    current = compute_all(folder)
    errors, changes = [], []
    for figure in KEY_FIGURES:
        old, new = written.get(figure.label), current[figure.label]
        if old is None:
            errors.append(f"absent du fichier : {figure.label}")
        elif old != new and figure.nature == FIXED:
            errors.append(f"{figure.label} : {old} écrit, {new} recalculé")
        elif old != new:
            changes.append(f"{figure.label} : {old} → {new}")
    errors += [f"inconnu : {label}" for label in written if label not in current]
    return errors, changes
