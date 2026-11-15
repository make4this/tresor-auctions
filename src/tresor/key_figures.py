"""Chiffres clés du rapport et de l'oral, recalculés depuis les fichiers de data/.

Chaque chiffre a son fichier source et son calcul. `render` écrit le contenu de
`reports/chiffres_cles.md` ; `check` recalcule tout et compare au fichier écrit. Deux natures :

- « figé » : fichiers écrits seulement par `--backtest --strategy`, lancés à la main. Un écart
  est toujours une erreur.
- « du jour » : fichiers réécrits chaque soir par le passage automatique (`--fetch --clean
  --predict`). Avec les mêmes données que le fichier écrit, un écart est une erreur ; avec des
  données plus récentes, la nouvelle valeur est seulement signalée.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tresor import features as ft
from tresor import live
from tresor.app_data import SOURCE_LABELS, fr
from tresor.config import DATA_DIR
from tresor.features import NEW_SOURCE_GROUPS

FIXED, DAILY = "figé", "du jour"
FAMILIES = {"nominal_coupon": "notes et bonds", "bill": "bills"}
TARGETS = {
    "btc": "bid-to-cover",
    "dealer_share": "part des dealers",
    "indirect_share": "part des indirects",
    "tail_bp": "tail",
}
REFERENCE = "moyenne_5"  # modèle de référence, retenu par principe en phase 6
EVALUATION = "2020-2026"  # période de jugement ; 2014-2019 sert au choix du meilleur modèle
DELAYS_SINCE = "2014-01-01"  # délais de publication mesurés sur la période du walk-forward
DOLLAR_INDEX = ("fred_dtwexb", "fred_dtwexbgs")
HEADER = ["Chiffre", "Valeur", "Nature", "Fichier", "Calcul"]
STAMP = "Données utilisées : "

DATA = "Données (`--fetch`, `--clean`)"
DELAYS = "Délais de publication (`--clean`)"
VALIDATION = "Validation walk-forward (`--backtest`)"
MARKET = "Lien avec le marché, notes et bonds (`--strategy`)"
STRATEGY = "Stratégie illustrative, notes et bonds (`--strategy`)"
LIVE = "Prévisions en direct (`--predict`)"


@dataclass(frozen=True)
class KeyFigure:
    """Un chiffre : où il est, comment il se calcule, et s'il change chaque soir."""

    section: str
    label: str
    source: str  # chemin dans data/
    how: str
    compute: Callable[[Path], str]  # reçoit le dossier data/
    nature: str = FIXED


# --- Lecture des fichiers et mise en forme ---------------------------------------------------


def _json(folder: Path, name: str) -> dict:
    return json.loads((folder / name).read_text(encoding="utf-8"))


def _row(table: pd.DataFrame, **conditions: object) -> pd.Series:
    """L'unique ligne qui remplit toutes les conditions (sinon erreur : pas de chiffre au
    hasard)."""
    mask = pd.Series(True, index=table.index)
    for column, value in conditions.items():
        mask &= table[column] == value
    found = table[mask]
    if len(found) != 1:
        raise ValueError(f"{len(found)} lignes pour {conditions}")
    return found.iloc[0]


def _conditions(**conditions: object) -> str:
    """Texte de la colonne « Calcul » : les mêmes conditions que celles passées à `_row`."""
    return ", ".join(f"{column}={value}" for column, value in conditions.items())


def _percent_interval(value: float, low: float, high: float) -> str:
    """R² en pourcentage, comme dans l'application et les figures : « +9,6 % [5,7 ; 14,5] »."""
    return f"{fr(100 * value, 1, sign=True)} % [{fr(100 * low, 1)} ; {fr(100 * high, 1)}]"


def _bp_interval(value: float, low: float, high: float, decimals: int) -> str:
    """Variation ou gain en points de base, signe toujours écrit : « +1,8 pb [+1,2 ; +2,5] »."""
    return (
        f"{fr(value, decimals, sign=True)} pb [{fr(low, decimals, sign=True)} ;"
        f" {fr(high, decimals, sign=True)}]"
    )


def _percent(value: float, decimals: int = 0, sign: bool = False) -> str:
    return f"{fr(100 * value, decimals, sign=sign)} %"


def _count(value: float) -> str:
    return fr(value, 0)


# --- Données ---------------------------------------------------------------------------------

CLEANING = "output/cleaning_report.json"
FEATURES = "clean/features.parquet"


def _cleaning(data: Path) -> dict:
    return _json(data / "output", "cleaning_report.json")["auctions"]


def _modeled(data: Path, status: str) -> pd.DataFrame:
    """Enchères des familles modélisées : réalisées depuis 2009 (échantillon des modèles), ou
    annoncées."""
    table = pd.read_parquet(data / FEATURES)
    if status == "completed":
        table = ft.model_sample(table)
    else:
        table = table[table["status"] == status]
    return table[table["family"].isin(ft.MODELED_FAMILIES)]


def auctions_overview(data: Path = DATA_DIR) -> dict[str, object]:
    """Chiffres de la figure des enchères par an (titre et sous-titre)."""
    sample, upcoming = _modeled(data, "completed"), _modeled(data, "upcoming")
    return {
        "kept": int(_cleaning(data)["rows_out"]),
        "sample": len(sample),
        "upcoming": len(upcoming),
        "last_completed": sample["auction_date"].max(),
        "last_upcoming": upcoming["auction_date"].max(),
    }


def _excluded(data: Path) -> int:
    return sum(int(info["rows"]) for info in _cleaning(data)["excluded"].values())


def _other_families(data: Path) -> int:
    by_family = _cleaning(data)["rows_out_by_family"]
    return sum(int(count) for name, count in by_family.items() if name not in FAMILIES)


def _variables(data: Path) -> pd.DataFrame:
    """Les variables de chaque modèle interprété (écrit par --backtest) : une liste unique."""
    table = pd.read_csv(data / "output" / "feature_importance.csv")
    counts = table.groupby(["family", "target", "model"])["feature"].nunique().unique()
    if len(counts) != 1:
        raise ValueError(f"nombres de variables différents selon le modèle : {counts}")
    return table.drop_duplicates("feature")


def _data_figure(label: str, source: str, how: str, compute: Callable[[Path], str]) -> KeyFigure:
    return KeyFigure(DATA, label, source, how, compute, DAILY)


DATA_FIGURES = (
    _data_figure(
        "Tables brutes collectées (Trésor américain, Fed de New York, FRED, CFTC, CBOE)",
        "output/data_quality.json",
        "nombre d'entrées de sources",
        lambda d: _count(len(_json(d / "output", "data_quality.json")["sources"])),
    ),
    _data_figure(
        "Enchères reçues de Fiscal Data (depuis 2000)",
        CLEANING,
        "auctions.rows_in.fiscaldata",
        lambda d: _count(_cleaning(d)["rows_in"]["fiscaldata"]),
    ),
    _data_figure(
        "Résultats récents reçus de TreasuryDirect",
        CLEANING,
        "auctions.rows_in.treasurydirect",
        lambda d: _count(_cleaning(d)["rows_in"]["treasurydirect"]),
    ),
    _data_figure(
        "Enchères annoncées reçues de TreasuryDirect (calendrier)",
        CLEANING,
        "auctions.rows_in.treasurydirect_upcoming",
        lambda d: _count(_cleaning(d)["rows_in"]["treasurydirect_upcoming"]),
    ),
    _data_figure(
        "Enchères présentes dans Fiscal Data et TreasuryDirect, fusionnées",
        CLEANING,
        "auctions.duplicates_removed",
        lambda d: _count(_cleaning(d)["duplicates_removed"]),
    ),
    _data_figure(
        "Enchères exclues au total",
        CLEANING,
        "somme des auctions.excluded.*.rows",
        lambda d: _count(_excluded(d)),
    ),
    _data_figure(
        "dont petites émissions hors calendrier (moins de 1 Md$)",
        CLEANING,
        "auctions.excluded.petite_emission_hors_calendrier.rows",
        lambda d: _count(_cleaning(d)["excluded"]["petite_emission_hors_calendrier"]["rows"]),
    ),
    _data_figure(
        "Enchères gardées après nettoyage, annoncées comprises",
        CLEANING,
        "auctions.rows_out",
        lambda d: _count(_cleaning(d)["rows_out"]),
    ),
    _data_figure(
        "dont enchères annoncées, pas encore réalisées",
        CLEANING,
        "auctions.rows_out_by_status.upcoming",
        lambda d: _count(_cleaning(d)["rows_out_by_status"].get("upcoming", 0)),
    ),
    _data_figure(
        "dont bills",
        CLEANING,
        "auctions.rows_out_by_family.bill",
        lambda d: _count(_cleaning(d)["rows_out_by_family"]["bill"]),
    ),
    _data_figure(
        "dont notes et bonds",
        CLEANING,
        "auctions.rows_out_by_family.nominal_coupon",
        lambda d: _count(_cleaning(d)["rows_out_by_family"]["nominal_coupon"]),
    ),
    _data_figure(
        "dont TIPS, FRN et CMB (non modélisés)",
        CLEANING,
        "somme des autres familles de auctions.rows_out_by_family",
        lambda d: _count(_other_families(d)),
    ),
    _data_figure(
        "Échantillon des modèles : bills, notes et bonds réalisés depuis 2009",
        FEATURES,
        "lignes de model_sample (status=completed, depuis 2009), family ∈ [bill, nominal_coupon]",
        lambda d: _count(len(_modeled(d, "completed"))),
    ),
    _data_figure(
        "dont bills dans l'échantillon des modèles",
        FEATURES,
        "même calcul, family=bill",
        lambda d: _count((_modeled(d, "completed")["family"] == "bill").sum()),
    ),
    _data_figure(
        "dont notes et bonds dans l'échantillon des modèles",
        FEATURES,
        "même calcul, family=nominal_coupon",
        lambda d: _count((_modeled(d, "completed")["family"] == "nominal_coupon").sum()),
    ),
    KeyFigure(
        DATA,
        "Variables explicatives point-in-time de chaque modèle",
        "output/feature_importance.csv",
        "variables distinctes (le même nombre pour chaque famille, cible et modèle)",
        lambda d: _count(len(_variables(d))),
    ),
    KeyFigure(
        DATA,
        "dont variables des nouvelles sources (dealers, CFTC, liquidité)",
        "output/feature_importance.csv",
        f"variables distinctes dont group ∈ {sorted(NEW_SOURCE_GROUPS)}",
        lambda d: _count(_variables(d)["group"].isin(NEW_SOURCE_GROUPS).sum()),
    ),
)


# --- Délais de publication -------------------------------------------------------------------


def publication_delays(market: pd.DataFrame, since: str = DELAYS_SINCE) -> pd.DataFrame:
    """Délai (jours calendaires) entre la date d'une donnée et sa publication, par source.

    FRED est coupé en deux : l'indice dollar, dont une partie des valeurs n'a été publiée que
    des années plus tard, et les autres séries. Trié du plus rapide au plus lent.
    """
    data = market[market["date"] >= pd.Timestamp(since)]
    label = data["source"].map(lambda name: SOURCE_LABELS.get(name, name))
    is_dollar = data["series"].isin(DOLLAR_INDEX)
    label = label.mask(is_dollar, "FRED : indice dollar")
    label = label.mask((data["source"] == "fred_series") & ~is_dollar, "FRED : autres séries")
    delays = pd.DataFrame({"label": label, "delay": (data["available_at"] - data["date"]).dt.days})
    table = (
        delays.groupby("label")["delay"]
        .agg(median="median", p90=lambda delay: delay.quantile(0.9))
        .reset_index()
    )
    notes = {"FRED : indice dollar": "indice créé en 2019, valeurs antérieures publiées alors"}
    table["note"] = table["label"].map(notes).fillna("")
    return table.sort_values(["median", "p90", "label"]).reset_index(drop=True)


def _delays(data: Path) -> pd.DataFrame:
    return publication_delays(pd.read_parquet(data / "clean" / "market.parquet"))


def _delay_figure(label: str) -> KeyFigure:
    def compute(data: Path) -> str:
        row = _row(_delays(data), label=label)
        return f"{fr(row['median'], 0)} j (9 sur 10 : {fr(row['p90'], 0)} j)"

    return KeyFigure(
        DELAYS,
        f"Délai médian de publication, {label}",
        "clean/market.parquet",
        f"available_at − date en jours, depuis {DELAYS_SINCE[:4]} ; médiane (9e décile)",
        compute,
        DAILY,
    )


DELAY_LABELS = (
    SOURCE_LABELS["treasury_yield_curve"],
    SOURCE_LABELS["cboe_vix"],
    SOURCE_LABELS["fiscaldata_cash_balance"],
    SOURCE_LABELS["fiscaldata_debt"],
    SOURCE_LABELS["cftc_positions"],
    SOURCE_LABELS["nyfed_dealer_positions"],
    "FRED : autres séries",
    "FRED : indice dollar",
)


# --- Validation ------------------------------------------------------------------------------

METRICS = "output/backtest_metrics.csv"


def _metric_figure(label: str, column: str, form: Callable[[pd.Series], str],
                   **conditions: str) -> KeyFigure:  # fmt: skip
    """Un chiffre de backtest_metrics.csv ; la colonne « Calcul » reprend les conditions."""
    full = {"model": REFERENCE, "feature_set": "toutes", "period": EVALUATION, **conditions}

    def compute(data: Path) -> str:
        return form(_row(pd.read_csv(data / METRICS), **full))

    return KeyFigure(VALIDATION, label, METRICS, f"{_conditions(**full)} ; {column}", compute)


def _r2(family: str, target: str, period: str = EVALUATION, suffix: str = "") -> KeyFigure:
    return _metric_figure(
        f"R² contre mean3, {FAMILIES[family]}, {TARGETS[target]}, {period}{suffix}",
        "r2_vs_mean3 [r2_ci_low ; r2_ci_high]",
        lambda row: _percent_interval(row["r2_vs_mean3"], row["r2_ci_low"], row["r2_ci_high"]),
        family=family,
        target=target,
        period=period,
    )


def _evaluated(family: str, period: str) -> KeyFigure:
    return _metric_figure(
        f"Enchères évaluées {period}, {FAMILIES[family]} (bid-to-cover)",
        "n",
        lambda row: _count(row["n"]),
        family=family,
        target="btc",
        period=period,
    )


NEW_SOURCES = "output/new_sources_comparison.csv"


def _new_sources(data: Path) -> pd.DataFrame:
    return pd.read_csv(data / NEW_SOURCES)


def _new_sources_gain(data: Path) -> str:
    row = _row(_new_sources(data), family="nominal_coupon", target="btc", model=REFERENCE)
    low = fr(100 * row["gain_ci_low"], 1, sign=True)
    high = fr(100 * row["gain_ci_high"], 1, sign=True)
    return f"{fr(100 * row['gain'], 1, sign=True)} point [{low} ; {high}]"


def _new_sources_count(data: Path, worse: bool, model: str | None) -> str:
    table = _new_sources(data)
    if model is not None:
        table = table[table["model"] == model]
    hits = (table["gain_ci_high"] < 0) if worse else (table["gain_ci_low"] > 0)
    return f"{_count(hits.sum())} sur {_count(len(table))}"


VALIDATION_FIGURES = (
    _evaluated("bill", "2014-2026"),
    _evaluated("nominal_coupon", "2014-2026"),
    _evaluated("bill", EVALUATION),
    _evaluated("nominal_coupon", EVALUATION),
    *(_r2(family, target) for family in FAMILIES for target in TARGETS),
    _r2("nominal_coupon", "btc", "2014-2026", " (fin de la courbe de gain cumulé)"),
    _r2("bill", "btc", "2014-2026", " (fin de la courbe de gain cumulé)"),
    _metric_figure(
        "Bonne direction, notes et bonds, bid-to-cover, 2020-2026 : part des enchères où le"
        " modèle dit correctement si le bid-to-cover sera au-dessus ou au-dessous de mean3",
        "direction",
        lambda row: _percent(row["direction"]),
        family="nominal_coupon",
        target="btc",
    ),
    _metric_figure(
        f"R² contre une constante, notes et bonds, tail, {EVALUATION}",
        "r2_vs_constant",
        lambda row: _percent(row["r2_vs_constant"], 1, sign=True),
        family="nominal_coupon",
        target="tail_bp",
    ),
    _metric_figure(
        f"R² contre mean3 sans le meilleur trimestre, bills, bid-to-cover, {EVALUATION}",
        "r2_without_best_quarter",
        lambda row: _percent(row["r2_without_best_quarter"], 1, sign=True),
        family="bill",
        target="btc",
    ),
    _metric_figure(
        f"R² du benchmark naïf (dernière enchère) contre mean3, notes et bonds, bid-to-cover,"
        f" {EVALUATION}",
        "r2_vs_mean3",
        lambda row: _percent(row["r2_vs_mean3"], 1, sign=True),
        family="nominal_coupon",
        target="btc",
        model="naive",
    ),
    KeyFigure(
        VALIDATION,
        "Apport des nouvelles sources au R², notes et bonds, bid-to-cover, 2014-2026",
        NEW_SOURCES,
        f"{_conditions(family='nominal_coupon', target='btc', model=REFERENCE)} ; gain"
        " [gain_ci_low ; gain_ci_high], en points de pourcentage",
        _new_sources_gain,
    ),
    KeyFigure(
        VALIDATION,
        "Cibles où les nouvelles sources dégradent la prévision (intervalle entièrement"
        " négatif), moyenne des 5, 2014-2026",
        NEW_SOURCES,
        f"{_conditions(model=REFERENCE)} ; gain_ci_high < 0",
        lambda d: _new_sources_count(d, worse=True, model=REFERENCE),
    ),
    KeyFigure(
        VALIDATION,
        "Cibles où les nouvelles sources améliorent la prévision (intervalle entièrement"
        " positif), moyenne des 5, 2014-2026",
        NEW_SOURCES,
        f"{_conditions(model=REFERENCE)} ; gain_ci_low > 0",
        lambda d: _new_sources_count(d, worse=False, model=REFERENCE),
    ),
    KeyFigure(
        VALIDATION,
        "Couples modèle-cible où les nouvelles sources dégradent la prévision, tous modèles,"
        " 2014-2026",
        NEW_SOURCES,
        "toutes les lignes ; gain_ci_high < 0",
        lambda d: _new_sources_count(d, worse=True, model=None),
    ),
)


# --- Marché et stratégie ---------------------------------------------------------------------

QUINTILES = "output/quintiles.csv"
CHANGE = "Variation du taux le jour de l'enchère"
WEAK_DEALERS = (
    f"{CHANGE}, 20 % des enchères où la part des dealers dépasse le plus la moyenne des 3"
    " dernières (demande faible)"
)


def _quintile(surprise: str, quintile: int, label: str, period: str = "2009-2026") -> KeyFigure:
    conditions = {"period": period, "surprise": surprise, "quintile": quintile}

    def compute(data: Path) -> str:
        row = _row(pd.read_csv(data / QUINTILES), **conditions)
        return _bp_interval(row["rate_change_mean_bp"], row["ci_low"], row["ci_high"], 1)

    return KeyFigure(
        MARKET,
        label,
        QUINTILES,
        f"{_conditions(**conditions)} ; rate_change_mean_bp [ci_low ; ci_high]",
        compute,
    )


def _quintile_count(data: Path) -> str:
    table = pd.read_csv(data / QUINTILES)
    part = table[(table["period"] == "2009-2026") & (table["surprise"] == "bid_to_cover")]
    return _count(part["n"].sum())


MARKET_FIGURES = (
    KeyFigure(
        MARKET,
        "Enchères de l'analyse par quintiles, 2009-2026",
        QUINTILES,
        f"{_conditions(period='2009-2026', surprise='bid_to_cover')} ; somme de n",
        _quintile_count,
    ),
    _quintile("part_des_dealers", 5, f"{WEAK_DEALERS}, 2009-2026"),
    _quintile(
        "part_des_dealers",
        1,
        f"{CHANGE}, 20 % des enchères où la part des dealers est le plus en dessous de la"
        " moyenne des 3 dernières (demande forte), 2009-2026",
    ),
    *(
        _quintile("part_des_dealers", 5, f"{WEAK_DEALERS}, {period}", period)
        for period in ("2009-2013", "2014-2019", "2020-2026")
    ),
    _quintile(
        "bid_to_cover",
        5,
        f"{CHANGE}, 20 % des enchères dont le bid-to-cover est le plus en dessous de la moyenne"
        " des 3 dernières (demande faible), 2009-2026",
    ),
    _quintile(
        "bid_to_cover",
        1,
        f"{CHANGE}, 20 % des enchères dont le bid-to-cover est le plus au-dessus de la moyenne"
        " des 3 dernières (demande forte), 2009-2026",
    ),
)

SUMMARY = "output/strategy_summary.csv"
RULES = {
    "strategie": "stratégie",
    "toujours_vendeur": "toujours vendeur",
    "information_parfaite": "information parfaite sur le signe de la surprise de bid-to-cover"
    " (réel − mean3), impossible en pratique",
}


def _strategy(period: str, rule: str, kind: str) -> KeyFigure:
    """Un chiffre de strategy_summary.csv : gain moyen, total, écart, trades, réussite ou
    Sharpe."""
    label, how, form = {
        "gain": (
            "Gain moyen par trade après coût",
            "mean_bp [mean_ci_low ; mean_ci_high]",
            lambda row: _bp_interval(row["mean_bp"], row["mean_ci_low"], row["mean_ci_high"], 2),
        ),
        "total": (
            "Gain total après coût",
            "total_bp",
            lambda row: f"{fr(row['total_bp'], 0, sign=True)} pb",
        ),
        "écart": (
            "Écart du gain moyen avec toujours vendeur",
            "vs_vendeur_bp [vs_vendeur_ci_low ; vs_vendeur_ci_high]",
            lambda row: _bp_interval(
                row["vs_vendeur_bp"], row["vs_vendeur_ci_low"], row["vs_vendeur_ci_high"], 2
            ),
        ),
        "trades": (
            "Trades (une position par enchère de notes et bonds)",
            "trades",
            lambda row: _count(row["trades"]),
        ),
        "réussite": (
            "Taux de réussite après coût (un taux inchangé compte comme une perte)",
            "hit_rate",
            lambda row: _percent(row["hit_rate"]),
        ),
        "sharpe": ("Sharpe annualisé", "sharpe", lambda row: fr(row["sharpe"], 2, sign=True)),
    }[kind]
    conditions = {"period": period, "rule": rule}

    def compute(data: Path) -> str:
        return form(_row(pd.read_csv(data / SUMMARY), **conditions))

    return KeyFigure(
        STRATEGY,
        f"{label}, {RULES[rule]}, {period}",
        SUMMARY,
        f"{_conditions(**conditions)} ; {how}",
        compute,
    )


def _diagnostic(name: str, label: str) -> KeyFigure:
    def compute(data: Path) -> str:
        row = _row(pd.read_csv(data / "output" / "strategy_diagnostics.csv"), name=name)
        return fr(row["value"], 2, sign=True)

    return KeyFigure(
        STRATEGY,
        label,
        "output/strategy_diagnostics.csv",
        f"{_conditions(name=name)} ; value",
        compute,
    )


STRATEGY_FIGURES = (
    *(
        _strategy(period, rule, kind)
        for period in (EVALUATION, "2014-2026")
        for rule, kind in (
            ("strategie", "trades"),
            ("strategie", "gain"),
            ("toujours_vendeur", "gain"),
            ("strategie", "écart"),
            ("information_parfaite", "gain"),
            ("strategie", "réussite"),
            ("strategie", "sharpe"),
        )
    ),
    *(_strategy("2014-2026", rule, "total") for rule in RULES),
    _diagnostic(
        "corr_surprise_move",
        "Corrélation entre la surprise réelle de bid-to-cover et la variation du taux, 2014-2026",
    ),
    _diagnostic(
        "corr_predicted_move",
        "Corrélation entre la partie prévue du bid-to-cover et la variation du taux, 2014-2026",
    ),
    _diagnostic(
        "corr_predicted_move_within_year",
        "Même corrélation, en écart à la moyenne de chaque année (sans l'effet de période),"
        " 2014-2026",
    ),
)


# --- Prévisions en direct --------------------------------------------------------------------


def _log_counts(data: Path, kind: str) -> str:
    output = data / "output"
    log = live.read_log(output / live.LOG_FILE)
    today = pd.Timestamp(_json(output, live.STATUS_FILE)["today_new_york"])
    auctions = log.drop_duplicates(["cusip", "auction_date"])
    if kind == "lignes":
        return _count(len(log))
    if kind == "figées":
        return _count((auctions["auction_date"] <= today).sum())
    tracking = pd.read_parquet(output / live.TRACKING_FILE)
    done = tracking[tracking["status"] == "realisee"].drop_duplicates(["cusip", "auction_date"])
    return _count(len(done))


LIVE_FIGURES = (
    KeyFigure(
        LIVE,
        "Enchères à venir prévues au dernier passage (bills, notes et bonds)",
        f"output/{live.STATUS_FILE}",
        "upcoming_auctions",
        lambda d: _count(_json(d / "output", live.STATUS_FILE)["upcoming_auctions"]),
        DAILY,
    ),
    KeyFigure(
        LIVE,
        "Lignes du journal des prévisions",
        f"output/{live.LOG_FILE}",
        "nombre de lignes",
        lambda d: _log_counts(d, "lignes"),
        DAILY,
    ),
    KeyFigure(
        LIVE,
        "Enchères du journal déjà figées (jour de l'enchère atteint)",
        f"output/{live.LOG_FILE}",
        "enchères distinctes dont auction_date ≤ today_new_york de live_status.json",
        lambda d: _log_counts(d, "figées"),
        DAILY,
    ),
    KeyFigure(
        LIVE,
        "Enchères du journal dont le résultat est publié",
        f"output/{live.TRACKING_FILE}",
        "enchères distinctes dont status=realisee",
        lambda d: _log_counts(d, "réalisées"),
        DAILY,
    ),
)

KEY_FIGURES: tuple[KeyFigure, ...] = (
    *DATA_FIGURES,
    *(_delay_figure(label) for label in DELAY_LABELS),
    *VALIDATION_FIGURES,
    *MARKET_FIGURES,
    *STRATEGY_FIGURES,
    *LIVE_FIGURES,
)


# --- Écriture et vérification ----------------------------------------------------------------


def data_stamp(data: Path = DATA_DIR) -> str:
    """Les données utilisées : date du contrôle qualité et heure de la dernière prévision."""
    quality = _json(data / "output", "data_quality.json")["today_new_york"]
    run_at = pd.Timestamp(_json(data / "output", live.STATUS_FILE)["run_at"]).tz_convert("UTC")
    return (
        f"contrôle qualité du {pd.Timestamp(quality):%d/%m/%Y} (date de New York), dernière"
        f" prévision le {run_at:%d/%m/%Y à %H:%M} UTC."
    )


def compute_all(data: Path = DATA_DIR) -> dict[str, str]:
    """Valeur de chaque chiffre, recalculée depuis les fichiers."""
    labels = [figure.label for figure in KEY_FIGURES]
    if len(set(labels)) != len(labels):
        raise ValueError("deux chiffres portent le même nom")
    return {figure.label: figure.compute(data) for figure in KEY_FIGURES}


INTRO = """# Chiffres clés

Tous les chiffres du rapport et de l'oral, recalculés depuis les fichiers de `data/` par \
`python reports/make_figures.py`. La commande `python reports/make_figures.py --check` vérifie \
qu'ils correspondent toujours aux fichiers. Le titre de chaque partie donne l'étape du pipeline \
qui produit ses fichiers.

Les chiffres marqués « du jour » changent chaque soir avec le passage automatique. Les chiffres \
marqués « figé » ne changent que si l'on relance \
`python -m tresor.pipeline --backtest --strategy`.

{stamp}

**Petit glossaire.**

- **Modèle de référence** : la moyenne des prévisions des 5 modèles ({reference}), retenue par \
principe et non d'après ses résultats. Le meilleur modèle individuel est choisi sur 2014-2019 ; \
tous les modèles sont jugés sur {evaluation}.
- **mean3** : la moyenne des 3 dernières enchères de même maturité, le benchmark de référence.
- **R² contre mean3** : 1 − erreur quadratique du modèle / erreur quadratique de mean3, en %. \
Au-dessus de 0, le modèle fait mieux que mean3.
- **Constante** : la moyenne de la cible sur la période d'entraînement.
- **Tail** : taux obtenu à l'enchère moins le taux de marché de la veille, en points de base \
(pb, 0,01 point de pourcentage).
- **Variation du taux le jour de l'enchère** : de la clôture de la veille à celle du jour de \
l'enchère, pour la maturité de l'enchère.
- **Trade** : une position sur cette même journée, vendeuse ou acheteuse de duration.
- **Sharpe** : gain moyen rapporté à sa variabilité, sur un an.
- Intervalles à 90 %, par bootstrap d'années entières. « j » : jours calendaires.
"""


def render(data: Path = DATA_DIR, link_prefix: str = "../data/") -> str:
    """Contenu de chiffres_cles.md."""
    values = compute_all(data)
    intro = INTRO.format(stamp=STAMP + data_stamp(data), reference=REFERENCE, evaluation=EVALUATION)
    lines = [intro.rstrip("\n")]
    for section in dict.fromkeys(figure.section for figure in KEY_FIGURES):
        lines += ["", f"## {section}", "", "| " + " | ".join(HEADER) + " |",
                  "|" + "---|" * len(HEADER)]  # fmt: skip
        for figure in KEY_FIGURES:
            if figure.section == section:
                name = figure.source.rsplit("/", 1)[-1]
                link = f"[{name}]({link_prefix}{figure.source})"
                cells = [figure.label, values[figure.label], figure.nature, link, figure.how]
                if any("|" in cell for cell in cells):
                    raise ValueError(f"« | » dans une cellule de {figure.label}")
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _is_row(line: str) -> bool:
    return line.startswith("| ") and not line.startswith(f"| {HEADER[0]} |")


def read_rows(text: str) -> tuple[dict[str, list[str]], list[str]]:
    """Lignes des tableaux de chiffres_cles.md (chiffre → cellules), et chiffres en double."""
    rows: dict[str, list[str]] = {}
    duplicates = []
    for line in text.splitlines():
        if _is_row(line):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells[0] in rows:
                duplicates.append(cells[0])
            rows[cells[0]] = cells
    return rows, duplicates


def read_values(text: str) -> dict[str, str]:
    """Chiffre → valeur, lus dans les tableaux de chiffres_cles.md."""
    return {label: cells[1] for label, cells in read_rows(text)[0].items()}


def _other_lines(text: str, keep_stamp: bool) -> list[str]:
    """Tout le texte hors des lignes de chiffres (et hors de la ligne des données)."""
    return [
        line
        for line in text.splitlines()
        if not _is_row(line) and (keep_stamp or not line.startswith(STAMP))
    ]


def check(path: Path, data: Path = DATA_DIR) -> tuple[list[str], list[str]]:
    """Erreurs, et changements normaux des chiffres du jour (données plus récentes).

    Chaque ligne du fichier doit être celle que l'on écrirait aujourd'hui. Seule exception :
    si les données ont changé depuis l'écriture, la valeur d'un chiffre du jour peut différer.
    """
    written_text = path.read_text(encoding="utf-8")
    try:
        expected_text = render(data)
    except (OSError, KeyError, ValueError) as error:  # fichier absent ou format inattendu
        return [f"recalcul impossible : {type(error).__name__} : {error}"], []
    written, duplicates = read_rows(written_text)
    expected, _ = read_rows(expected_text)
    stamp = next((line for line in written_text.splitlines() if line.startswith(STAMP)), None)
    same_data = stamp == STAMP + data_stamp(data)
    errors = [f"en double : {label}" for label in duplicates]
    changes = []
    natures = {figure.label: figure.nature for figure in KEY_FIGURES}
    for label, cells in expected.items():
        old = written.get(label)
        if old is None:
            errors.append(f"absent du fichier : {label}")
        elif old == cells:
            continue
        elif old[0] == cells[0] and old[2:] == cells[2:] and len(old) == len(cells):
            message = f"{label} : {old[1]} écrit, {cells[1]} recalculé"
            if natures[label] == DAILY and not same_data:
                changes.append(message)
            else:
                errors.append(message)
        else:
            errors.append(f"{label} : nature, fichier ou calcul modifié")
    errors += [f"inconnu : {label}" for label in written if label not in expected]
    if _other_lines(written_text, same_data) != _other_lines(expected_text, same_data):
        errors.append("texte modifié hors des tableaux (titre, introduction ou glossaire)")
    return errors, changes
