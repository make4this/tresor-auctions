"""Test des nouvelles données (novembre 2026) : `python -m tresor.pipeline --new-data`.

Trois groupes de variables candidates (tresor.features.CANDIDATE_GROUPS) :

- banques centrales étrangères : la dette américaine que la Fed garde pour elles ;
- calendrier : jours jusqu'à la prochaine décision de la Fed, inflation et emploi ;
- taux étrangers : taux allemand et japonais à 10 ans.

Pour chaque groupe, le walk-forward du site est relancé à l'identique (mêmes modèles, mêmes
trimestres, mêmes enchères) avec les variables du modèle du site plus ce groupe, puis avec les
trois groupes ensemble. Le modèle de base est recalculé dans la même exécution, sur les mêmes
données, pour que l'écart ne mesure que l'effet des nouvelles variables.

Deux lectures du gain de la moyenne des 5 :

- l'intervalle à 90 % par bootstrap d'années entières (la règle fixée avant le premier test) ;
- un test exact par inversion des signes des gains annuels (2^années tirages), corrigé de Holm
  sur toutes les comparaisons de la période (4 groupes × 2 familles × 4 cibles). Avec 32
  comparaisons, un intervalle positif peut venir du hasard : un groupe n'entre dans le modèle du
  site que si son gain tient aussi après cette correction. Cette exigence a été ajoutée après le
  premier test, dans le sens prudent (elle ne peut que refuser plus de groupes).
"""

from __future__ import annotations

import itertools
import logging
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tresor import features as ft
from tresor import models as md
from tresor.config import CLEAN_DIR, OUTPUT_DIR, SIGNAL_THRESHOLD

logger = logging.getLogger(__name__)

BASE = "toutes"
ALL_NEW = "all_new"
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    BASE: md.FEATURE_SETS[BASE],
    **{
        f"{BASE}+{group}": (*md.FEATURE_SETS[BASE], *names)
        for group, names in ft.CANDIDATE_GROUPS.items()
    },
    f"{BASE}+{ALL_NEW}": (*md.FEATURE_SETS[BASE], *ft.CANDIDATE_COLUMNS),
}
# Les 4 cibles. Compter de 30 minutes à 1 heure par cible sur 8 cœurs libres ; on peut n'en
# relancer qu'une partie avec `targets`, les résultats des autres cibles sont alors gardés.
TARGETS = ft.TARGETS
TEST_PERIODS = ("2014-2026", "2020-2026")
PREDICTIONS_FILE = "new_data_predictions.parquet"
RESULTS_FILE = "new_data_test.csv"


def tasks(
    families: Sequence[str] = ft.MODELED_FAMILIES, targets: Sequence[str] = TARGETS
) -> list[md.Task]:
    return [
        md.Task(family, target, model, feature_set)
        for family in families
        for target in targets
        for model in md.MODEL_NAMES
        for feature_set in FEATURE_SETS
    ]


def weak_signal(part: pd.DataFrame) -> tuple[int, float]:
    """Signal « plus faible » du site (prévision sous mean3 de 0,08 ou plus) : combien de fois il
    est donné, et dans quelle part des cas l'enchère finit bien sous mean3."""
    usable = part.dropna(subset=["actual", "prediction", "mean3"])
    # Même règle que le site : écart arrondi à 10⁻⁹ (−0,08 exactement compte comme « plus faible »).
    weak = (usable["prediction"] - usable["mean3"]).round(9) <= -SIGNAL_THRESHOLD
    if not weak.any():
        return 0, float("nan")
    return int(weak.sum()), float((usable.loc[weak, "actual"] < usable.loc[weak, "mean3"]).mean())


def yearly_gains(base: pd.DataFrame, candidate: pd.DataFrame) -> np.ndarray:
    """Gain de R² contre mean3 année par année (leur somme est le gain de la période), sur les
    mêmes enchères pour la base et le candidat."""
    keys = ["cusip", "auction_date"]
    usable = ["actual", "prediction", "mean3"]
    left = base.dropna(subset=usable).set_index(keys)
    right = candidate.dropna(subset=usable).set_index(keys)
    common = left.index.intersection(right.index)
    left, right = left.loc[common], right.loc[common]
    reference = ((left["actual"] - left["mean3"]) ** 2).sum()
    gain = (left["actual"] - left["prediction"]) ** 2 - (right["actual"] - right["prediction"]) ** 2
    return (gain.groupby(common.get_level_values("auction_date").year).sum() / reference).to_numpy()


def sign_flip_p(gains: np.ndarray) -> float:
    """Probabilité d'un gain total au moins aussi grand si chaque année pouvait aussi bien aller
    dans l'autre sens (test exact sur les 2^n changements de signe, unilatéral)."""
    if len(gains) == 0:
        return float("nan")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(gains))))
    return float(((signs * gains).sum(axis=1) >= gains.sum() - 1e-12).mean())


def holm(p_values: pd.Series) -> pd.Series:
    """Probabilités corrigées de Holm (dans l'ordre d'origine)."""
    ordered = p_values.sort_values()
    count = len(ordered)
    adjusted = (ordered * (count - np.arange(count))).cummax().clip(upper=1)
    return adjusted.reindex(p_values.index)


def compare(predictions: pd.DataFrame) -> pd.DataFrame:
    """Par période, famille, cible, modèle et jeu : R², gain contre la base et son intervalle."""
    rows = []
    keys = ["family", "target", "model"]
    kept = predictions[predictions["model"].isin([*md.MODEL_NAMES, md.ENSEMBLE])]
    for period in TEST_PERIODS:
        for (family, target, model), part in md.in_period(kept, period).groupby(keys):
            base = part[part["feature_set"] == BASE]
            base_score = md.score(base)
            for feature_set, candidate in part.groupby("feature_set", sort=False):
                result = md.score(candidate)
                low, high = (
                    (np.nan, np.nan)
                    if feature_set == BASE
                    else md.bootstrap_r2(candidate, other=base)
                )
                gains = yearly_gains(base, candidate) if feature_set != BASE else np.array([])
                # Le signal « plus faible » du site ne porte que sur le bid-to-cover.
                signals, right = weak_signal(candidate) if target == "btc" else (np.nan, np.nan)
                rows.append(
                    {
                        "period": period,
                        "family": family,
                        "target": target,
                        "model": model,
                        "feature_set": feature_set,
                        "n": result["n"],
                        "r2_vs_mean3": result["r2_vs_mean3"],
                        "r2_base": base_score["r2_vs_mean3"],
                        "gain": result["r2_vs_mean3"] - base_score["r2_vs_mean3"],
                        "gain_ci_low": low,
                        "gain_ci_high": high,
                        "direction": result["direction"],
                        "weak_signals": signals,
                        "weak_right": right,
                        "years_positive": int((gains > 0).sum()) if len(gains) else np.nan,
                        "years": len(gains) if len(gains) else np.nan,
                        "p_sign_flip": sign_flip_p(gains) if len(gains) else np.nan,
                    }
                )
    order = {name: index for index, name in enumerate(FEATURE_SETS)}
    table = pd.DataFrame(rows)
    # Correction de Holm sur les comparaisons de la moyenne des 5, période par période.
    table["p_holm"] = np.nan
    tested = (table["model"] == md.ENSEMBLE) & (table["feature_set"] != BASE)
    for _, index in table[tested].groupby("period").groups.items():
        table.loc[index, "p_holm"] = holm(table.loc[index, "p_sign_flip"])
    table["set_order"] = table["feature_set"].map(order)
    return table.sort_values(["period", "family", "target", "model", "set_order"]).drop(
        columns="set_order"
    )


def _write(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary_path, index=False)
    else:
        frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def run_new_data_test(
    clean_dir: Path = CLEAN_DIR,
    output_dir: Path = OUTPUT_DIR,
    targets: Sequence[str] = TARGETS,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Walk-forward avec et sans chaque groupe de nouvelles données, puis la comparaison.

    Les prévisions déjà enregistrées pour les autres cibles sont gardées : la comparaison porte
    toujours sur toutes les cibles testées.
    """
    sample = ft.model_sample(pd.read_parquet(clean_dir / "features.parquet"))
    predictions = md.run_walk_forward(
        sample, tasks(targets=targets), n_jobs=n_jobs, feature_sets=FEATURE_SETS
    )
    predictions = md.add_ensemble(md.add_constant_benchmark(predictions, sample))
    previous_path = output_dir / PREDICTIONS_FILE
    if previous_path.exists():
        previous = pd.read_parquet(previous_path)
        kept = previous[~previous["target"].isin(targets)]
        predictions = pd.concat([kept, predictions[kept.columns]], ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(predictions, output_dir / PREDICTIONS_FILE)
    table = compare(predictions)
    _write(table, output_dir / RESULTS_FILE)
    summary = table[(table["model"] == md.ENSEMBLE) & (table["feature_set"] != BASE)]
    logger.info(
        "Nouvelles données, moyenne des 5 :\n%s",
        summary[
            [
                "period",
                "family",
                "target",
                "feature_set",
                "r2_base",
                "r2_vs_mean3",
                "gain",
                "gain_ci_low",
                "gain_ci_high",
            ]
        ]
        .round(4)
        .to_string(index=False),
    )
    return table
