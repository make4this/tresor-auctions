"""Test des nouvelles données (novembre 2026) : `python -m tresor.pipeline --new-data`.

Trois groupes de variables candidates (tresor.features.CANDIDATE_GROUPS) :

- banques centrales étrangères : la dette américaine que la Fed garde pour elles ;
- calendrier : jours jusqu'à la prochaine décision de la Fed, inflation et emploi ;
- taux étrangers : taux allemand et japonais à 10 ans.

Pour chaque groupe, le walk-forward du site est relancé à l'identique (mêmes modèles, mêmes
trimestres, mêmes enchères) avec les variables du modèle du site plus ce groupe, puis avec les
trois groupes ensemble. Le modèle de base est recalculé dans la même exécution, sur les mêmes
données, pour que l'écart ne mesure que l'effet des nouvelles variables. Un groupe n'entre dans
le modèle du site que si son gain est positif avec un intervalle à 90 % au-dessus de zéro.
"""

from __future__ import annotations

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
# Le bid-to-cover d'abord : c'est la cible du signal du site. Les autres cibles se testent en
# passant `targets` (environ 25 minutes par cible sur 8 cœurs).
TARGETS = ("btc",)
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
                signals, right = weak_signal(candidate)
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
                    }
                )
    order = {name: index for index, name in enumerate(FEATURE_SETS)}
    table = pd.DataFrame(rows)
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
    """Walk-forward avec et sans chaque groupe de nouvelles données, puis la comparaison."""
    sample = ft.model_sample(pd.read_parquet(clean_dir / "features.parquet"))
    predictions = md.run_walk_forward(
        sample, tasks(targets=targets), n_jobs=n_jobs, feature_sets=FEATURE_SETS
    )
    predictions = md.add_ensemble(md.add_constant_benchmark(predictions, sample))
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
