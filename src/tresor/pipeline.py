"""Orchestration du projet : `python -m tresor.pipeline --all`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from tresor.clean.run import run_cleaning
from tresor.config import OUTPUT_DIR
from tresor.features import run_features
from tresor.http import FetchError
from tresor.sources.collect import FETCH_STATUS_FILE, collect_all

logger = logging.getLogger(__name__)

StepFunction = Callable[[], None]


def _run_fetch() -> None:
    """Télécharge les sources, enregistre les tables brutes et le résultat de chaque source."""
    collect_all(status_path=OUTPUT_DIR / FETCH_STATUS_FILE)


def _run_backtest() -> None:
    """Walk-forward de tous les modèles, interprétation et figures (environ 2 heures)."""
    import pandas as pd

    from tresor import models, plots

    metrics = models.run_backtest()
    importance = models.run_interpretation()
    plots.all_figures(
        metrics,
        pd.read_parquet(OUTPUT_DIR / "backtest_predictions.parquet"),
        pd.read_csv(OUTPUT_DIR / "new_sources_comparison.csv"),
        pd.read_csv(OUTPUT_DIR / "best_models.csv"),
        importance,
    )


def _run_strategy() -> None:
    """Quintiles de surprise et stratégie, à partir des prévisions du backtest (phase 7)."""
    from tresor import strategy

    strategy.run_strategy()


def _run_new_data() -> None:
    """Test des nouvelles données : walk-forward avec et sans chaque groupe (environ 1 heure)."""
    from tresor import new_data

    new_data.run_new_data_test()


def _run_train() -> None:
    """Modèles finaux sur toutes les enchères réalisées, sauvegardés dans data/models/."""
    from tresor import live

    live.run_train()


def _run_predict() -> None:
    """Prévisions des enchères à venir, journal figé et suivi en direct."""
    from tresor import live

    live.run_predict()


def _run_clean() -> None:
    """Nettoie les tables brutes, écrit les contrôles, puis construit les variables."""
    run_cleaning()
    run_features()


# Une source en panne au téléchargement n'empêche pas la prévision du soir : les autres
# sources sont enregistrées, la suite tourne sur les dernières données disponibles (leur
# fraîcheur est contrôlée par --clean), et le code de sortie signale quand même l'échec.
CONTINUE_AFTER_FETCH_ERROR = {"fetch"}

# Codes de sortie. Le workflow du soir les distingue : avec 3, les données sont enregistrées
# (une panne du Trésor ne doit pas bloquer la prévision) ; avec 1, rien n'est enregistré.
BLOCKING_ERROR_EXIT = 1  # une étape indispensable a échoué : la suite n'a pas tourné
PARTIAL_FETCH_EXIT = 3  # des sources ont échoué au téléchargement, tout le reste a tourné

# L'ordre de ce dictionnaire est l'ordre d'exécution de --all.
STEPS: dict[str, tuple[str, StepFunction]] = {
    "fetch": ("télécharger toutes les sources dans data/raw/", _run_fetch),
    "clean": ("nettoyer les données, lancer les contrôles, construire les variables", _run_clean),
    "backtest": ("validation walk-forward, interprétation et figures", _run_backtest),
    "strategy": ("lien avec le marché et stratégie illustrative", _run_strategy),
    "new_data": ("tester les nouvelles données, avec et sans, en walk-forward", _run_new_data),
    "train": ("entraîner les modèles finaux sur tout l'historique", _run_train),
    "predict": ("prévoir les enchères à venir et mettre à jour le journal figé", _run_predict),
}


def build_parser() -> argparse.ArgumentParser:
    """Décrit les options de la ligne de commande."""
    parser = argparse.ArgumentParser(
        prog="python -m tresor.pipeline",
        description="Prévision des adjudications du Trésor américain : pipeline complet.",
    )
    parser.add_argument("--all", action="store_true", help="lancer toutes les étapes, dans l'ordre")
    for name, (description, _) in STEPS.items():
        flag = name.replace("_", "-")  # --new-data
        parser.add_argument(f"--{flag}", dest=name, action="store_true", help=description)
    return parser


def select_steps(args: argparse.Namespace) -> list[str]:
    """Garde l'ordre logique du pipeline, quel que soit l'ordre des options."""
    return [name for name in STEPS if args.all or getattr(args, name)]


def main(argv: Sequence[str] | None = None) -> int:
    """Lance les étapes demandées et renvoie le code de sortie du programme."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    selected_steps = select_steps(parser.parse_args(argv))
    if not selected_steps:
        parser.error("choisis au moins une étape (--all, --fetch, --clean, ...)")
    failed = False
    for name in selected_steps:
        logger.info("Étape : %s", name)
        try:
            STEPS[name][1]()
        except FetchError as error:
            logger.error("Étape « %s » : %s", name, error)
            if name not in CONTINUE_AFTER_FETCH_ERROR:
                return BLOCKING_ERROR_EXIT
            failed = True  # on continue avec les dernières données enregistrées
        except Exception:
            # Erreur imprévue : on garde le détail pour comprendre, et on s'arrête proprement.
            logger.exception("Étape « %s » : erreur inattendue", name)
            return BLOCKING_ERROR_EXIT
    return PARTIAL_FETCH_EXIT if failed else 0


if __name__ == "__main__":
    sys.exit(main())
