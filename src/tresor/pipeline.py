"""Orchestration du projet : `python -m tresor.pipeline --all`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from tresor.clean.run import run_cleaning
from tresor.http import FetchError
from tresor.sources.collect import collect_all

logger = logging.getLogger(__name__)

StepFunction = Callable[[], None]


def _run_fetch() -> None:
    """Télécharge les sources et enregistre les tables brutes."""
    collect_all()


def _run_clean() -> None:
    """Nettoie les tables brutes et écrit les comptes rendus de nettoyage et de qualité."""
    run_cleaning()


def _pending(phase: int) -> StepFunction:
    """Étape pas encore écrite : elle échoue clairement au lieu de faire semblant de réussir."""

    def run() -> None:
        raise NotImplementedError(f"pas encore disponible (prévue en phase {phase})")

    return run


# L'ordre de ce dictionnaire est l'ordre d'exécution de --all.
STEPS: dict[str, tuple[str, StepFunction]] = {
    "fetch": ("télécharger toutes les sources dans data/raw/", _run_fetch),
    "clean": ("nettoyer les données et lancer les contrôles", _run_clean),
    "backtest": ("validation walk-forward et stratégie", _pending(6)),
    "train": ("entraîner les modèles finaux sur tout l'historique", _pending(8)),
    "predict": ("prévoir les enchères à venir", _pending(8)),
}


def build_parser() -> argparse.ArgumentParser:
    """Décrit les options de la ligne de commande."""
    parser = argparse.ArgumentParser(
        prog="python -m tresor.pipeline",
        description="Prévision des adjudications du Trésor américain : pipeline complet.",
    )
    parser.add_argument("--all", action="store_true", help="lancer toutes les étapes, dans l'ordre")
    for name, (description, _) in STEPS.items():
        parser.add_argument(f"--{name}", action="store_true", help=description)
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
    for name in selected_steps:
        logger.info("Étape : %s", name)
        try:
            STEPS[name][1]()
        except (NotImplementedError, FetchError) as error:
            logger.error("Étape « %s » : %s", name, error)
            return 1
        except Exception:
            # Erreur imprévue : on garde le détail pour comprendre, et on s'arrête proprement.
            logger.exception("Étape « %s » : erreur inattendue", name)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
