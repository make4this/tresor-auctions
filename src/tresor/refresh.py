"""Mise à jour légère, pour le bouton « Mettre à jour maintenant » de l'application.

Elle retélécharge seulement les sources légères (TreasuryDirect récent et calendrier, courbe
des taux et taux des bills de l'année en cours), refait la préparation des données et prévoit
les enchères à venir avec les modèles déjà entraînés, sans ré-entraîner.

Tout se fait sur une copie des données, dans un dossier temporaire supprimé ensuite : les
fichiers du dépôt ne sont jamais modifiés, et surtout pas le journal figé, que seul le passage
de nuit écrit. La prévision obtenue est provisoire et affichée comme telle.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import pandas as pd

from tresor import live
from tresor.clean.run import run_cleaning
from tresor.config import MODELS_DIR, RAW_DIR
from tresor.features import run_features
from tresor.http import FetchError, build_client
from tresor.sources.collect import RAW_SOURCES, collect_all

LIGHT_SOURCES = (
    "treasurydirect_recent",
    "treasurydirect_upcoming",
    "treasury_yield_curve",
    "treasury_bill_rates",
)
TIMEOUT_SECONDS = 20.0  # l'utilisateur attend devant l'écran : on abandonne plus vite
_RUNNING = threading.Lock()  # une seule mise à jour à la fois pour toute l'application


class ModelsUnavailable(RuntimeError):
    """Pas de modèles utilisables ici (absents, ou entraînés avec d'autres versions)."""


class RefreshBusy(RuntimeError):
    """Une autre mise à jour est déjà en cours."""


def light_refresh(
    now: pd.Timestamp | None = None,
    models_dir: Path = MODELS_DIR,
    raw_dir: Path = RAW_DIR,
) -> tuple[pd.DataFrame, str | None]:
    """Prévisions provisoires des enchères à venir, et le message d'échec éventuel d'une source.

    Les modèles sont vérifiés avant tout téléchargement : sans eux, rien n'est fait.
    """
    if not _RUNNING.acquire(blocking=False):
        raise RefreshBusy("une mise à jour est déjà en cours")
    try:
        now = pd.Timestamp.now(tz="UTC") if now is None else now
        try:
            models = live.load_models(models_dir)
        except (FileNotFoundError, RuntimeError) as error:
            raise ModelsUnavailable(str(error)) from error
        with tempfile.TemporaryDirectory(prefix="tresor-maj-") as folder:
            work = Path(folder)
            shutil.copytree(raw_dir, work / "raw")
            failure = None
            sources = [source for source in RAW_SOURCES if source.name in LIGHT_SOURCES]
            with build_client(timeout=TIMEOUT_SECONDS) as client:
                try:
                    collect_all(sources, client=client, raw_dir=work / "raw")
                except FetchError as error:  # les autres sources ont été enregistrées
                    failure = str(error)
            # Le statut « à venir » est calculé à la même date que la prévision.
            today = live.new_york_today(now)
            run_cleaning(raw_dir=work / "raw", clean_dir=work / "clean",
                         output_dir=work / "out", today=today.date())  # fmt: skip
            run_features(clean_dir=work / "clean", output_dir=work / "out")
            features = pd.read_parquet(work / "clean" / "features.parquet")
        rows = live.upcoming_rows(features, today)
        return live.predict_upcoming(rows, models).assign(computed_at=now), failure
    finally:
        _RUNNING.release()
