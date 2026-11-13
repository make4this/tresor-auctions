"""Mise à jour légère, pour le bouton « Mettre à jour maintenant » de l'application.

Elle retélécharge seulement les sources légères (TreasuryDirect récent et calendrier, courbe
des taux et taux des bills de l'année en cours), refait la préparation des données et prévoit
les enchères à venir avec les modèles déjà entraînés, sans ré-entraîner.

Elle n'écrit pas le journal figé : seul le workflow de nuit l'écrit. La prévision obtenue est
provisoire et affichée comme telle.
"""

from __future__ import annotations

import pandas as pd

from tresor import live
from tresor.clean.run import run_cleaning
from tresor.config import CLEAN_DIR, MODELS_DIR
from tresor.features import run_features
from tresor.http import FetchError
from tresor.sources.collect import RAW_SOURCES, collect_all

LIGHT_SOURCES = (
    "treasurydirect_recent",
    "treasurydirect_upcoming",
    "treasury_yield_curve",
    "treasury_bill_rates",
)


def light_refresh(now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, str | None]:
    """Prévisions provisoires des enchères à venir, et le message d'échec éventuel d'une source.

    Les modèles sont vérifiés avant tout téléchargement : sans eux, rien n'est modifié.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    models = live.load_models(MODELS_DIR)
    failure = None
    try:
        collect_all([source for source in RAW_SOURCES if source.name in LIGHT_SOURCES])
    except FetchError as error:  # les autres sources ont été enregistrées
        failure = str(error)
    run_cleaning()
    run_features()
    features = pd.read_parquet(CLEAN_DIR / "features.parquet")
    rows = live.upcoming_rows(features, live.new_york_today(now))
    return live.predict_upcoming(rows, models).assign(computed_at=now), failure
