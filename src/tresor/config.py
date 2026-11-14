"""Paramètres centraux du projet : chemins, dates, maturités, réseau et clé FRED."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Final, Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# config.py est dans src/tresor/ : on remonte de deux dossiers pour trouver la racine du dépôt.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"  # réponses brutes des API, jamais modifiées
CLEAN_DIR: Final[Path] = DATA_DIR / "clean"  # tables nettoyées
OUTPUT_DIR: Final[Path] = DATA_DIR / "output"  # prédictions, métriques, contrôles
CACHE_DIR: Final[Path] = DATA_DIR / "cache"  # téléchargements qui ne changent plus
MODELS_DIR: Final[Path] = DATA_DIR / "models"  # modèles finaux entraînés (joblib)
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "reports" / "figures"

# Les heures d'enchère et le statut « à venir » se raisonnent à l'heure de New York.
NEW_YORK_TZ: Final[ZoneInfo] = ZoneInfo("America/New_York")

# Dépôt GitHub du projet : le passage du soir y dépose les modèles entraînés dans une
# « release », que l'application en ligne télécharge (l'historique git ne grossit pas).
GITHUB_REPOSITORY: Final[str] = os.environ.get(
    "TRESOR_GITHUB_REPOSITORY", "make4this/tresor-auctions"
)
MODELS_RELEASE_TAG: Final[str] = "modeles-du-soir"
MODELS_ARCHIVE: Final[str] = "modeles.tar.gz"

# Signal de l'application (section 11 du plan) : bid-to-cover prévu moins la moyenne des 3
# dernières enchères. Demande plus faible si l'écart est <= -0,08, plus forte s'il est >= +0,08.
SIGNAL_THRESHOLD: Final[float] = 0.08

HISTORY_START: Final[date] = date(2000, 1, 1)  # début de la collecte
MODEL_SAMPLE_START: Final[date] = date(2009, 1, 1)  # le 7 ans reprend en 2009
WALK_FORWARD_START: Final[date] = date(2014, 1, 1)  # premier trimestre testé

# Maturités standard vers lesquelles on arrondit la maturité résiduelle (section 6.1 du plan).
COUPON_MATURITIES_YEARS: Final[tuple[int, ...]] = (2, 3, 5, 7, 10, 20, 30)
BILL_MATURITIES_WEEKS: Final[tuple[int, ...]] = (4, 6, 8, 13, 17, 26, 52)

Family = Literal["bill", "cmb", "nominal_coupon", "tips", "frn"]

# En dessous de ce montant, ce sont de petites émissions hors calendrier : on les exclut.
MIN_OFFERING_AMOUNT_USD: Final[float] = 1_000_000_000.0

RANDOM_STATE: Final[int] = 42  # même graine partout, pour des résultats reproductibles

HTTP_TIMEOUT_SECONDS: Final[float] = 60.0
HTTP_MAX_ATTEMPTS: Final[int] = 5
HTTP_BACKOFF_BASE_SECONDS: Final[float] = 1.0
HTTP_USER_AGENT: Final[str] = "tresor-auctions/0.1 (projet academique)"


def get_fred_api_key() -> str:
    """Renvoie la clé FRED, lue dans l'environnement ou, en local, dans le fichier .env."""
    # Sans override, une variable déjà définie (secret GitHub ou Streamlit) reste prioritaire.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FRED_API_KEY manquante : copie .env.example vers .env et renseigne ta clé."
        )
    return key
