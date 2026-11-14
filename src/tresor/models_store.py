"""Modèles entraînés hors de git : archive déposée par le passage du soir dans une « release »
GitHub, téléchargée par l'application en ligne quand elle n'a pas de modèles sur place.

Les modèles (8,6 Mo, réentraînés chaque soir) ne sont pas commités : l'historique du dépôt
grossirait d'environ 1,6 Go par an. Une release garde seulement la dernière version.
"""

from __future__ import annotations

import io
import shutil
import tarfile
import tempfile
from pathlib import Path

import httpx

from tresor import live
from tresor.config import GITHUB_REPOSITORY, MODELS_ARCHIVE, MODELS_DIR, MODELS_RELEASE_TAG
from tresor.http import build_client

MAX_ARCHIVE_BYTES = 50_000_000  # garde-fou : l'archive fait environ 3 Mo
DOWNLOADED_DIR = Path(tempfile.gettempdir()) / "tresor-modeles"


def release_url(repository: str = GITHUB_REPOSITORY) -> str:
    return (
        f"https://github.com/{repository}/releases/download/{MODELS_RELEASE_TAG}/{MODELS_ARCHIVE}"
    )


def archive(models_dir: Path, path: Path) -> Path:
    """Archive les modèles et leurs métadonnées (étape du passage du soir)."""
    with tarfile.open(path, "w:gz") as bundle:
        for file in sorted(models_dir.iterdir()):
            bundle.add(file, arcname=file.name)
    return path


def download(
    target: Path = DOWNLOADED_DIR,
    url: str | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """Télécharge et extrait l'archive des modèles ; remplace le dossier d'un seul coup."""
    own_client = client if client is not None else build_client(timeout=30.0)
    try:
        response = own_client.get(url or release_url())
        response.raise_for_status()
    finally:
        if client is None:
            own_client.close()
    if len(response.content) > MAX_ARCHIVE_BYTES:
        raise ValueError("archive des modèles anormalement grosse : refusée")
    staging = target.with_name(f"{target.name}.part")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as bundle:
        bundle.extractall(staging, filter="data")  # refuse les chemins hors du dossier
    if not (staging / live.METADATA_FILE).exists():
        raise ValueError("archive des modèles incomplète")
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    return target


def models_dir_for_app(local: Path = MODELS_DIR) -> Path:
    """Les modèles sur place s'ils existent (ordinateur, passage du soir), sinon ceux de la
    release, téléchargés (application en ligne)."""
    if (local / live.METADATA_FILE).exists():
        return local
    return download()
