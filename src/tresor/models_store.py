"""Modèles entraînés hors de git : archive déposée par le passage du soir dans une « release »
GitHub, téléchargée par l'application en ligne quand elle n'a pas de modèles sur place.

Les modèles (8,6 Mo, réentraînés si les données ont changé) ne sont pas commités : l'historique
du dépôt grossirait de plusieurs gigaoctets par an. Le passage du soir dépose l'archive sous un
nom tiré de son empreinte SHA-256, puis commite un petit fichier (`models_release.json`) qui
donne ce nom et cette empreinte. L'application télécharge exactement l'archive annoncée, vérifie
l'empreinte et garde les modèles extraits : ils ne sont retéléchargés que s'ils changent.

`python -m tresor.models_store pack data/models dossier` prépare l'archive et le fichier
d'empreinte (étape du workflow).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path

import httpx

from tresor import live
from tresor.config import GITHUB_REPOSITORY, MODELS_DIR, MODELS_RELEASE_TAG, OUTPUT_DIR
from tresor.http import build_client

MANIFEST_FILE = "models_release.json"  # nom et empreinte de l'archive en ligne (commité)
MAX_ARCHIVE_BYTES = 50_000_000  # garde-fou : l'archive fait environ 8,6 Mo
MAX_UNPACKED_BYTES = 200_000_000  # garde-fou contre une archive qui « explose » à l'extraction
DOWNLOADED_DIR = Path(tempfile.gettempdir()) / "tresor-modeles"
ASSET_PATTERN = re.compile(r"modeles-[0-9a-f]{16}\.tar\.gz")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def asset_url(asset: str, repository: str = GITHUB_REPOSITORY) -> str:
    return f"https://github.com/{repository}/releases/download/{MODELS_RELEASE_TAG}/{asset}"


def archive(models_dir: Path, path: Path) -> Path:
    """Archive les modèles et leurs métadonnées, sans date ni propriétaire : des modèles
    identiques donnent une archive identique, donc la même empreinte."""
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        with tarfile.open(fileobj=packed, mode="w", format=tarfile.PAX_FORMAT) as bundle:
            for file in sorted(models_dir.iterdir()):
                info = tarfile.TarInfo(file.name)
                info.size = file.stat().st_size
                info.mode = 0o644
                with file.open("rb") as content:
                    bundle.addfile(info, content)
    return path


def pack(models_dir: Path, out_dir: Path) -> dict[str, object]:
    """Archive nommée d'après son empreinte, et le fichier qui l'annonce, dans `out_dir`.

    Le fichier ne contient pas de date : des modèles identiques ne changent rien au dépôt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    draft = archive(models_dir, out_dir / "modeles.tar.gz.part")
    digest = hashlib.sha256(draft.read_bytes()).hexdigest()
    asset = f"modeles-{digest[:16]}.tar.gz"
    draft.replace(out_dir / asset)
    manifest: dict[str, object] = {
        "asset": asset,
        "sha256": digest,
        "bytes": (out_dir / asset).stat().st_size,
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_manifest(path: Path = OUTPUT_DIR / MANIFEST_FILE) -> dict[str, str]:
    """Nom et empreinte de l'archive annoncée ; refuse un fichier absent ou mal formé."""
    content = json.loads(path.read_text(encoding="utf-8"))
    asset, digest = content.get("asset"), content.get("sha256")
    if not (isinstance(asset, str) and ASSET_PATTERN.fullmatch(asset)):
        raise ValueError(f"{path.name} : nom d'archive invalide")
    if not (isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest)):
        raise ValueError(f"{path.name} : empreinte invalide")
    if not asset.startswith(f"modeles-{digest[:16]}"):
        raise ValueError(f"{path.name} : le nom de l'archive ne correspond pas à l'empreinte")
    return {"asset": asset, "sha256": digest}


def _fetch_capped(client: httpx.Client, url: str) -> bytes:
    """Télécharge en flux et abandonne dès que la taille dépasse le garde-fou."""
    buffer = io.BytesIO()
    with client.stream("GET", url) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            buffer.write(chunk)
            if buffer.tell() > MAX_ARCHIVE_BYTES:
                raise ValueError("archive des modèles anormalement grosse : refusée")
    return buffer.getvalue()


def _extract(content: bytes, folder: Path) -> None:
    """Extrait une archive plate de fichiers ordinaires, de taille raisonnable."""
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            if not member.isfile() or "/" in member.name or member.name in ("", ".", ".."):
                raise ValueError(f"archive des modèles : entrée refusée ({member.name})")
        if sum(member.size for member in members) > MAX_UNPACKED_BYTES:
            raise ValueError("archive des modèles anormalement grosse une fois extraite")
        bundle.extractall(folder, filter="data")  # refuse aussi les chemins hors du dossier


def download(
    manifest: dict[str, str],
    root: Path = DOWNLOADED_DIR,
    client: httpx.Client | None = None,
    repository: str = GITHUB_REPOSITORY,
) -> Path:
    """Modèles de l'archive annoncée, téléchargés une seule fois par empreinte."""
    target = root / manifest["sha256"][:16]
    if (target / live.METADATA_FILE).exists():
        return target  # déjà là : le dossier n'apparaît qu'une fois l'extraction complète
    own_client = client if client is not None else build_client(timeout=30.0)
    try:
        content = _fetch_capped(own_client, asset_url(manifest["asset"], repository))
    finally:
        if client is None:
            own_client.close()
    if hashlib.sha256(content).hexdigest() != manifest["sha256"]:
        raise ValueError("empreinte de l'archive des modèles différente de celle annoncée")
    staging = root / f"{target.name}.part"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    _extract(content, staging)
    if not (staging / live.METADATA_FILE).exists():
        raise ValueError("archive des modèles incomplète")
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    for old in root.iterdir():  # les versions précédentes ne servent plus
        if old != target:
            shutil.rmtree(old, ignore_errors=True)
    return target


def models_dir_for_app(
    local: Path = MODELS_DIR, manifest: Path = OUTPUT_DIR / MANIFEST_FILE
) -> Path:
    """Les modèles sur place s'ils existent (ordinateur, passage du soir), sinon ceux de la
    release, téléchargés (application en ligne)."""
    if (local / live.METADATA_FILE).exists():
        return local
    return download(read_manifest(manifest))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 3 or arguments[0] != "pack":
        print("usage : python -m tresor.models_store pack dossier_des_modeles dossier_de_sortie")
        return 2
    if not (Path(arguments[1]) / live.METADATA_FILE).exists():
        print(f"Pas de modèles dans {arguments[1]}")
        return 1
    manifest = pack(Path(arguments[1]), Path(arguments[2]))
    print(manifest["asset"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
