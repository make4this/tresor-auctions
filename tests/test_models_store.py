"""Tests de l'archive des modèles déposée dans la release GitHub."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import httpx
import pytest
from tests.test_live import trained  # noqa: F401  (fixture : modèles rapides entraînés)

from tresor import live, models_store
from tresor.http import build_client


def serve(content: bytes, status: int = 200) -> httpx.Client:
    """Faux serveur GitHub : renvoie toujours le même contenu."""
    return build_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, content=content))
    )


def test_models_go_through_the_release_and_come_back(trained: Path, tmp_path: Path) -> None:  # noqa: F811
    bundle = models_store.archive(trained / "models", tmp_path / "modeles.tar.gz")

    folder = models_store.download(tmp_path / "telecharges", client=serve(bundle.read_bytes()))

    assert sorted(p.name for p in folder.iterdir()) == sorted(
        p.name for p in (trained / "models").iterdir()
    )
    assert len(live.load_models(folder)) == 2 * 4 * 5  # versions et variables vérifiées


def test_the_release_address() -> None:
    assert models_store.release_url("moi/projet") == (
        "https://github.com/moi/projet/releases/download/modeles-du-soir/modeles.tar.gz"
    )


def tar_with(name: str, content: bytes = b"x") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        bundle.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_an_archive_writing_outside_its_folder_is_refused(tmp_path: Path) -> None:
    with pytest.raises(tarfile.TarError):
        models_store.download(tmp_path / "dossier", client=serve(tar_with("../evade.txt")))

    assert not (tmp_path / "evade.txt").exists()


def test_an_incomplete_or_huge_archive_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="incomplète"):
        models_store.download(tmp_path / "dossier", client=serve(tar_with("modele.joblib")))
    monkeypatch.setattr(models_store, "MAX_ARCHIVE_BYTES", 10)
    with pytest.raises(ValueError, match="grosse"):
        models_store.download(tmp_path / "dossier", client=serve(tar_with(live.METADATA_FILE)))


def test_a_missing_release_raises(tmp_path: Path) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        models_store.download(tmp_path / "dossier", client=serve(b"", status=404))


def test_local_models_are_preferred(trained: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    monkeypatch.setattr(models_store, "download", lambda: pytest.fail("téléchargement inutile"))

    assert models_store.models_dir_for_app(trained / "models") == trained / "models"
