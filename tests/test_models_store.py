"""Tests de l'archive des modèles déposée dans la release GitHub."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import httpx
import pytest
from tests.test_live import trained  # noqa: F401  (fixture : modèles rapides entraînés)

from tresor import live, models_store
from tresor.http import build_client


def serve(content: bytes, status: int = 200, seen: list[str] | None = None) -> httpx.Client:
    """Faux serveur GitHub : renvoie toujours le même contenu et note les adresses demandées."""

    def answer(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(str(request.url))
        return httpx.Response(status, content=content)

    return build_client(transport=httpx.MockTransport(answer))


def manifest_for(content: bytes) -> dict[str, str]:
    digest = hashlib.sha256(content).hexdigest()
    return {"asset": f"modeles-{digest[:16]}.tar.gz", "sha256": digest}


def test_models_go_through_the_release_and_come_back(trained: Path, tmp_path: Path) -> None:  # noqa: F811
    manifest = models_store.pack(trained / "models", tmp_path / "paquet")
    bundle = (tmp_path / "paquet" / str(manifest["asset"])).read_bytes()
    seen: list[str] = []

    folder = models_store.download(
        models_store.read_manifest(tmp_path / "paquet" / models_store.MANIFEST_FILE),
        tmp_path / "telecharges",
        client=serve(bundle, seen=seen),
        repository="moi/projet",
    )

    assert seen == [
        f"https://github.com/moi/projet/releases/download/modeles-du-soir/{manifest['asset']}"
    ]
    assert manifest["sha256"] == hashlib.sha256(bundle).hexdigest()
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        p.name for p in (trained / "models").iterdir()
    )
    assert len(live.load_models(folder)) == 2 * 4 * 5  # versions et variables vérifiées


def test_the_same_models_give_the_same_archive(trained: Path, tmp_path: Path) -> None:  # noqa: F811
    first = models_store.pack(trained / "models", tmp_path / "a")
    second = models_store.pack(trained / "models", tmp_path / "b")

    assert first["sha256"] == second["sha256"]  # rien à republier si rien n'a changé


def test_models_are_downloaded_once_per_fingerprint(trained: Path, tmp_path: Path) -> None:  # noqa: F811
    old_bundle = tar_with(live.METADATA_FILE, b"{}")
    root = tmp_path / "telecharges"
    old = models_store.download(manifest_for(old_bundle), root, client=serve(old_bundle))
    new_bundle = models_store.archive(trained / "models", tmp_path / "x.tar.gz").read_bytes()
    seen: list[str] = []

    new = models_store.download(manifest_for(new_bundle), root, client=serve(new_bundle, seen=seen))
    again = models_store.download(manifest_for(new_bundle), root, client=serve(b"", seen=seen))

    assert new == again and len(seen) == 1  # le second appel ne télécharge rien
    assert not old.exists()  # l'ancienne version est remplacée
    assert sorted(p.name for p in root.iterdir()) == [new.name]


def test_an_archive_with_another_fingerprint_is_refused(tmp_path: Path) -> None:
    bundle = tar_with(live.METADATA_FILE, b"{}")
    expected = manifest_for(tar_with(live.METADATA_FILE, b"{autre}"))

    with pytest.raises(ValueError, match="empreinte"):
        models_store.download(expected, tmp_path / "dossier", client=serve(bundle))

    assert not (tmp_path / "dossier" / expected["sha256"][:16]).exists()


def tar_with(name: str, content: bytes = b"x", kind: bytes = tarfile.REGTYPE) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo(name)
        info.type = kind
        if kind == tarfile.SYMTYPE:
            info.linkname = "/etc/passwd"
            bundle.addfile(info)
        else:
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@pytest.mark.parametrize(
    "bundle",
    [
        tar_with("../evade.txt"),  # hors du dossier
        tar_with("sous/dossier.txt"),  # l'archive doit être plate
        tar_with("lien", kind=tarfile.SYMTYPE),  # lien symbolique
    ],
)
def test_an_archive_with_odd_entries_is_refused(bundle: bytes, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="entrée refusée"):
        models_store.download(manifest_for(bundle), tmp_path / "dossier", client=serve(bundle))

    assert not (tmp_path / "evade.txt").exists()


def test_an_incomplete_or_huge_archive_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete = tar_with("modele.joblib")
    with pytest.raises(ValueError, match="incomplète"):
        models_store.download(manifest_for(incomplete), tmp_path / "a", client=serve(incomplete))
    big = tar_with(live.METADATA_FILE, b"0" * 1000)
    with monkeypatch.context() as patch:
        patch.setattr(models_store, "MAX_UNPACKED_BYTES", 999)
        with pytest.raises(ValueError, match="une fois extraite"):
            models_store.download(manifest_for(big), tmp_path / "b", client=serve(big))
    monkeypatch.setattr(models_store, "MAX_ARCHIVE_BYTES", 10)
    with pytest.raises(ValueError, match="grosse : refusée"):  # dès le téléchargement
        models_store.download(manifest_for(big), tmp_path / "c", client=serve(big))


def test_a_missing_release_raises(tmp_path: Path) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        models_store.download(
            manifest_for(b""), tmp_path / "dossier", client=serve(b"", status=404)
        )


@pytest.mark.parametrize(
    "content",
    [
        {"asset": "modeles-0123456789abcdef.tar.gz"},  # sans empreinte
        {"asset": "../../autre.tar.gz", "sha256": "0" * 64},  # nom imprévu
        {"asset": "modeles-0123456789abcdef.tar.gz", "sha256": "f" * 64},  # nom ≠ empreinte
    ],
)
def test_a_bad_manifest_is_refused(content: dict[str, str], tmp_path: Path) -> None:
    path = tmp_path / models_store.MANIFEST_FILE
    path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(ValueError):
        models_store.read_manifest(path)


def test_local_models_are_preferred(trained: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    monkeypatch.setattr(models_store, "download", lambda _: pytest.fail("téléchargement inutile"))

    assert models_store.models_dir_for_app(trained / "models") == trained / "models"


def test_without_local_models_the_announced_archive_is_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / models_store.MANIFEST_FILE
    path.write_text(json.dumps(manifest_for(b"abc")), encoding="utf-8")
    asked: list[dict[str, str]] = []
    monkeypatch.setattr(models_store, "download", lambda m: asked.append(m) or tmp_path)

    assert models_store.models_dir_for_app(tmp_path / "rien", path) == tmp_path
    assert asked == [manifest_for(b"abc")]
    with pytest.raises(FileNotFoundError):
        models_store.models_dir_for_app(tmp_path / "rien", tmp_path / "absent.json")


def test_the_pack_command(
    trained: Path,  # noqa: F811
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert models_store.main(["pack", str(trained / "models"), str(tmp_path / "out")]) == 0
    asset = capsys.readouterr().out.strip()
    assert (tmp_path / "out" / asset).exists()
    assert (
        models_store.read_manifest(tmp_path / "out" / models_store.MANIFEST_FILE)["asset"] == asset
    )
    assert models_store.main(["pack", str(tmp_path / "vide"), str(tmp_path / "out")]) == 1
    assert models_store.main([]) == 2
