"""L'application relit le code du projet quand un de ses fichiers a changé."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
spec = importlib.util.spec_from_file_location("fresh_project", APP / "fresh_project.py")
assert spec is not None and spec.loader is not None
fresh_project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fresh_project)


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Petit paquet `paquet_essai` avec un sous-module, importable depuis tmp_path."""
    folder = tmp_path / "paquet_essai"
    folder.mkdir()
    # Comme le paquet tresor : il note son empreinte quand il est chargé.
    (folder / "__init__.py").write_text(
        "from pathlib import Path\n"
        "_source_fingerprint = tuple((str(f), f.stat().st_mtime_ns, f.stat().st_size)\n"
        "    for f in sorted(Path(__file__).resolve().parent.rglob('*.py')))\n"
    )
    (folder / "valeur.py").write_text("NOMBRE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in [n for n in sys.modules if n.startswith("paquet_essai")]:
        sys.modules.pop(name)


def change(path: Path, text: str) -> None:
    """Réécrit le fichier et avance sa date (deux écritures peuvent tomber dans la même ns)."""
    before = path.stat().st_mtime_ns
    path.write_text(text)
    os.utime(path, ns=(before + 10**9, before + 10**9))


def test_unchanged_code_is_not_reloaded(package: Path) -> None:
    fresh_project.load_fresh("paquet_essai", package)
    first = importlib.import_module("paquet_essai.valeur")

    fresh_project.load_fresh("paquet_essai", package)

    assert importlib.import_module("paquet_essai.valeur") is first


def test_changed_code_is_read_again_whole(package: Path) -> None:
    fresh_project.load_fresh("paquet_essai", package)
    assert importlib.import_module("paquet_essai.valeur").NOMBRE == 1

    change(package / "paquet_essai" / "valeur.py", "NOMBRE = 2\n")
    fresh_project.load_fresh("paquet_essai", package)

    assert importlib.import_module("paquet_essai.valeur").NOMBRE == 2


def test_a_package_loaded_before_without_fingerprint_is_read_again(package: Path) -> None:
    """Cas de l'application déjà lancée avant cette mise à jour (paquet sans empreinte)."""
    old = importlib.import_module("paquet_essai.valeur")
    del sys.modules["paquet_essai"]._source_fingerprint

    fresh_project.load_fresh("paquet_essai", package)

    assert importlib.import_module("paquet_essai.valeur") is not old


def test_the_project_and_the_app_compute_the_same_fingerprint() -> None:
    """Sinon l'application relirait le code à chaque clic."""
    import tresor

    source = Path(tresor.__file__).resolve().parents[1]

    assert tresor._source_fingerprint == fresh_project.source_fingerprint(source / "tresor")
