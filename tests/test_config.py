"""Tests de la lecture de la clé FRED."""

from __future__ import annotations

from pathlib import Path

import pytest

from tresor import config


@pytest.fixture
def project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Dossier vide à la place du vrai dépôt, pour ne jamais lire ton vrai fichier .env."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    # setenv puis delenv : pytest remettra l'état initial, même si load_dotenv crée la variable.
    monkeypatch.setenv("FRED_API_KEY", "")
    monkeypatch.delenv("FRED_API_KEY")
    return tmp_path


def test_missing_key_gives_a_clear_error(project_root: Path) -> None:
    with pytest.raises(RuntimeError, match="FRED_API_KEY manquante"):
        config.get_fred_api_key()


def test_key_is_read_from_dotenv_file(project_root: Path) -> None:
    (project_root / ".env").write_text("FRED_API_KEY=depuis_fichier\n", encoding="utf-8")

    assert config.get_fred_api_key() == "depuis_fichier"


def test_environment_takes_priority_over_dotenv_file(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (project_root / ".env").write_text("FRED_API_KEY=depuis_fichier\n", encoding="utf-8")
    monkeypatch.setenv("FRED_API_KEY", "  depuis_secret_github  ")

    assert config.get_fred_api_key() == "depuis_secret_github"
