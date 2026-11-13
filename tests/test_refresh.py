"""Tests du bouton « Mettre à jour maintenant » (sans réseau)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from tests.test_live import fast_model

from tresor import features as ft
from tresor import live, refresh
from tresor import models as md
from tresor.config import CLEAN_DIR, DATA_DIR, RAW_DIR
from tresor.http import FetchError

NOW = pd.Timestamp("2026-11-13 23:00", tz="UTC")
CHANGED = "912810UZ9"  # le 20 ans du 18 novembre 2026
pytestmark = pytest.mark.skipif(
    not (CLEAN_DIR / "features.parquet").exists(), reason="données du dépôt absentes"
)


def fingerprints() -> dict[str, str]:
    """Empreinte de chaque fichier de data/ (hors modèles, ignorés par git)."""
    return {
        str(path.relative_to(DATA_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(DATA_DIR.rglob("*"))
        if path.is_file() and "models" not in path.parts
    }


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Modèles rapides entraînés sur les vraies variables, dans un dossier temporaire."""
    monkeypatch.setattr(md, "make_model", fast_model)
    sample = ft.model_sample(pd.read_parquet(CLEAN_DIR / "features.parquet"))
    live.train_models(sample, tmp_path / "models", n_jobs=1)
    return tmp_path / "models"


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Remplace le téléchargement (pas de réseau) et note l'ordre des appels."""
    recorded: list[object] = []
    original = live.load_models

    def load(directory: Path) -> object:
        recorded.append("modèles")
        return original(directory)

    def collect(sources: list[object], client: object, raw_dir: Path) -> list[object]:
        recorded.append(("téléchargement", [s.name for s in sources], raw_dir, client))
        # Le « téléchargement » change le montant annoncé d'une enchère, dans la copie.
        path = Path(raw_dir) / "treasurydirect_upcoming.parquet"
        upcoming = pd.read_parquet(path)
        upcoming.loc[upcoming["cusip"] == CHANGED, "offeringAmount"] = "1000000000"
        upcoming.to_parquet(path, index=False)
        return []

    monkeypatch.setattr(live, "load_models", load)
    monkeypatch.setattr(refresh, "collect_all", collect)
    return recorded


def test_the_button_never_touches_the_project_files(models_dir: Path, calls: list) -> None:
    before = fingerprints()

    predictions, failure = refresh.light_refresh(now=NOW, models_dir=models_dir)

    assert fingerprints() == before  # ni données, ni journal figé, ni suivi
    assert failure is None
    assert calls[0] == "modèles"  # vérifiés avant tout téléchargement
    _, names, raw_dir, client = calls[1]
    assert sorted(names) == sorted(refresh.LIGHT_SOURCES)  # pas de FRED ni d'autre source
    assert DATA_DIR not in Path(raw_dir).parents and raw_dir != RAW_DIR
    assert not Path(raw_dir).exists()  # dossier temporaire supprimé
    assert client.timeout.read == refresh.TIMEOUT_SECONDS
    assert not predictions.empty
    assert (predictions["auction_date"] > pd.Timestamp("2026-11-13")).all()
    assert set(predictions["model"]) == set(live.LIVE_MODELS)
    assert (predictions["computed_at"] == NOW).all()
    # Les variables ont bien été refaites à partir de la copie mise à jour.
    assert set(predictions.loc[predictions["cusip"] == CHANGED, "offering_amt"]) == {1e9}


def test_without_models_nothing_is_downloaded(tmp_path: Path, calls: list) -> None:
    with pytest.raises(refresh.ModelsUnavailable):
        refresh.light_refresh(now=NOW, models_dir=tmp_path)

    assert calls == ["modèles"]


def test_a_failed_source_still_gives_a_prediction(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(sources: list[object], client: object, raw_dir: Path) -> list[object]:
        raise FetchError("Sources en échec : treasurydirect_upcoming")

    monkeypatch.setattr(refresh, "collect_all", broken)

    predictions, failure = refresh.light_refresh(now=NOW, models_dir=models_dir)

    assert failure == "Sources en échec : treasurydirect_upcoming"
    assert not predictions.empty


def test_a_second_click_during_an_update_is_refused(models_dir: Path) -> None:
    assert refresh._RUNNING.acquire(blocking=False)
    try:
        with pytest.raises(refresh.RefreshBusy):
            refresh.light_refresh(now=NOW, models_dir=models_dir)
    finally:
        refresh._RUNNING.release()
