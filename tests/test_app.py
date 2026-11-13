"""Chaque page de l'application tourne sans erreur sur les données du dépôt (sans navigateur)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tresor import app_ui, refresh
from tresor.config import OUTPUT_DIR

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = {
    "prochaines": "Prochaines enchères",
    "suivi": "Suivi en direct",
    "backtest": "Backtest",
    "explorer": "Explorer",
    "qualite": "Qualité des données",
    "methodologie": "Méthodologie",
}
pytestmark = pytest.mark.skipif(
    not (OUTPUT_DIR / "predictions_log.parquet").exists(), reason="données du dépôt absentes"
)


@pytest.mark.parametrize(("page", "title"), PAGES.items())
def test_each_page_runs(page: str, title: str) -> None:
    app = AppTest.from_file(str(APP / "vues" / f"{page}.py"), default_timeout=60).run()

    assert not app.exception, app.exception
    assert app.title[0].value == title


def test_the_main_app_opens_on_the_upcoming_auctions() -> None:
    app = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()

    assert not app.exception, app.exception
    assert app.title[0].value == "Prochaines enchères"
    assert any(button.label == "Mettre à jour maintenant" for button in app.button)


def test_the_explorer_follows_the_selection() -> None:
    app = AppTest.from_file(str(APP / "vues" / "explorer.py"), default_timeout=60).run()
    app.selectbox[0].set_value("bill").run()
    app.selectbox[1].set_value("26W").run()
    app.selectbox[2].set_value("dealer_share").run()

    assert not app.exception, app.exception


def test_the_backtest_tabs_follow_the_selection() -> None:
    app = AppTest.from_file(str(APP / "vues" / "backtest.py"), default_timeout=60).run()
    app.selectbox[0].set_value("tail_bp").run()
    app.selectbox[1].set_value("lightgbm").run()
    app.selectbox[2].set_value("dealer_share").run()

    assert not app.exception, app.exception


def test_the_live_page_shows_realised_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avant la première enchère réalisée, on simule des résultats publiés."""
    tracking = pd.read_parquet(OUTPUT_DIR / "live_tracking.parquet")
    tracking["actual"] = tracking["bench_mean3"] + 0.05
    tracking["status"] = np.where(tracking["cusip"] == tracking["cusip"].iloc[0], "en_attente",
                                  "realisee")  # fmt: skip
    tracking["error"] = tracking["actual"] - tracking["pred"]
    tracking["bench_error"] = tracking["actual"] - tracking["bench_mean3"]
    original = app_ui.parquet
    monkeypatch.setattr(
        app_ui, "parquet",
        lambda name, folder="output": tracking if name == "live_tracking.parquet"
        else original(name, folder),
    )  # fmt: skip

    app = AppTest.from_file(str(APP / "vues" / "suivi.py"), default_timeout=60).run()

    assert not app.exception, app.exception
    auctions = tracking.drop_duplicates(["cusip", "auction_date"])
    assert app.metric[1].value == str(len(auctions) - 1)  # toutes réalisées sauf une
    assert len(app.get("plotly_chart")) == 2  # prévu contre réalisé, gain cumulé


def test_the_update_button_shows_a_provisional_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    log = pd.read_parquet(OUTPUT_DIR / "predictions_log.parquet").drop(columns="predicted_at")
    computed = pd.Timestamp("2026-11-13 18:00", tz="UTC")
    monkeypatch.setattr(refresh, "light_refresh",
                        lambda: (log.assign(computed_at=computed), None))  # fmt: skip
    monkeypatch.setattr(app_ui, "today_new_york", lambda: pd.Timestamp("2026-11-13"))

    app = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()
    next(b for b in app.button if b.label == "Mettre à jour maintenant").click().run()

    assert not app.exception, app.exception
    assert "provisional" in app.session_state
    assert any("provisoire" in expander.label for expander in app.expander)


def test_the_update_button_without_models_explains_why(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> None:
        raise FileNotFoundError("modèles absents")

    monkeypatch.setattr(refresh, "light_refresh", missing)

    app = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()
    next(b for b in app.button if b.label == "Mettre à jour maintenant").click().run()

    assert not app.exception, app.exception
    assert any("Modèles absents" in warning.value for warning in app.sidebar.warning)
