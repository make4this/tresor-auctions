"""Chaque page de l'application tourne et affiche les bonnes valeurs (sans navigateur)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tresor import app_data as ad
from tresor import app_ui, refresh
from tresor.config import CLEAN_DIR, OUTPUT_DIR

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = {
    "prochaines": "Prochaines enchères",
    "suivi": "Suivi en direct",
    "backtest": "Backtest",
    "explorer": "Explorer",
    "qualite": "Qualité des données",
    "methodologie": "Méthodologie",
}
TODAY = pd.Timestamp("2026-11-13")
pytestmark = pytest.mark.skipif(
    not (OUTPUT_DIR / "predictions_log.parquet").exists(), reason="données du dépôt absentes"
)


def page(name: str) -> AppTest:
    return AppTest.from_file(str(APP / "vues" / f"{name}.py"), default_timeout=60)


def main() -> AppTest:
    return AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60)


@pytest.fixture
def today(monkeypatch: pytest.MonkeyPatch) -> pd.Timestamp:
    monkeypatch.setattr(app_ui, "today_new_york", lambda: TODAY)
    return TODAY


def replace_json(monkeypatch: pytest.MonkeyPatch, name: str, content: dict) -> None:
    original = app_ui.json_file
    monkeypatch.setattr(app_ui, "json_file",
                        lambda n: content if n == name else original(n))  # fmt: skip


@pytest.mark.parametrize(("name", "title"), PAGES.items())
def test_each_page_runs(name: str, title: str) -> None:
    app = page(name).run()

    assert not app.exception, app.exception
    assert app.title[0].value == title


def test_the_main_app_opens_on_the_upcoming_auctions() -> None:
    app = main().run()

    assert not app.exception, app.exception
    assert app.title[0].value == "Prochaines enchères"
    assert any(button.label == "Mettre à jour maintenant" for button in app.button)
    assert any("pas un conseil en investissement" in c.value for c in app.sidebar.caption)


def test_the_upcoming_table_shows_the_reference_model_from_today(today: pd.Timestamp) -> None:
    log = pd.read_parquet(OUTPUT_DIR / "predictions_log.parquet")
    auctions = pd.read_parquet(CLEAN_DIR / "auctions.parquet")
    expected = ad.upcoming_table(log, auctions, today)

    app = page("prochaines").run()
    shown = app.dataframe[0].value

    assert len(shown) == len(expected) == log.loc[log["auction_date"] >= today, "cusip"].nunique()
    assert list(shown["Demande attendue"]) == list(expected["signal"])
    assert list(shown["Bid-to-cover prévu"]) == [ad.fr(v, 2) for v in expected["pred_btc"]]
    assert list(shown["Part des dealers prévue"]) == [
        ad.format_target(v, "dealer_share") for v in expected["pred_dealer_share"]
    ]
    record = ad.signal_track_record(pd.read_parquet(OUTPUT_DIR / "backtest_predictions.parquet"))
    weak = record.set_index("family").loc["nominal_coupon", "share_weak"]
    text = " ".join(m.value for m in app.markdown)
    assert f"« plus faible » sur {ad.fr(weak * 100, 0)} % des enchères" in text


def test_the_live_page_shows_realised_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avant la première enchère réalisée, on simule des résultats publiés."""
    tracking = pd.read_parquet(OUTPUT_DIR / "live_tracking.parquet")
    tracking["actual"] = tracking["bench_mean3"] + 0.05
    first = tracking["cusip"].iloc[0]
    tracking["status"] = np.where(tracking["cusip"] == first, "en_attente", "realisee")
    tracking["error"] = tracking["actual"] - tracking["pred"]
    tracking["bench_error"] = tracking["actual"] - tracking["bench_mean3"]
    original = app_ui.parquet
    monkeypatch.setattr(
        app_ui, "parquet",
        lambda name, folder="output": tracking if name == "live_tracking.parquet"
        else original(name, folder),
    )  # fmt: skip

    app = page("suivi").run()
    auctions = tracking.drop_duplicates(["cusip", "auction_date"])

    assert not app.exception, app.exception
    assert app.metric[1].value == str(len(auctions) - 1)  # toutes réalisées sauf une
    assert len(app.get("plotly_chart")) == 2  # prévu contre réalisé, gain cumulé
    app.segmented_control[0].set_value("dealer_share").run()
    table = app.dataframe[0].value
    realised = tracking[(tracking["model"] == "moyenne_5") & (tracking["target"] == "dealer_share")]
    realised = realised[realised["status"] == "realisee"].sort_values(["auction_date", "cusip"])
    assert list(table["Prévu"]) == [ad.format_target(v, "dealer_share") for v in realised["pred"]]


def test_the_explorer_follows_the_selection_without_rescaling_the_bid_to_cover() -> None:
    app = page("explorer").run()
    trace = json.loads(app.get("plotly_chart")[0].proto.spec)["data"][0]["y"]
    if isinstance(trace, dict):  # Plotly encode les valeurs en binaire
        trace = np.frombuffer(base64.b64decode(trace["bdata"]), dtype=trace["dtype"])
    features = pd.read_parquet(CLEAN_DIR / "features.parquet")
    history = features[
        (features["maturity_bucket"] == "10Y")
        & (features["status"] == "completed")
        & (features["auction_date"] >= "2009-01-01")
    ].dropna(subset=["btc"])
    assert np.allclose(sorted(trace), sorted(history["btc"]))

    app.selectbox[0].set_value("bill").run()
    app.selectbox[1].set_value("26W").run()
    app.selectbox[2].set_value("dealer_share").run()
    assert not app.exception, app.exception


def test_the_backtest_page_defaults_and_tabs() -> None:
    app = page("backtest").run()

    assert app.selectbox[1].value == "moyenne_5"  # modèle de référence par défaut
    strategy = app.dataframe[-1].value
    assert strategy["Trades"].iloc[0] == "1 004"  # période 2014-2026
    app.selectbox[0].set_value("tail_bp").run()
    app.selectbox[1].set_value("lightgbm").run()
    app.selectbox[2].set_value("dealer_share").run()
    app.segmented_control[0].set_value("bill").run()
    assert not app.exception, app.exception


def test_the_quality_page_survives_a_failing_check(monkeypatch: pytest.MonkeyPatch) -> None:
    quality = json.loads((OUTPUT_DIR / "data_quality.json").read_text())
    quality["market"]["rates_outside_bounds"] = 2
    quality["sources"]["cboe_vix"]["status"] = "rouge"
    replace_json(monkeypatch, "data_quality.json", quality)

    app = page("qualite").run()

    assert not app.exception, app.exception
    sources = table_with(app, "État").set_index("Source")
    assert sources.loc["CBOE : VIX", "État"] == "🔴 problème"
    checks = table_with(app, "Contrôle").set_index("Contrôle")
    assert checks.loc["Marché : taux hors bornes", "Cas trouvés"] == 2


def table_with(app: AppTest, column: str) -> pd.DataFrame:
    """Le tableau de la page qui a cette colonne (l'ordre des tableaux peut changer)."""
    return next(frame.value for frame in app.dataframe if column in frame.value.columns)


FETCH_FAILED = {
    "run_at": "2026-11-14T23:17:00+00:00",
    "failed": ["fiscaldata_auctions"],
    "sources": {
        "cboe_vix": {
            "ok": True,
            "rows": 9,
            "last_date": "2026-11-13",
            "last_success_at": "2026-11-14T23:17:00+00:00",
        },  # fmt: skip
        "fiscaldata_auctions": {
            "ok": False,
            "error": "FetchError : HTTP 503",
            "last_success_at": "2026-11-13T23:20:00+00:00",
        },  # fmt: skip
    },
}


def test_a_source_outage_is_shown_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    replace_json(monkeypatch, "fetch_status.json", FETCH_FAILED)

    app = page("qualite").run()
    main_app = main().run()

    assert not app.exception, app.exception
    assert any("1 source(s) en échec" in w.value and "Fiscal Data : adjudications" in w.value
               for w in app.warning)  # fmt: skip
    downloads = table_with(app, "Résultat")
    assert list(downloads["Résultat"]) == ["🔴 échec", "🟢 reçue"]  # les échecs d'abord
    assert downloads["Dernière réussite"][0] == "ven. 13/11/2026 à 18:20 (New York)"
    assert downloads["Erreur"][0] == "FetchError : HTTP 503"
    assert any("1 source(s) en échec" in c.value for c in main_app.sidebar.caption)


def test_a_complete_download_is_said_and_a_missing_report_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = {
        **FETCH_FAILED,
        "failed": [],
        "sources": {"cboe_vix": FETCH_FAILED["sources"]["cboe_vix"]},
    }
    replace_json(monkeypatch, "fetch_status.json", complete)
    app = main().run()
    assert any("toutes les sources reçues" in c.value for c in app.sidebar.caption)

    replace_json(monkeypatch, "fetch_status.json", {})
    quality = page("qualite").run()
    assert not quality.exception, quality.exception
    assert not quality.warning or all("en échec" not in w.value for w in quality.warning)
    assert any("Rapport de téléchargement absent" in c.value for c in quality.caption)


def test_a_broken_status_file_does_not_break_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    replace_json(monkeypatch, "live_status.json", {"run_at": "pas une date"})

    app = main().run()

    assert not app.exception, app.exception
    assert any("inconnue" in c.value for c in app.sidebar.caption)


def click_update(app: AppTest) -> AppTest:
    next(b for b in app.button if b.label == "Mettre à jour maintenant").click().run()
    return app


def test_the_update_button_shows_a_provisional_prediction(
    monkeypatch: pytest.MonkeyPatch, today: pd.Timestamp
) -> None:
    log = pd.read_parquet(OUTPUT_DIR / "predictions_log.parquet").drop(columns="predicted_at")
    provisional = log.assign(
        pred=log["pred"] + 1.0, computed_at=pd.Timestamp("2026-11-13 18:00", tz="UTC")
    )
    monkeypatch.setattr(refresh, "light_refresh", lambda: (provisional, None))

    app = click_update(main().run())

    assert not app.exception, app.exception
    assert "provisional" in app.session_state
    assert any("provisoire" in expander.label for expander in app.expander)
    fresh = app.dataframe[1].value  # le tableau provisoire, pas le journal figé
    expected = ad.upcoming_table(
        provisional.assign(predicted_at=provisional["computed_at"]),
        pd.read_parquet(CLEAN_DIR / "auctions.parquet"),
        today,
    )
    assert list(fresh["Bid-to-cover prévu"]) == [ad.fr(v, 2) for v in expected["pred_btc"]]
    assert any("Prévision provisoire calculée" in s.value for s in app.sidebar.success)


@pytest.mark.parametrize(
    ("error", "kind", "text"),
    [
        (refresh.ModelsUnavailable("absents"), "warning", "Modèles indisponibles"),
        (refresh.RefreshBusy("occupé"), "info", "déjà en cours"),
        (ValueError("format imprévu"), "error", "format imprévu"),
    ],
)
def test_the_update_button_explains_each_failure(
    monkeypatch: pytest.MonkeyPatch, error: Exception, kind: str, text: str
) -> None:
    def broken() -> None:
        raise error

    monkeypatch.setattr(refresh, "light_refresh", broken)

    app = click_update(main().run())

    assert not app.exception, app.exception
    assert any(text in element.value for element in getattr(app.sidebar, kind))


def test_a_partial_update_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    log = pd.read_parquet(OUTPUT_DIR / "predictions_log.parquet").drop(columns="predicted_at")
    provisional = log.assign(computed_at=pd.Timestamp("2026-11-13 18:00", tz="UTC"))
    message = "Sources en échec : treasurydirect_upcoming"
    monkeypatch.setattr(refresh, "light_refresh", lambda: (provisional, message))

    app = click_update(main().run())

    assert any("partielle" in w.value for w in app.sidebar.warning)


def test_a_rewritten_file_is_read_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_ui, "OUTPUT_DIR", tmp_path)
    path = tmp_path / "etat.json"
    path.write_text('{"version": 1}')
    assert app_ui.json_file("etat.json") == {"version": 1}

    path.write_text('{"version": 22}')  # le passage du soir réécrit le fichier

    assert app_ui.json_file("etat.json") == {"version": 22}
    path.write_text("pas du json")
    assert app_ui.json_file("etat.json") == {}


def test_the_method_page_links_the_code_and_gives_the_versions() -> None:
    app = page("methodologie").run()

    assert not app.exception, app.exception
    links = " ".join(m.value for m in app.markdown)
    assert "https://github.com/make4this/tresor-auctions/blob/main/journal_llm.md" in links
    assert any("python 3.11" in c.value and "lightgbm" in c.value for c in app.caption)
