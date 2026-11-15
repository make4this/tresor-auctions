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
from tresor.formats import en

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = {
    "prochaines": "Upcoming auctions",
    "suivi": "Live track record",
    "backtest": "Model performance",
    "explorer": "History",
    "qualite": "Data quality",
    "methodologie": "Methodology",
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


def test_the_main_app_opens_on_the_home_page() -> None:
    app = main().run()

    assert not app.exception, app.exception
    html = home_html(app)
    assert "US Treasury Auctions" in html and "Upcoming auctions" in html  # bandeau, panneau
    assert any(button.label == "Refresh now" for button in app.button)
    assert all("Academic project" not in c.value for c in [*app.caption, *app.sidebar.caption])


def home_html(app: AppTest) -> str:
    return "\n".join(m.value for m in app.markdown)


def test_the_home_page_lists_the_next_upcoming_auctions(today: pd.Timestamp) -> None:
    log = pd.read_parquet(OUTPUT_DIR / "predictions_log.parquet")
    auctions = pd.read_parquet(CLEAN_DIR / "auctions.parquet")
    upcoming = ad.upcoming_table(log, auctions, today)

    app = main().run()
    html = home_html(app)
    panel = html.split("Upcoming auctions", 1)[1].split("</table>", 1)[0]

    assert not app.exception, app.exception
    assert panel.count("<tr>") == min(10, len(upcoming)) + 1  # en-tête et 10 enchères au plus
    for row in upcoming.head(10).itertuples():
        assert f"<b>{row.title}</b>" in panel
        assert en(row.gap_btc, 2, sign=True) in panel
    counts = ad.demand_counts(upcoming)
    assert f'<div class="tr-value">{counts[ad.WEAK]}</div>' in html


def test_the_home_page_panels_come_from_the_files() -> None:
    market = ad.market_snapshot(pd.read_parquet(CLEAN_DIR / "market.parquet"))
    card = ad.scorecard(pd.read_csv(OUTPUT_DIR / "backtest_metrics.csv"))

    html = home_html(main().run())

    ten_year = market.set_index("series").loc["yield_10y"]
    assert f"{en(ten_year['value'], 2)}%" in html
    for row in card.itertuples():
        value = en(row.r2_vs_mean3 * 100, 0, sign=True) + "%"
        assert f'tr-cell tr-{row.verdict}">{value}</div>' in html


def test_the_home_page_numbers_come_from_the_files() -> None:
    facts = ad.key_facts(
        pd.read_csv(OUTPUT_DIR / "backtest_metrics.csv"),
        pd.read_csv(OUTPUT_DIR / "strategy_summary.csv"),
        pd.read_csv(OUTPUT_DIR / "quintiles.csv"),
    )

    html = home_html(main().run())

    assert en(-facts["r2_coupons"] * 100, 0, sign=True, suffix="%") in html
    assert en(-facts["r2_bills"] * 100, 0, sign=True, suffix="%") in html
    assert f"{en(facts['weak_move_bp'], 1, sign=True)} bp" in html
    assert f"{en(facts['strategy_bp'], 1, sign=True)} bp" in html


def test_the_home_page_without_upcoming_auction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ad, "upcoming_table", lambda *args, **kwargs: pd.DataFrame())

    app = main().run()

    assert not app.exception, app.exception
    assert "No bill, note or bond auction has been announced yet." in home_html(app)


def test_the_upcoming_table_shows_the_reference_model_from_today(today: pd.Timestamp) -> None:
    log = pd.read_parquet(OUTPUT_DIR / "predictions_log.parquet")
    auctions = pd.read_parquet(CLEAN_DIR / "auctions.parquet")
    expected = ad.upcoming_table(log, auctions, today)

    app = page("prochaines").run()
    shown = app.dataframe[0].value

    assert len(shown) == len(expected) == log.loc[log["auction_date"] >= today, "cusip"].nunique()
    assert list(shown["Expected demand"]) == list(expected["signal"])
    assert list(shown["Bid-to-cover forecast"]) == [en(v, 2) for v in expected["pred_btc"]]
    assert list(shown["Dealer share forecast"]) == [
        ad.format_target(v, "dealer_share") for v in expected["pred_dealer_share"]
    ]
    record = ad.signal_track_record(pd.read_parquet(OUTPUT_DIR / "backtest_predictions.parquet"))
    row = record.set_index("family").loc["nominal_coupon"]
    text = " ".join(m.value for m in app.markdown)
    assert f"{en(row['weak_then_below'] * 100, 0)}% of the time" in text
    assert f"versus {en(row['base_rate_below'] * 100, 0)}% for all auctions" in text


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
    assert list(table["Forecast"]) == [
        ad.format_target(v, "dealer_share") for v in realised["pred"]
    ]


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
    assert strategy["Trades"].iloc[0] == "1,004"  # période 2014-2026
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
    sources = table_with(app, "Status").set_index("Source")
    assert sources.loc["Cboe: VIX", "Status"] == "🔴 problem"
    checks = table_with(app, "Check").set_index("Check")
    assert checks.loc["Market: rates out of bounds", "Cases found"] == 2


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
        "fiscaldata_debt": {
            "ok": False,
            "error": "FetchError : HTTP 503",
            "last_success_at": None,  # jamais réussie
        },
    },
}


def test_a_source_outage_is_shown_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    replace_json(monkeypatch, "fetch_status.json", FETCH_FAILED)

    app = page("qualite").run()
    main_app = main().run()

    assert not app.exception, app.exception
    assert any("2 source(s) failed" in w.value and "Fiscal Data: auction results" in w.value
               for w in app.warning)  # fmt: skip
    downloads = table_with(app, "Result")
    assert list(downloads["Result"]) == ["🔴 failed", "🔴 failed", "🟢 received"]  # échecs d'abord
    assert list(downloads["Last success"][:2]) == [
        "Fri, Nov 13, 2026, 6:20 PM ET",
        "unknown",  # jamais réussie : la page ne doit pas planter
    ]
    assert downloads["Error"][0] == "FetchError : HTTP 503"
    assert any("2 source(s) failed to update" in c.value for c in main_app.sidebar.caption)


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("2026-11-14T20:31:56+00:00", "Sat, Nov 14, 2026, 3:31 PM ET"),
        ("2026-11-14T14:05:00+00:00", "Sat, Nov 14, 2026, 9:05 AM ET"),
        ("2026-07-06T03:00:00+00:00", "Sun, Jul 5, 2026, 11:00 PM ET"),  # heure d'été
        (None, "unknown"),
        (float("nan"), "unknown"),
        ("", "unknown"),
        ("2026-11-14", "unknown"),  # sans fuseau
        ("pas une date", "unknown"),
    ],
)
def test_moments_are_shown_in_new_york_time(text: object, shown: str) -> None:
    assert app_ui.moment_en(text) == shown


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
    assert all("failed" not in c.value for c in app.sidebar.caption)  # rien à signaler

    replace_json(monkeypatch, "fetch_status.json", {})
    quality = page("qualite").run()
    assert not quality.exception, quality.exception
    assert not quality.warning or all("failed" not in w.value for w in quality.warning)
    assert any("Download report missing" in c.value for c in quality.caption)


def test_a_broken_status_file_does_not_break_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    replace_json(monkeypatch, "live_status.json", {"run_at": "pas une date"})

    app = main().run()

    assert not app.exception, app.exception
    assert "Data updated unknown" in home_html(app)


def click_update(app: AppTest) -> AppTest:
    next(b for b in app.button if b.label == "Refresh now").click().run()
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
    assert any("Provisional forecast ready" in s.value for s in app.sidebar.success)
    app.switch_page("vues/prochaines.py").run()  # la prévision provisoire est sur cette page

    assert not app.exception, app.exception
    assert "provisional" in app.session_state
    assert any("Provisional forecast" in expander.label for expander in app.expander)
    fresh = app.dataframe[1].value  # le tableau provisoire, pas le journal figé
    expected = ad.upcoming_table(
        provisional.assign(predicted_at=provisional["computed_at"]),
        pd.read_parquet(CLEAN_DIR / "auctions.parquet"),
        today,
    )
    assert list(fresh["Bid-to-cover forecast"]) == [en(v, 2) for v in expected["pred_btc"]]


@pytest.mark.parametrize(
    ("error", "kind", "text"),
    [
        (refresh.ModelsUnavailable("absents"), "warning", "Models are not available"),
        (refresh.RefreshBusy("occupé"), "info", "already running"),
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

    assert any("Partial refresh" in w.value for w in app.sidebar.warning)


def test_a_rewritten_file_is_read_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_ui, "OUTPUT_DIR", tmp_path)
    path = tmp_path / "etat.json"
    path.write_text('{"version": 1}')
    assert app_ui.json_file("etat.json") == {"version": 1}

    path.write_text('{"version": 22}')  # le passage du soir réécrit le fichier

    assert app_ui.json_file("etat.json") == {"version": 22}
    path.write_text("pas du json")
    assert app_ui.json_file("etat.json") == {}


def test_the_method_page_quotes_the_files() -> None:
    r2 = ad.backtest_r2(pd.read_csv(OUTPUT_DIR / "backtest_metrics.csv"))
    coupons = r2.set_index(["family", "target"]).loc[("nominal_coupon", "btc"), "r2_vs_mean3"]

    app = page("methodologie").run()

    assert not app.exception, app.exception
    text = " ".join(m.value for m in app.markdown)
    assert f"{en(coupons * 100, 1, sign=True)}% R²" in text
    assert "LLM" not in text and not app.caption  # ni mention des LLM ni versions sur le site
