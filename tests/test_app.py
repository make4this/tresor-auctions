"""Chaque page de l'application tourne et affiche les bonnes valeurs (sans navigateur)."""

from __future__ import annotations

import base64
import io
import json
import re
from html import escape, unescape
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
    "donnees": "Dataset",
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
    assert (
        "CoverCast" in html and "Upcoming auctions" in html
    )  # nom du site dans le bandeau, panneau
    assert any(button.label == "Refresh now" for button in app.button)
    assert all("Academic project" not in c.value for c in app.caption)
    assert not app.sidebar.markdown and not app.sidebar.button  # navigation en onglets en haut


def home_html(app: AppTest) -> str:
    return "\n".join(m.value for m in app.markdown)


def html_table(app: AppTest, title: str) -> pd.DataFrame:
    """Le premier tableau HTML qui suit le titre de panneau `title`, en texte brut."""
    part = home_html(app).split(f">{title}<", 1)[1].split("</table>", 1)[0]
    rows = [
        [
            unescape(re.sub(r"<[^>]+>", "", cell)).strip()
            for cell in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row)
        ]
        for row in re.findall(r"<tr>(.*?)</tr>", part)
    ]
    return pd.DataFrame(rows[1:], columns=rows[0])


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
        assert app_ui.signed_number(row.gap_btc, 3) in panel
    counts = ad.demand_counts(upcoming)
    assert (
        f"All {len(upcoming)} announced auctions: {counts[ad.WEAK]} weaker,"
        f" {counts[ad.NORMAL]} normal, {counts[ad.STRONG]} stronger." in html
    )
    first = upcoming.iloc[0]
    hero = html.split('class="tr-hero"', 1)[1].split('class="tr-foot"', 1)[0]
    assert escape(app_ui.date_en(first["auction_date"], year=False)) in hero
    assert en(first["pred_btc"], 2) in hero and en(first["gap_btc"], 3, sign=True) in hero
    ticker = html.split('class="tr-track"', 1)[1].split("</div></div>", 1)[0]
    assert ticker.count('class="tr-tick"') == 2 * len(upcoming)  # deux fois, pour boucler
    gaps = json.loads(app.get("plotly_chart")[0].proto.spec)["data"][0]["y"]
    if isinstance(gaps, dict):  # Plotly encode les valeurs en binaire
        gaps = np.frombuffer(base64.b64decode(gaps["bdata"]), dtype=gaps["dtype"])
    assert np.allclose(gaps, upcoming["gap_btc"])  # graphique des écarts, dans l'ordre


def test_the_home_page_reads_one_auction(today: pd.Timestamp) -> None:
    log = pd.read_parquet(OUTPUT_DIR / "predictions_log.parquet")
    auctions = pd.read_parquet(CLEAN_DIR / "auctions.parquet")
    upcoming = ad.upcoming_table(log, auctions, today)
    record = ad.signal_track_record(pd.read_parquet(OUTPUT_DIR / "backtest_predictions.parquet"))

    app = main().run()

    assert not app.exception, app.exception
    for index in (
        0,
        next(i for i, f in enumerate(upcoming["family"]) if f != upcoming["family"].iloc[0]),
    ):
        if index:  # une enchère de l'autre famille, choisie dans la liste
            app.selectbox[0].set_value(index).run()
        row = upcoming.iloc[index]
        html = home_html(app)
        assert escape(ad.auction_reading(row)) in html
        sentence = ad.signal_record_sentence(record, row["family"], row["signal"])
        assert escape(sentence) in html
        period = app.segmented_control(key=f"focus_period_{row['family']}")
        assert period.value == ("6M" if row["family"] == "bill" else "3Y")


def test_the_expected_demand_chart_can_be_reshaped(today: pd.Timestamp) -> None:
    log = pd.read_parquet(OUTPUT_DIR / "predictions_log.parquet")
    upcoming = ad.upcoming_table(log, pd.read_parquet(CLEAN_DIR / "auctions.parquet"), today)
    app = main().run()

    def traces() -> list[dict]:
        return json.loads(app.get("plotly_chart")[0].proto.spec)["data"]

    assert [t["type"] for t in traces()] == ["bar"]  # par défaut : l'écart en barres
    app.segmented_control(key="hero_view").set_value("levels").run()
    # Valeurs côte à côte : des barres cacheraient l'écart, la vue passe en points reliés.
    assert [t.get("name") for t in traces()] == [None, "Last-3 average", "Forecast"]
    assert all(t["type"] == "scatter" for t in traces())
    app.segmented_control(key="hero_style").set_value("line").run()
    assert all(t["mode"] == "lines+markers" for t in traces())
    app.segmented_control(key="hero_axis").set_value("date").run()
    app.segmented_control(key="hero_horizon").set_value("7").run()
    week = upcoming[upcoming["auction_date"] < today + pd.Timedelta(days=7)]
    assert len(traces()[0]["x"]) == len(week)
    app.pills(key="hero_families").set_value(["nominal_coupon"]).run()
    coupons = week[week["family"] == "nominal_coupon"]
    assert len(traces()[0]["x"]) == len(coupons) if len(coupons) else True
    assert not app.exception, app.exception


def test_signed_numbers_are_rounded_before_their_colour() -> None:
    assert app_ui.signed_number(0.0004, 3) == '<span class="tr-flat">0.000</span>'
    assert app_ui.signed_number(-0.0004, 3) == '<span class="tr-flat">0.000</span>'  # pas de −0
    assert app_ui.signed_number(-0.085, 3) == '<span class="tr-down">−0.085</span>'
    assert app_ui.signed_number(4.6, 0, " bp") == '<span class="tr-up">+5 bp</span>'


def test_table_cells_can_carry_a_colour_and_long_tables_scroll() -> None:
    html = app_ui.table(["Model", "R²"], [["Ridge", ("+9.6%", "tr-better")]], right={1})

    assert '<td class="tr-right tr-better">+9.6%</td>' in html
    assert app_ui.table(["A"], [["x"]], scroll=True).startswith('<div class="tr-scroll">')


def test_a_panel_title_opens_its_guide() -> None:
    with_guide = app_ui.panel("Market today", "<table></table>", guide=["<b>VIX</b>: stress."])
    plain = app_ui.panel("Key findings", "<div></div>")

    assert '<details class="tr-help">' in with_guide and "How to read" in with_guide
    assert "<li><b>VIX</b>: stress.</li>" in with_guide
    assert "<details" not in plain and "How to read" not in plain


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


def test_the_upcoming_page_shows_every_auction_and_the_track_record(
    today: pd.Timestamp,
) -> None:
    log = pd.read_parquet(OUTPUT_DIR / "predictions_log.parquet")
    auctions = pd.read_parquet(CLEAN_DIR / "auctions.parquet")
    expected = ad.upcoming_table(log, auctions, today)

    app = page("prochaines").run()
    html = "\n".join(m.value for m in app.markdown)
    table = html.split("All announced auctions", 1)[1].split("</table>", 1)[0]

    assert not app.exception, app.exception
    rows = table.count("<tr>") - 1  # sans l'en-tête
    assert rows == len(expected) == log.loc[log["auction_date"] >= today, "cusip"].nunique()
    css = {ad.WEAK: "weak", ad.NORMAL: "normal", ad.STRONG: "strong"}
    assert re.findall(r"tr-tag tr-(\w+)", table) == [css[s] for s in expected["signal"]]
    for row in expected.itertuples():
        assert ad.format_target(row.pred_dealer_share, "dealer_share") in table
    record = ad.signal_record_table(
        ad.signal_track_record(pd.read_parquet(OUTPUT_DIR / "backtest_predictions.parquet"))
    )
    for row in record.itertuples():
        assert f"{en(row.right, 0)} ({en(row.share * 100, 0)}%)" in html
        assert f"{en(row.base * 100, 0)}%" in html


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
    results_in = en(len(auctions) - 1, 0)  # toutes réalisées sauf une
    assert f'<div class="tr-value">{results_in}</div><div class="tr-text">results in' in home_html(
        app
    )
    assert len(app.get("plotly_chart")) == 2  # prévu contre réalisé, gain cumulé
    app.segmented_control[0].set_value("dealer_share").run()
    table = html_table(app, "Results")
    realised = tracking[(tracking["model"] == "moyenne_5") & (tracking["target"] == "dealer_share")]
    realised = realised[realised["status"] == "realisee"].sort_values(["auction_date", "cusip"])
    expected = [ad.format_target(v, "dealer_share") for v in realised["pred"]]
    assert list(table["Forecast"]) == expected[::-1]  # les plus récentes d'abord
    winners = [  # parts : comparées en %, à 1 décimale, comme à l'écran
        ad.closer(e * 100, b * 100, 1)
        for e, b in zip(realised["error"], realised["bench_error"], strict=True)
    ]
    labels = {ad.FORECAST: "Forecast", ad.LAST3: "Last 3", ad.TIE: "Tie"}
    assert list(table["Closer"]) == [labels[w] for w in winners[::-1]]
    wins = en(winners.count(ad.FORECAST), 0)
    assert f"last-3 average for {wins} of {en(len(winners), 0)} auctions" in home_html(app)


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

    summary = ad.error_summary(
        ad.explorer_series(
            features, pd.read_parquet(OUTPUT_DIR / "backtest_predictions.parquet"),
            "nominal_coupon", "10Y", "btc",
        )
    )  # fmt: skip
    html = home_html(app)
    assert f'<div class="tr-value">{en(summary["forecast"], 2)}</div>' in html
    assert f'<div class="tr-value">{en(summary["last3"], 2)}</div>' in html

    app.selectbox[0].set_value("bill").run()
    app.selectbox[1].set_value("26W").run()
    app.selectbox[2].set_value("dealer_share").run()
    assert not app.exception, app.exception
    assert "\xa0pts</div>" in home_html(app)  # erreur des parts en points de %


def test_the_backtest_page_defaults_and_tabs() -> None:
    app = page("backtest").run()

    assert app.selectbox[1].value == "moyenne_5"  # modèle de référence par défaut
    strategy = html_table(app, "Strategy results, 2014-2026")
    assert strategy["Trades"].iloc[0] == "1,004"  # période 2014-2026
    scores = html_table(app, "Scores by model").set_index("Model")
    metrics = pd.read_csv(OUTPUT_DIR / "backtest_metrics.csv")
    row = metrics[
        (metrics["family"] == "nominal_coupon") & (metrics["period"] == "2020-2026")
        & (metrics["feature_set"] == "toutes") & (metrics["model"] == "moyenne_5")
        & (metrics["target"] == "btc")
    ].iloc[0]  # fmt: skip
    assert scores.loc["5-model average", "Bid-to-cover"].startswith(
        en(row["r2_vs_mean3"] * 100, 1, sign=True)
    )
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
    sources = html_table(app, "Sources").set_index("Source")
    assert sources.loc["Cboe: VIX", "Status"] == "Problem"
    checks = html_table(app, "Checks").set_index("Check")
    assert checks.loc["Market: rates out of bounds", "Cases found"] == "2"


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
    downloads = html_table(app, "Last download")
    assert list(downloads["Result"]) == ["Failed", "Failed", "Received"]  # échecs d'abord
    assert list(downloads["Last success"][:2]) == [
        "Fri, Nov 13, 2026, 6:20 PM ET",
        "unknown",  # jamais réussie : la page ne doit pas planter
    ]
    assert downloads["Error"][0] == "FetchError : HTTP 503"
    assert "2 source(s) failed to update" in home_html(main_app)  # dans le bandeau du haut


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
    assert "failed to update" not in home_html(app)  # rien à signaler

    replace_json(monkeypatch, "fetch_status.json", {})
    quality = page("qualite").run()
    assert not quality.exception, quality.exception
    assert not quality.warning or all("failed" not in w.value for w in quality.warning)
    assert "Download report missing" in home_html(quality)


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
    assert any("Provisional forecast ready" in s.value for s in app.success)
    app.switch_page("vues/prochaines.py").run()  # la prévision provisoire est sur cette page

    assert not app.exception, app.exception
    assert "provisional" in app.session_state
    assert any("Provisional forecast" in expander.label for expander in app.expander)
    box = next(e for e in app.expander if "Provisional forecast" in e.label)
    fresh = "\n".join(m.value for m in box.markdown)  # le tableau provisoire, pas le journal
    expected = ad.upcoming_table(
        provisional.assign(predicted_at=provisional["computed_at"]),
        pd.read_parquet(CLEAN_DIR / "auctions.parquet"),
        today,
    )
    assert fresh.count("<tr>") == len(expected) + 1
    for value in expected["pred_btc"]:
        assert f'<td class="tr-right">{en(value, 2)}</td>' in fresh


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
    assert any(text in element.value for element in getattr(app, kind))


def test_a_partial_update_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    log = pd.read_parquet(OUTPUT_DIR / "predictions_log.parquet").drop(columns="predicted_at")
    provisional = log.assign(computed_at=pd.Timestamp("2026-11-13 18:00", tz="UTC"))
    message = "Sources en échec : treasurydirect_upcoming"
    monkeypatch.setattr(refresh, "light_refresh", lambda: (provisional, message))

    app = click_update(main().run())

    assert any("Partial refresh" in w.value for w in app.warning)


def test_a_rewritten_file_is_read_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_ui, "OUTPUT_DIR", tmp_path)
    path = tmp_path / "etat.json"
    path.write_text('{"version": 1}')
    assert app_ui.json_file("etat.json") == {"version": 1}

    path.write_text('{"version": 22}')  # le passage du soir réécrit le fichier

    assert app_ui.json_file("etat.json") == {"version": 22}
    path.write_text("pas du json")
    assert app_ui.json_file("etat.json") == {}


def test_the_dataset_page_offers_each_file_without_restricted_columns() -> None:
    app = page("donnees").run()

    assert not app.exception, app.exception
    assert len(app.get("download_button")) == 3
    for name, folder in (
        ("auctions.parquet", "clean"),
        ("features.parquet", "clean"),
        ("live_tracking.parquet", "output"),
    ):
        source = pd.read_parquet((CLEAN_DIR if folder == "clean" else OUTPUT_DIR) / name)
        shared = pd.read_csv(io.BytesIO(app_ui.public_csv(name, folder=folder)))
        assert len(shared) == len(source)  # toutes les lignes
        assert not set(ad.RESTRICTED_COLUMNS) & set(shared.columns)  # rediffusion limitée
        assert set(shared.columns) == set(source.columns) - set(ad.RESTRICTED_COLUMNS)
    html = home_html(app)
    assert "Predictor: " in html and "<code>btc</code>" in html  # guide des colonnes


def test_the_method_page_quotes_the_files() -> None:
    r2 = ad.backtest_r2(pd.read_csv(OUTPUT_DIR / "backtest_metrics.csv"))
    coupons = r2.set_index(["family", "target"]).loc[("nominal_coupon", "btc"), "r2_vs_mean3"]

    app = page("methodologie").run()

    assert not app.exception, app.exception
    text = " ".join(m.value for m in app.markdown)
    assert f"{en(coupons * 100, 1, sign=True)}% R²" in text
    assert "LLM" not in text and not app.caption  # ni mention des LLM ni versions sur le site
