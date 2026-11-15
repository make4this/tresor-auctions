"""Tests des données de l'application (sans Streamlit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tresor import app_data as ad
from tresor.formats import en, fr

NBSP, MINUS = " ", "−"


def test_numbers_for_the_report_and_the_site() -> None:
    """Rapport : nombres à la française ; site : à l'anglaise."""
    assert fr(1234.5, 1) == f"1{NBSP}234,5"
    assert fr(-0.08, 2, sign=True) == f"{MINUS}0,08"
    assert fr(95, 0, suffix=" Md$") == f"95{NBSP}Md$"  # pas de retour à la ligne avant l'unité
    assert en(1234.5, 1) == "1,234.5"
    assert en(-0.08, 2, sign=True) == f"{MINUS}0.08"
    assert en(0.0632, 3, sign=True) == "+0.063"
    assert en(2.9, 1, suffix=" bp") == f"2.9{NBSP}bp"
    for formatter in (fr, en):
        assert formatter(-0.3, 0) == "0"  # pas de « −0 »
        assert formatter(np.nan) == "–"


def test_values_in_their_unit() -> None:
    assert ad.format_target(0.3327, "dealer_share") == "33.3%"
    assert ad.format_target(2.857, "tail_bp") == f"+2.9{NBSP}bp"
    assert ad.format_target(2.7917, "btc") == "2.79"
    assert ad.amount(95e9) == "$95bn" and ad.amount(16e9, estimated=True) == "$16bn (est.)"


def test_auction_titles() -> None:
    assert ad.auction_title("10Y", "nominal_coupon", True) == "10-year (reopening)"
    assert ad.auction_title("20Y", "nominal_coupon", False) == "20-year"
    # Pour les bills, « réouverture » ne dit rien d'utile : non affiché.
    assert ad.auction_title("13W", "bill", True) == "13-week"


def test_feature_names_are_readable() -> None:
    assert ad.feature_label("vix") == "VIX"
    assert ad.feature_label("btc_mean3") == "Bid-to-cover (average of last 3)"
    assert ad.feature_label("dealer_share_lag1") == "Dealer share (last auction)"
    assert ad.feature_label("inconnue") == "inconnue"


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (-0.0816, ad.WEAK),  # 13 semaines du 16 novembre 2026
        (-0.0783, ad.NORMAL),  # 20 ans du 18 novembre : affiché « −0,08 » une fois arrondi
        (2.53 - 2.61, ad.WEAK),  # −0,08000000000000007
        (2.49 - 2.57, ad.WEAK),  # −0,07999999999999963 : la règle décimale reste exacte
        (0.08, ad.STRONG),
        (0.0799, ad.NORMAL),
        (0.0632, ad.NORMAL),
        (np.nan, "–"),
    ],
)
def test_signal_uses_unrounded_values_and_an_exact_threshold(gap: float, expected: str) -> None:
    assert ad.signal(gap) == expected


def log_rows(models: tuple[str, ...] = ("lasso", "moyenne_5")) -> pd.DataFrame:
    """Journal de 4 enchères ; le Lasso vient avant la moyenne des 5 (ordre du vrai journal)."""
    rows = []
    for cusip, day, bucket, family, btc in (
        ("A", "2026-11-12", "10Y", "nominal_coupon", 2.5),  # passée : pas affichée
        ("B", "2026-11-13", "13W", "bill", 2.79),  # aujourd'hui : affichée (résultat à 13 h)
        ("D", "2026-11-18", "52W", "bill", 3.0),
        ("C", "2026-11-18", "20Y", "nominal_coupon", 2.53),
        ("E", "2026-11-18", "6W", "bill", 2.7),
    ):
        for target, pred, mean3 in (
            ("btc", btc, btc + 0.0816 if cusip == "B" else 2.61),
            ("dealer_share", 0.14, 0.13),
            ("indirect_share", 0.6, 0.58),
            ("tail_bp", 0.4, -0.9),
        ):
            for model in models:
                shift = 0.5 if model == "lasso" else 0.0
                rows.append(
                    {
                        "cusip": cusip,
                        "auction_date": pd.Timestamp(day),
                        "target": target,
                        "model": model,
                        "family": family,
                        "maturity_bucket": bucket,
                        "predicted_at": pd.Timestamp("2026-11-12 23:17", tz="UTC"),
                        "pred": pred + shift,
                        "bench_mean3": mean3,
                        "offering_amt": 16e9 if cusip != "D" else 54e9,
                        "offering_estimated": int(cusip == "D"),
                    }
                )
    return pd.DataFrame(rows)


AUCTIONS = pd.DataFrame(
    {
        "cusip": ["B", "C", "C", "D", "E"],  # C en double : ne doit pas doubler la ligne
        "auction_date": pd.to_datetime(
            ["2026-11-13", "2026-11-18", "2026-11-18", "2026-11-18", "2026-11-18"]
        ),
        "reopening": [True, True, True, False, False],
        "security_term": ["13-Week", "19-Year 10-Month", "19-Year 10-Month", "52-Week", "6-Week"],
    }
)


def test_upcoming_table_keeps_today_and_later_with_the_reference_model() -> None:
    table = ad.upcoming_table(log_rows(), AUCTIONS, pd.Timestamp("2026-11-13"))

    assert list(table["cusip"]) == ["B", "C", "E", "D"]  # par date, puis par maturité
    assert list(table["title"]) == [
        "13-week",
        "20-year (reopening)",
        "6-week",
        "52-week",
    ]
    assert table["pred_btc"].tolist() == pytest.approx([2.79, 2.53, 2.7, 3.0])  # pas le Lasso
    assert table["gap_btc"].tolist() == pytest.approx([-0.0816, -0.08, 0.09, 0.39])
    assert list(table["signal"]) == [ad.WEAK, ad.WEAK, ad.STRONG, ad.STRONG]
    assert table["pred_dealer_share"].tolist() == pytest.approx([0.14] * 4)
    assert table["pred_tail_bp"].tolist() == pytest.approx([0.4] * 4)
    assert list(table["offering_estimated"]) == [0, 0, 0, 1]
    assert table.loc[3, "offering_amt"] == 54e9


def test_upcoming_table_without_any_mean3_does_not_crash() -> None:
    """Première enchère d'une nouvelle série : pas encore de moyenne des 3."""
    log = log_rows().assign(bench_mean3=np.nan)

    table = ad.upcoming_table(log, AUCTIONS, pd.Timestamp("2026-11-13"))

    assert len(table) == 4
    assert table["gap_btc"].isna().all() and set(table["signal"]) == {"–"}


def test_upcoming_table_keeps_auctions_missing_from_the_calendar() -> None:
    table = ad.upcoming_table(log_rows(), AUCTIONS.iloc[:1], pd.Timestamp("2026-11-13"))

    assert len(table) == 4
    assert list(table["title"])[1] == "20-year"  # réouverture inconnue : non affichée


def backtest_rows() -> pd.DataFrame:
    """Huit enchères de bills : deux signaux « plus faible », un « plus fort »."""
    frame = pd.DataFrame(
        {
            "family": "bill",
            "target": "btc",
            "model": "moyenne_5",
            "feature_set": "toutes",
            "prediction": [2.0, 2.0, 2.2, 2.1, 2.1, 2.1, 2.1, 2.1],
            "mean3": [2.1] * 8,
            "actual": [2.0, 2.3, 2.3, 2.0, 2.0, 2.0, 2.3, 2.1],
        }
    )
    other = frame.assign(feature_set="sans_nouvelles_sources", prediction=2.5)  # ignoré
    return pd.concat([frame, other], ignore_index=True)


def test_signal_track_record() -> None:
    row = ad.signal_track_record(backtest_rows()).iloc[0]

    assert row["auctions"] == 8
    assert row["weak_count"] == 2 and row["share_weak"] == 0.25 and row["weak_then_below"] == 0.5
    assert row["strong_count"] == 1 and row["strong_then_above"] == 1.0
    assert row["base_rate_below"] == 0.5 and row["base_rate_above"] == 0.375


def test_signal_record_table_and_sentence() -> None:
    record = ad.signal_track_record(backtest_rows())

    table = ad.signal_record_table(record).set_index("signal")

    assert table.loc[ad.WEAK, ["cases", "right"]].tolist() == [2, 1]
    assert table.loc[ad.STRONG, ["cases", "right"]].tolist() == [1, 1]
    assert ad.signal_record_sentence(record, "bill", ad.WEAK) == (
        "Track record in tests over 2014-2026: when the signal said weaker for bills, the"
        " bid-to-cover came in below the last-3 average in 1 of 2 cases (50%), against 50% for"
        " all bill auctions."
    )
    assert "1 of 1 cases (100%)" in ad.signal_record_sentence(record, "bill", ad.STRONG)
    assert ad.signal_record_sentence(record, "bill", ad.NORMAL).startswith("When the gap")
    assert ad.signal_record_sentence(record, "nominal_coupon", ad.WEAK) == ""  # pas de données


def test_closer_verdict_and_error_summary() -> None:
    assert ad.closer(0.01, -0.05) == ad.FORECAST
    assert ad.closer(-0.08, 0.02) == ad.LAST3
    assert ad.closer(0.03, -0.03) == ad.TIE
    assert ad.closer(0.0502, 0.05) == ad.LAST3
    assert ad.closer(0.0502, 0.05, decimals=2) == ad.TIE  # « +0.05 » contre « +0.05 »
    assert [ad.verdict(0.01, 0.2), ad.verdict(-0.3, -0.1), ad.verdict(-0.1, 0.2)] == [
        "better", "worse", "unclear",
    ]  # fmt: skip
    assert ad.verdict(np.nan, np.nan) == "unclear"
    series = pd.DataFrame(
        {
            "actual": [2.0, 2.4, 2.2, 2.6],
            "mean3": [np.nan, 2.0, 2.0, 2.4],
            "prediction": [2.1, 2.3, np.nan, 2.5],
        }
    )

    summary = ad.error_summary(series)

    assert summary["auctions"] == 4 and summary["compared"] == 2  # mêmes enchères pour les deux
    assert summary["forecast"] == pytest.approx(0.1)
    assert summary["last3"] == pytest.approx(0.3)


def test_download_errors_are_shown_in_english() -> None:
    raw = (
        "FetchError : Échec du téléchargement de https://api.example/v1/auctions"
        " après 5 essai(s) : HTTP 503"
    )
    assert ad.error_en(raw) == "Download failed after 5 attempts: HTTP 503"
    assert ad.error_en("FetchError : HTTP 503") == "FetchError : HTTP 503"  # inconnu : tel quel


def reading(pred: float, mean3: float) -> str:
    row = pd.Series({"maturity_bucket": "10Y", "pred_btc": pred, "bench_mean3_btc": mean3})
    return ad.auction_reading(row)


def test_auction_reading() -> None:
    start = "Forecast {} against 2.61 on average over the last 3 10-year auctions: a gap of {}"
    assert reading(2.52, 2.61) == start.format("2.52", f"{MINUS}0.090") + (
        f", at or below the {MINUS}0.08 threshold, so demand should be weaker than usual."
    )
    assert reading(2.53, 2.61).endswith("weaker than usual.")  # −0,08 pile, malgré l'arrondi
    assert reading(2.70, 2.61).endswith("at or above the +0.08 threshold, so demand should be"
                                        " stronger than usual.")  # fmt: skip
    assert reading(2.60, 2.61) == start.format("2.60", f"{MINUS}0.010") + (
        ", within ±0.08, so demand should be normal."
    )
    assert (
        reading(2.60, np.nan) == "No forecast or no last-3 average yet for the 10-year: no signal."
    )


def tracking_rows() -> pd.DataFrame:
    tracking = log_rows(models=("moyenne_5",))
    tracking["status"] = np.select(
        [tracking["cusip"].isin(["A", "C"]), tracking["cusip"].eq("B")],
        ["realisee", "sans_resultat"],
        "en_attente",
    )
    tracking["actual"] = np.where(tracking["status"] == "realisee", 2.6, np.nan)
    tracking["error"] = tracking["actual"] - tracking["pred"]
    tracking["bench_error"] = tracking["actual"] - tracking["bench_mean3"]
    return tracking.sample(frac=1, random_state=0)  # ordre quelconque


def test_live_summary_counts_auctions_once() -> None:
    summary = ad.live_summary(tracking_rows())

    assert summary == {"suivies": 5, "realisees": 2, "en_attente": 2, "sans_resultat": 1}


def test_live_results_are_sorted_and_normalised_like_the_backtest() -> None:
    done = ad.live_results(tracking_rows(), "btc")

    assert list(done["cusip"]) == ["A", "C"]  # par date
    reference = np.array([(2.6 - 2.61) ** 2, (2.6 - 2.61) ** 2])
    model = np.array([(2.6 - 2.5) ** 2, (2.6 - 2.53) ** 2])
    expected = np.cumsum(reference - model) / reference.sum()
    np.testing.assert_allclose(done["gain"], expected)  # la courbe finit au R² en direct


def test_cumulative_gain_ends_at_the_r2() -> None:
    backtest = pd.DataFrame(
        {
            "family": "bill",
            "target": "btc",
            "model": ["moyenne_5"] * 3 + ["lasso"] * 3,
            "feature_set": "toutes",
            "auction_date": pd.to_datetime(["2015-03-01", "2014-01-01", "2016-01-01"] * 2),
            "actual": [3.0, 1.0, 2.0] * 2,
            "prediction": [2.5, 1.5, 2.0, 9.0, 9.0, 9.0],
            "mean3": [2.0, 2.0, 2.0] * 2,
        }
    )

    gain = ad.cumulative_gain(backtest, "bill", "btc")

    assert list(gain["auction_date"]) == list(
        pd.to_datetime(["2014-01-01", "2015-03-01", "2016-01-01"])
    )
    np.testing.assert_allclose(gain["gain"], [0.375, 0.75, 0.75])  # 1 − 0,5 / 2


def test_backtest_r2_reads_the_reference_model_on_2020_2026() -> None:
    metrics = pd.DataFrame(
        {
            "period": ["2020-2026", "2014-2026", "2020-2026", "2020-2026"],
            "family": "bill",
            "target": "btc",
            "model": ["moyenne_5", "moyenne_5", "lasso", "moyenne_5"],
            "feature_set": ["toutes", "toutes", "toutes", "sans_nouvelles_sources"],
            "r2_vs_mean3": [0.2, 0.9, 0.9, 0.9],
            "r2_ci_low": 0.1,
            "r2_ci_high": 0.3,
            "r2_vs_constant": 0.8,
            "direction": 0.6,
        }
    )

    table = ad.backtest_r2(metrics)

    assert len(table) == 1 and table["r2_vs_mean3"].item() == 0.2
    assert table["r2_vs_constant"].item() == 0.8


def test_quality_table() -> None:
    quality = {
        "sources": {
            "cboe_vix": {
                "status": "vert",
                "rows": 10,
                "last_date": "2026-11-12",
                "last_published": "2026-11-12",
                "business_days_since_last": 1,
                "alerts": [],
            },  # fmt: skip
            "cftc_positions": {"status": "rouge", "rows": 5, "alerts": ["retard"]},
        }
    }

    table = ad.quality_table(quality)

    assert table.iloc[0].to_dict() == {
        "source": "cboe_vix",
        "status": "vert",
        "rows": 10,
        "last_date": "2026-11-12",
        "last_published": "2026-11-12",
        "business_days_since_last": 1,
        "alerts": "none",
    }
    assert table.iloc[1]["alerts"] == "retard" and pd.isna(table.iloc[1]["last_date"])
    assert ad.quality_table({}).empty


def test_explorer_series_starts_in_2009_and_overlays_the_backtest() -> None:
    features = pd.DataFrame(
        {
            "cusip": ["O", "A", "B", "N", "C"],
            "auction_date": pd.to_datetime(
                ["2008-10-01", "2013-05-01", "2014-05-01", "2014-06-01", "2026-11-18"]
            ),
            "family": "nominal_coupon",
            "maturity_bucket": "10Y",
            "status": ["completed", "completed", "completed", "completed", "upcoming"],
            "btc": [1.2, 2.5, 2.6, np.nan, np.nan],
            "btc_mean3": [2.4, 2.4, 2.5, 2.55, 2.55],
        }
    )
    backtest = pd.DataFrame(
        {
            "cusip": ["B"],
            "auction_date": pd.to_datetime(["2014-05-01"]),
            "family": "nominal_coupon",
            "maturity_bucket": "10Y",
            "target": "btc",
            "model": "moyenne_5",
            "feature_set": "toutes",
            "prediction": [2.55],
        }
    )

    series = ad.explorer_series(features, backtest, "nominal_coupon", "10Y", "btc")

    assert list(series["cusip"]) == ["A", "B"]  # ni 2008, ni sans valeur, ni à venir
    assert np.isnan(series.loc[0, "prediction"]) and series.loc[1, "prediction"] == 2.55


def test_fetch_table_puts_failures_first_and_survives_bad_files() -> None:
    status = {
        "sources": {
            "cboe_vix": {"ok": True, "last_success_at": "2026-11-14T23:17:00+00:00"},
            "fred_series": {"ok": False, "error": "FetchError : HTTP 503", "last_success_at": None},
            "bizarre": "pas un dictionnaire",
        }
    }

    table = ad.fetch_table(status)

    assert list(table["source"]) == ["fred_series", "cboe_vix"]
    assert list(table["ok"]) == [False, True]
    assert table.loc[1, "error"] == ""
    assert ad.failed_sources(status) == ["FRED: macro series"]
    for broken in ({}, {"sources": "rien"}, {"sources": None}):
        assert ad.fetch_table(broken).empty
        assert ad.failed_sources(broken) == []


def test_market_snapshot_gives_the_last_close_and_its_change() -> None:
    market = pd.DataFrame(
        {
            "series": ["yield_10y", "yield_10y", "vix_close", "vix_close", "fred_walcl"],
            "date": pd.to_datetime(["2026-11-12", "2026-11-13"] * 2 + ["2026-11-13"]),
            "value": [5.10, 5.15, 18.7, 19.3, 1.0],
        }
    )

    snapshot = ad.market_snapshot(market).set_index("series")

    assert list(snapshot.index) == ["yield_10y", "vix_close"]  # séries absentes ignorées
    assert snapshot.loc["yield_10y", "change"] == pytest.approx(5.0)  # en points de base
    assert snapshot.loc["yield_10y", "unit"] == "bp"
    assert snapshot.loc["vix_close", "change"] == pytest.approx(0.6)  # en points
    assert snapshot.loc["vix_close", "unit"] == "pts"


def test_demand_counts() -> None:
    table = pd.DataFrame({"signal": [ad.WEAK, ad.WEAK, ad.NORMAL, "–"]})

    assert ad.demand_counts(table) == {ad.WEAK: 2, ad.NORMAL: 1, ad.STRONG: 0}
    assert ad.demand_counts(pd.DataFrame()) == {ad.WEAK: 0, ad.NORMAL: 0, ad.STRONG: 0}


def test_scorecard_reads_the_interval() -> None:
    metrics = pd.DataFrame(
        {
            "model": "moyenne_5",
            "feature_set": "toutes",
            "period": "2020-2026",
            "family": "bill",
            "target": ["btc", "dealer_share", "tail_bp"],
            "r2_vs_mean3": [0.2, -0.3, 0.1],
            "r2_ci_low": [0.1, -0.5, -0.1],
            "r2_ci_high": [0.3, -0.1, 0.3],
            "r2_vs_constant": 0.5,
            "direction": 0.6,
        }
    )

    verdicts = ad.scorecard(metrics).set_index("target")["verdict"]

    assert verdicts.to_dict() == {"btc": "better", "dealer_share": "worse", "tail_bp": "unclear"}


def test_btc_history_keeps_completed_auctions_of_the_chosen_maturities() -> None:
    features = pd.DataFrame(
        {
            "auction_date": pd.to_datetime(
                ["2020-01-02", "2025-01-02", "2025-02-02", "2026-12-01"]
            ),
            "maturity_bucket": ["10Y", "10Y", "2Y", "10Y"],
            "status": ["completed", "completed", "completed", "upcoming"],
            "btc": [2.4, 2.5, 2.6, np.nan],
        }
    )

    history = ad.btc_history(features, ["10Y"], pd.Timestamp("2021-01-01"))

    assert history["btc"].tolist() == [2.5]


def test_key_facts_refuse_unexpected_files() -> None:
    metrics = pd.DataFrame(
        {
            "model": "moyenne_5",
            "feature_set": "toutes",
            "period": "2020-2026",
            "family": ["nominal_coupon", "bill"],
            "target": "btc",
            "r2_vs_mean3": [0.1, 0.2],
            "r2_ci_low": 0.0,
            "r2_ci_high": 0.3,
            "r2_vs_constant": 0.5,
            "direction": 0.6,
        }
    )
    summary = pd.DataFrame({"period": ["2020-2026"], "rule": ["strategie"], "mean_bp": [-1.4]})
    quintiles = pd.DataFrame(
        {
            "period": ["2009-2026"],
            "surprise": ["bid_to_cover"],
            "quintile": [5],
            "rate_change_mean_bp": [2.0],
        }  # fmt: skip
    )

    facts = ad.key_facts(metrics, summary, quintiles)

    assert facts == {"r2_coupons": 0.1, "r2_bills": 0.2, "weak_move_bp": 2.0, "strategy_bp": -1.4}
    with pytest.raises(ValueError):
        ad.key_facts(metrics, pd.concat([summary, summary]), quintiles)
