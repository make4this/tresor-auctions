"""Tests des données de l'application (sans Streamlit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tresor import app_data as ad

NBSP, MINUS = " ", "−"


def test_french_numbers() -> None:
    assert ad.fr(1234.5, 1) == f"1{NBSP}234,5"
    assert ad.fr(-0.08, 2, sign=True) == f"{MINUS}0,08"
    assert ad.fr(0.0632, 3, sign=True) == "+0,063"
    assert ad.fr(95, 0, suffix=" Md$") == f"95{NBSP}Md$"  # pas de retour à la ligne avant l'unité
    assert ad.fr(-0.3, 0) == "0"  # pas de « −0 »
    assert ad.fr(np.nan) == "–"


def test_values_in_their_unit() -> None:
    assert ad.format_target(0.3327, "dealer_share") == f"33,3{NBSP}%"
    assert ad.format_target(2.857, "tail_bp") == f"+2,9{NBSP}pb"
    assert ad.format_target(2.7917, "btc") == "2,79"


def test_auction_titles() -> None:
    assert ad.auction_title("10Y", "nominal_coupon", True) == "10 ans, réouverture"
    assert ad.auction_title("20Y", "nominal_coupon", False) == "20 ans"
    # Pour les bills, « réouverture » ne dit rien d'utile : non affiché.
    assert ad.auction_title("13W", "bill", True) == "13 semaines"


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
        "13 semaines",
        "20 ans, réouverture",
        "6 semaines",
        "52 semaines",
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
    assert list(table["title"])[1] == "20 ans"  # réouverture inconnue : non affichée


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
        "alerts": "aucune",
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
    assert ad.failed_sources(status) == ["FRED : séries macroéconomiques"]
    for broken in ({}, {"sources": "rien"}, {"sources": None}):
        assert ad.fetch_table(broken).empty
        assert ad.failed_sources(broken) == []
