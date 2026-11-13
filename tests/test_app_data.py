"""Tests des données de l'application (sans Streamlit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tresor import app_data as ad


def test_french_numbers() -> None:
    assert ad.fr(1234.5, 1) == "1 234,5"
    assert ad.fr(-0.08, 2, sign=True) == "−0,08"
    assert ad.fr(0.0632, 3, sign=True) == "+0,063"
    assert ad.fr(95, 0, suffix=" Md$") == "95 Md$"
    assert ad.fr(np.nan) == "–"


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
        (-0.08, ad.WEAK),
        (0.08, ad.STRONG),
        (0.0632, ad.NORMAL),
        (np.nan, "–"),
    ],
)
def test_signal_uses_unrounded_values(gap: float, expected: str) -> None:
    assert ad.signal(gap) == expected


def log_rows() -> pd.DataFrame:
    rows = []
    for cusip, day, bucket, family, btc in (
        ("A", "2026-11-12", "10Y", "nominal_coupon", 2.5),  # passée : pas affichée
        ("B", "2026-11-13", "13W", "bill", 2.79),  # aujourd'hui : affichée (résultat à 13 h)
        ("C", "2026-11-18", "20Y", "nominal_coupon", 2.53),
    ):
        for target, pred, mean3 in (
            ("btc", btc, btc + 0.0816 if cusip == "B" else 2.61),
            ("dealer_share", 0.14, 0.13),
            ("indirect_share", 0.6, 0.58),
            ("tail_bp", 0.4, -0.9),
        ):
            for model, shift in (("moyenne_5", 0.0), ("lasso", 0.5)):
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
                        "offering_amt": 16e9,
                        "offering_estimated": 0,
                    }
                )
    return pd.DataFrame(rows)


def test_upcoming_table_keeps_today_and_later_with_the_reference_model() -> None:
    auctions = pd.DataFrame(
        {
            "cusip": ["B", "C"],
            "auction_date": pd.to_datetime(["2026-11-13", "2026-11-18"]),
            "reopening": [True, True],
            "security_term": ["13-Week", "19-Year 10-Month"],
        }
    )

    table = ad.upcoming_table(log_rows(), auctions, pd.Timestamp("2026-11-13"))

    assert list(table["cusip"]) == ["B", "C"]
    assert list(table["title"]) == ["13 semaines", "20 ans, réouverture"]
    assert table["pred_btc"].tolist() == pytest.approx([2.79, 2.53])  # pas le Lasso (+0,5)
    assert table["gap_btc"].tolist() == pytest.approx([-0.0816, -0.08])
    assert list(table["signal"]) == [ad.WEAK, ad.WEAK]
    assert table["pred_tail_bp"].tolist() == pytest.approx([0.4, 0.4])


def test_signal_track_record() -> None:
    backtest = pd.DataFrame(
        {
            "family": "bill",
            "target": "btc",
            "model": "moyenne_5",
            "feature_set": "toutes",
            "prediction": [2.0, 2.0, 2.2, 2.1],
            "mean3": [2.1, 2.1, 2.1, 2.1],
            "actual": [2.0, 2.3, 2.3, 2.0],
        }
    )

    row = ad.signal_track_record(backtest).iloc[0]

    assert row["share_weak"] == 0.5 and row["weak_then_below"] == 0.5
    assert row["share_strong"] == 0.25 and row["strong_then_above"] == 1.0
    assert row["base_rate_below"] == 0.5


def test_live_results_and_summary() -> None:
    tracking = log_rows().assign(
        status=["realisee"] * 8 + ["en_attente"] * 8 + ["sans_resultat"] * 8,
        actual=[2.6] * 8 + [np.nan] * 16,
    )
    tracking["error"] = tracking["actual"] - tracking["pred"]
    tracking["bench_error"] = tracking["actual"] - tracking["bench_mean3"]

    summary = ad.live_summary(tracking)
    done = ad.live_results(tracking, "btc")

    assert summary == {"suivies": 3, "realisees": 1, "en_attente": 1, "sans_resultat": 1}
    assert len(done) == 1
    assert done["gain"].item() == pytest.approx((2.6 - 2.61) ** 2 - (2.6 - 2.5) ** 2)


def test_quality_table() -> None:
    quality = {"sources": {"cboe_vix": {"status": "vert", "rows": 10, "last_date": "2026-11-12",
                                        "business_days_since_last": 1, "alerts": []}}}  # fmt: skip

    table = ad.quality_table(quality)

    assert table.iloc[0].to_dict() == {
        "source": "cboe_vix",
        "status": "vert",
        "rows": 10,
        "last_date": "2026-11-12",
        "business_days_since_last": 1,
        "alerts": "aucune",
    }


def test_explorer_series_overlays_the_backtest() -> None:
    features = pd.DataFrame(
        {
            "cusip": ["A", "B", "C"],
            "auction_date": pd.to_datetime(["2013-05-01", "2014-05-01", "2026-11-18"]),
            "family": "nominal_coupon",
            "maturity_bucket": "10Y",
            "status": ["completed", "completed", "upcoming"],
            "btc": [2.5, 2.6, np.nan],
            "btc_mean3": [2.4, 2.5, 2.55],
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

    assert list(series["cusip"]) == ["A", "B"]  # l'enchère à venir n'a pas de résultat
    assert np.isnan(series.loc[0, "prediction"]) and series.loc[1, "prediction"] == 2.55
