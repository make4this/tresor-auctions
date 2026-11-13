"""Tests du lien avec le marché et de la stratégie (section 9 du plan)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tresor import strategy as st


def market(rates: dict[str, float], series: str = "yield_10y") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": "treasury_yield_curve",
            "series": series,
            "unit": "%",
            "date": pd.to_datetime(list(rates)),
            "value": list(rates.values()),
            "available_at": pd.to_datetime(list(rates)),
        }
    )


def test_rate_change_runs_from_the_previous_close_to_the_auction_day_close() -> None:
    rates = market({"2020-01-02": 1.80, "2020-01-03": 1.82, "2020-01-06": 1.87})
    auctions = pd.DataFrame(
        {
            "auction_date": pd.to_datetime(["2020-01-06", "2020-01-03", "2020-01-07"]),
            "maturity_bucket": "10Y",
        },
        index=[10, 11, 12],
    )

    changes = st.rate_changes(auctions, rates)

    assert changes[10] == pytest.approx(5.0)  # lundi : clôture du vendredi -> clôture du lundi
    assert changes[11] == pytest.approx(2.0)  # vendredi : jeudi -> vendredi, jamais le jour même
    assert np.isnan(changes[12])  # pas de taux publié le jour de l'enchère


def test_rate_change_ignores_a_stale_previous_close_and_uses_each_maturity() -> None:
    rates = pd.concat(
        [
            market({"2020-01-02": 1.80, "2020-01-20": 1.90}),
            market({"2020-01-17": 1.50, "2020-01-20": 1.47}, series="yield_2y"),
        ]
    )
    auctions = pd.DataFrame(
        {"auction_date": pd.to_datetime(["2020-01-20"] * 2), "maturity_bucket": ["10Y", "2Y"]}
    )

    changes = st.rate_changes(auctions, rates)

    assert np.isnan(changes[0])  # veille trop ancienne (18 jours)
    assert changes[1] == pytest.approx(-3.0)


def test_duplicated_rates_are_refused() -> None:
    rates = market({"2020-01-02": 1.80, "2020-01-03": 1.82})
    rates = pd.concat([rates, rates.iloc[[1]]])
    auctions = pd.DataFrame(
        {"auction_date": [pd.Timestamp("2020-01-03")], "maturity_bucket": "10Y"}
    )

    with pytest.raises(ValueError, match="plusieurs taux"):
        st.rate_changes(auctions, rates)


def test_quintile_5_is_always_the_weakest_demand() -> None:
    surprise = np.linspace(-1, 1, 100)
    sample = pd.DataFrame(
        {
            "dealer_share": 0.3 + surprise / 10,  # plus de dealers : demande plus faible
            "dealer_share_mean3": 0.3,
            "btc": 2.5 - surprise,  # bid-to-cover plus bas : demande plus faible
            "btc_mean3": 2.5,
            "rate_change_bp": surprise * 2,  # demande faible : le taux monte
        }
    )

    table = st.quintile_table(sample)

    for name in st.SURPRISES:
        part = table[table["surprise"] == name].sort_values("quintile")
        assert list(part["n"]) == [20] * 5
        assert part["rate_change_mean_bp"].is_monotonic_increasing
        assert (part["ci_low"] <= part["rate_change_mean_bp"]).all()


def signal(actual: list[float], prediction: list[float]) -> pd.DataFrame:
    days = pd.to_datetime(["2015-03-10", "2015-03-10", "2021-06-08", "2021-06-09"])
    return pd.DataFrame(
        {
            "family": "nominal_coupon",
            "target": "btc",
            "model": "moyenne_5",
            "feature_set": "toutes",
            "cusip": ["A", "B", "C", "D"],
            "auction_date": days,
            "maturity_bucket": ["10Y", "2Y", "5Y", "30Y"],
            "actual": actual,
            "prediction": prediction,
            "mean3": 2.5,
        }
    )


def test_positions_and_pnl_of_each_rule() -> None:
    rows = signal(actual=[2.4, 2.6, 2.4, 2.6], prediction=[2.4, 2.4, 2.6, 2.6])
    changes = pd.Series([3.0, 3.0, -2.0, 1.0], index=rows.index)

    trades = st.strategy_trades(rows, changes)

    # Prévision sous mean3 : vendeur (+1), on gagne si le taux monte.
    assert list(trades["position"]) == [1, 1, -1, -1]
    assert list(trades["strategie"]) == [2.5, 2.5, 1.5, -1.5]
    assert list(trades["toujours_vendeur"]) == [2.5, 2.5, -2.5, 0.5]
    assert list(trades["information_parfaite"]) == [2.5, -3.5, -2.5, -1.5]


def test_the_strategy_never_looks_at_the_auction_result() -> None:
    changes = pd.Series([3.0, 3.0, -2.0, 1.0])
    first = st.strategy_trades(signal([2.4, 2.6, 2.4, 2.6], [2.4, 2.4, 2.6, 2.6]), changes)
    second = st.strategy_trades(signal([9.0, 0.1, 9.0, 0.1], [2.4, 2.4, 2.6, 2.6]), changes)

    pd.testing.assert_series_equal(first["strategie"], second["strategie"])
    assert not first["information_parfaite"].equals(second["information_parfaite"])


def test_only_the_reference_model_on_coupon_btc_is_traded() -> None:
    rows = signal([2.4] * 4, [2.4] * 4)
    other = pd.concat(
        [rows.assign(model="lasso"), rows.assign(target="dealer_share"), rows.assign(family="bill")]
    )

    kept = st.signal_rows(pd.concat([rows, other], ignore_index=True))

    assert len(kept) == 4 and set(kept["model"]) == {"moyenne_5"}


def test_summary_counts_trades_and_annualizes_on_auction_days() -> None:
    rows = signal(actual=[2.4, 2.6, 2.4, 2.6], prediction=[2.4, 2.4, 2.6, 2.6])
    trades = st.strategy_trades(rows, pd.Series([3.0, 3.0, -2.0, 1.0]))

    summary = st.summarize(trades).set_index(["period", "rule"])

    row = summary.loc[("2014-2026", "strategie")]
    assert row["trades"] == 4
    assert row["mean_bp"] == pytest.approx(1.25)
    assert row["hit_rate"] == 0.75
    # Deux enchères le 10 mars 2015 : un seul jour de P&L (5), puis 1,5 et -1,5.
    daily = pd.Series([5.0, 1.5, -1.5])
    assert row["sharpe"] == pytest.approx(daily.mean() / daily.std() * np.sqrt(3 / 2))
    assert summary.loc[("2014-2019", "strategie"), "trades"] == 2
    assert row["share_short"] == 0.5
