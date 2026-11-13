"""Tests du lien avec le marché et de la stratégie (section 9 du plan)."""

from __future__ import annotations

from pathlib import Path

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
    # Surprises mélangées dans le temps : chaque quintile couvre les deux années.
    surprise = np.random.default_rng(0).permutation(np.linspace(-1, 1, 100))
    sample = pd.DataFrame(
        {
            "dealer_share": 0.3 + surprise / 10,  # plus de dealers : demande plus faible
            "dealer_share_mean3": 0.3,
            "btc": 2.5 - surprise,  # bid-to-cover plus bas : demande plus faible
            "btc_mean3": 2.5,
            # Demande faible : le taux monte. Surprises les plus proches de 0 : taux stable.
            "rate_change_bp": np.where(np.abs(surprise) < 0.02, 0.0, surprise * 2),
            "auction_date": pd.date_range("2010-01-05", periods=100, freq="7D"),
        }
    )

    table = st.quintile_table(sample)

    for name in st.SURPRISES:
        part = table[(table["surprise"] == name) & (table["period"] == "2009-2026")]
        part = part.sort_values("quintile")
        assert list(part["n"]) == [20] * 5
        assert part["rate_change_mean_bp"].is_monotonic_increasing
        assert list(part["quintile"]) == [1, 2, 3, 4, 5]
        assert (part["ci_low"] <= part["rate_change_mean_bp"]).all()
        assert (part["rate_change_mean_bp"] <= part["ci_high"]).all()
        assert (part["ci_low"] < part["ci_high"]).all()
        assert list(part["share_rate_up"]) == [0.0, 0.0, 0.45, 1.0, 1.0]  # stable : pas une hausse
    assert set(table["period"]) == {"2009-2026", "2009-2013"}  # données de 2010-2011 seulement


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
        [
            rows.assign(model="lasso"),
            rows.assign(target="dealer_share"),
            rows.assign(family="bill"),
            rows.assign(feature_set="sans_nouvelles_sources"),  # sinon chaque enchère compte 2 fois
        ]
    )

    kept = st.signal_rows(pd.concat([rows, other], ignore_index=True))

    assert len(kept) == 4 and set(kept["model"]) == {"moyenne_5"}


def test_summary_counts_trades_and_annualizes_on_auction_days() -> None:
    rows = signal(actual=[2.4, 2.6, 2.4, 2.6], prediction=[2.4, 2.4, 2.4, 2.6])
    trades = st.strategy_trades(rows, pd.Series([3.0, 3.0, -2.0, 1.0]))

    summary = st.summarize(trades).set_index(["period", "rule"])

    row = summary.loc[("2014-2026", "strategie")]
    # Positions : vendeur, vendeur, vendeur, acheteur ; P&L 2,5 ; 2,5 ; -2,5 ; -1,5.
    assert row["trades"] == 4
    assert row["share_short"] == 0.75
    assert row["mean_bp"] == pytest.approx(0.25)
    assert row["mean_gross_bp"] == pytest.approx(0.75)
    assert row["total_bp"] == pytest.approx(1.0)
    assert row["cost_total_bp"] == pytest.approx(2.0)
    assert row["hit_rate"] == 0.5
    # Écart avec « toujours vendeur » : seule la 4e enchère diffère (-1,5 contre 0,5).
    assert row["vs_vendeur_bp"] == pytest.approx(-0.5)
    assert np.isnan(summary.loc[("2014-2026", "toujours_vendeur"), "vs_vendeur_bp"])
    # Deux enchères le 10 mars 2015 : un seul jour de P&L (5), puis -2,5 et -1,5.
    daily = pd.Series([5.0, -2.5, -1.5])
    years = ((pd.Timestamp("2021-06-09") - pd.Timestamp("2015-03-10")).days + 1) / 365.25
    assert row["sharpe"] == pytest.approx(daily.mean() / daily.std() * np.sqrt(3 / years))
    assert summary.loc[("2014-2019", "strategie"), "trades"] == 2


def test_summary_skips_periods_without_trades() -> None:
    rows = signal([2.4] * 4, [2.4] * 4).assign(auction_date=pd.Timestamp("2015-03-10"))

    summary = st.summarize(st.strategy_trades(rows, pd.Series([1.0, 2.0, 3.0, 4.0])))

    assert set(summary["period"]) == {"2014-2026", "2014-2019"}


def test_rate_change_maps_20_and_30_year_auctions_to_their_own_yield() -> None:
    rates = pd.concat(
        [
            market({"2020-01-02": 2.00, "2020-01-03": 2.03}, series="yield_20y"),
            market({"2020-01-02": 2.20, "2020-01-03": 2.10}, series="yield_30y"),
            market({"2020-01-02": 1.80, "2020-01-03": 1.81}),
        ]
    )
    auctions = pd.DataFrame(
        {
            "auction_date": pd.to_datetime(["2020-01-03"] * 3),
            "maturity_bucket": ["20Y", "30Y", "10Y"],
        }
    )

    changes = st.rate_changes(auctions, rates)

    np.testing.assert_allclose(changes, [3.0, -10.0, 1.0])


def test_previous_close_may_be_ten_days_old_but_not_eleven() -> None:
    rates = pd.concat(
        [
            market({"2020-01-01": 1.00, "2020-01-11": 1.05}),
            market({"2020-01-01": 1.00, "2020-01-12": 1.05}, series="yield_2y"),
        ]
    )
    auctions = pd.DataFrame(
        {
            "auction_date": pd.to_datetime(["2020-01-11", "2020-01-12"]),
            "maturity_bucket": ["10Y", "2Y"],
        }
    )

    changes = st.rate_changes(auctions, rates)

    assert changes[0] == pytest.approx(5.0)
    assert np.isnan(changes[1])


def test_auction_dates_with_a_time_are_refused() -> None:
    rates = market({"2020-01-02": 1.80, "2020-01-03": 1.82})
    auctions = pd.DataFrame(
        {"auction_date": [pd.Timestamp("2020-01-03 13:00")], "maturity_bucket": "10Y"}
    )

    with pytest.raises(ValueError, match="heure"):
        st.rate_changes(auctions, rates)


def test_bootstrap_resamples_whole_years() -> None:
    """Trois années de moyennes 0, 1 et 2 : en tirant des années entières, la moyenne vaut 0
    une fois sur 27 (moins de 5 %), donc l'intervalle à 90 % va de 1/3 à 5/3. En tirant les
    enchères une à une, il serait plus étroit."""
    dates = pd.Series(
        pd.to_datetime(
            ["2015-01-05", "2015-06-01", "2016-01-05", "2016-06-01", "2017-01-05", "2017-06-01"]
        )
    )

    low, high = st._mean_interval(pd.Series([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]), dates)

    assert (low, high) == pytest.approx((1 / 3, 5 / 3))


def test_bootstrap_weights_each_year_by_its_number_of_auctions() -> None:
    """Deux années d'une enchère à 0, une année de quatre enchères à 3. Le haut de l'intervalle
    (deux tirages de la grosse année sur trois) vaut 24 / 9, pas la moyenne des moyennes (2)."""
    dates = pd.Series(
        pd.to_datetime(
            ["2015-01-05", "2016-01-05", "2017-01-05", "2017-03-06", "2017-06-05", "2017-09-04"]
        )
    )

    low, high = st._mean_interval(pd.Series([0.0, 0.0, 3.0, 3.0, 3.0, 3.0]), dates)

    assert (low, high) == pytest.approx((0.0, 24 / 9))


def test_event_moves_before_during_and_after_the_auction() -> None:
    days = pd.bdate_range("2020-01-01", periods=30)
    rates = market({day.strftime("%Y-%m-%d"): 1 + 0.01 * k for k, day in enumerate(days)})
    auctions = pd.DataFrame(
        {"auction_date": days[[10, 1, 28]], "maturity_bucket": "10Y"}, index=[0, 1, 2]
    )

    moves = st.event_moves(auctions, rates)

    # Le taux monte de 1 pb par jour de cotation.
    assert moves.loc[0].tolist() == pytest.approx([5.0, 1.0, 3.0])
    assert np.isnan(moves.loc[1, "avant_bp"])  # pas 6 jours d'historique
    assert np.isnan(moves.loc[2, "apres_bp"])  # pas 3 jours après
    assert moves.loc[2, "jour_bp"] == pytest.approx(1.0)


def test_cost_sensitivity_moves_the_mean_by_the_cost() -> None:
    rows = signal(actual=[2.4, 2.6, 2.4, 2.6], prediction=[2.4, 2.4, 2.4, 2.6])
    trades = st.strategy_trades(rows, pd.Series([3.0, 3.0, -2.0, 1.0]))

    table = st.cost_sensitivity(trades).set_index(["cost_bp", "rule"])

    assert table.loc[(0.0, "strategie"), "mean_bp"] == pytest.approx(0.75)
    assert table.loc[(1.0, "strategie"), "mean_bp"] == pytest.approx(-0.25)
    assert table.loc[(0.0, "toujours_vendeur"), "mean_bp"] == pytest.approx(1.25)


def clean_data(directory: Path) -> tuple[Path, Path]:
    """Petit jeu complet : taux quotidiens, variables et prévisions d'enchères de 10 ans."""
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2012-12-03", "2022-01-31")
    walk = 2 + np.cumsum(rng.normal(scale=0.03, size=len(days)))
    rates = pd.concat(
        [
            market(dict(zip(days.strftime("%Y-%m-%d"), walk, strict=True))),
            market(dict(zip(days.strftime("%Y-%m-%d"), walk / 2, strict=True)), "bill_13w"),
        ]
    )
    auctions = pd.date_range("2013-01-08", "2021-12-28", freq="7D")
    btc = 2.5 + rng.normal(scale=0.2, size=len(auctions))
    coupons = pd.DataFrame(
        {
            "cusip": [f"C{i}" for i in range(len(auctions))],
            "auction_date": auctions,
            "family": "nominal_coupon",
            "maturity_bucket": "10Y",
            "status": "completed",
            "dealer_share": 0.3 + rng.normal(scale=0.05, size=len(auctions)),
            "dealer_share_mean3": 0.3,
            "btc": btc,
            "btc_mean3": 2.5,
        }
    )
    bills = coupons.assign(family="bill", maturity_bucket="13W", cusip=coupons["cusip"] + "B")
    # Enchères annoncées, pas encore réalisées : hors de l'échantillon des modèles.
    announced = coupons.head(10).assign(status="announced", cusip=coupons["cusip"].head(10) + "A")
    clean, output = directory / "clean", directory / "output"
    clean.mkdir()
    output.mkdir()
    rates.to_parquet(clean / "market.parquet")
    pd.concat([coupons, bills, announced]).to_parquet(clean / "features.parquet")
    tested = coupons[coupons["auction_date"] >= pd.Timestamp("2014-01-01")]
    predictions = pd.concat(
        [
            tested.assign(model=model, prediction=2.5 + rng.normal(scale=0.1, size=len(tested)))
            for model in ("moyenne_5", "lasso")
        ]
    ).assign(target="btc", feature_set="toutes", actual=lambda f: f["btc"], mean3=2.5)
    predictions.to_parquet(output / "backtest_predictions.parquet")
    return clean, output


def test_run_strategy_end_to_end(tmp_path: Path) -> None:
    clean, output = clean_data(tmp_path)

    summary = st.run_strategy(clean, output, figures=False)

    expected = {
        "quintiles.csv",
        "strategy_trades.parquet",
        "strategy_summary.csv",
        "strategy_by_year.csv",
        "strategy_costs.csv",
        "strategy_signals.csv",
        "event_study.csv",
        "strategy_diagnostics.csv",
    }
    assert expected <= {path.name for path in output.iterdir()}
    quintiles = pd.read_csv(output / "quintiles.csv")
    coupons = (
        469  # enchères de 10 ans du 8 janvier 2013 au 28 décembre 2021 ; les bills sont exclus
    )
    full = quintiles[
        (quintiles["period"] == "2009-2026") & (quintiles["surprise"] == "bid_to_cover")
    ]
    assert full["n"].sum() == coupons
    tested = summary[(summary["period"] == "2014-2026") & (summary["rule"] == "strategie")]
    assert tested["trades"].item() == 417  # enchères depuis 2014
    assert set(pd.read_csv(output / "strategy_signals.csv")["model"]) == {"moyenne_5", "lasso"}


def test_run_strategy_refuses_predictions_out_of_sync(tmp_path: Path) -> None:
    clean, output = clean_data(tmp_path)
    path = output / "backtest_predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions.loc[predictions.index[0], "mean3"] = 9.9
    predictions.to_parquet(path)

    with pytest.raises(ValueError, match="désynchronisées"):
        st.run_strategy(clean, output, figures=False)
