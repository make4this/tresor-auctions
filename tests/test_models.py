"""Tests des modèles et de la validation walk-forward (section 8 du plan)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tresor import features as ft
from tresor import models as md

QUARTER_START = pd.Timestamp("2014-04-01")


def test_clipper_learns_its_bounds_on_the_training_window_only() -> None:
    train = np.column_stack([np.arange(100.0), np.full(100, np.nan)])  # 2e colonne vide

    clipper = md.QuantileClipper().fit(train)
    clipped = clipper.transform(np.array([[-50.0, 1.0], [500.0, 2.0]]))

    assert clipped[0, 0] == pytest.approx(np.quantile(np.arange(100.0), 0.01))
    assert clipped[1, 0] == pytest.approx(np.quantile(np.arange(100.0), 0.99))
    assert np.isnan(clipped[:, 1]).all()  # une colonne vide ne fait pas échouer l'entraînement


def test_quarters() -> None:
    folds = md.quarters(pd.Timestamp("2014-01-01"), pd.Timestamp("2014-08-15"))

    assert [(s.date().isoformat(), e.date().isoformat()) for s, e in folds] == [
        ("2014-01-01", "2014-03-31"),
        ("2014-04-01", "2014-06-30"),
        ("2014-07-01", "2014-09-30"),
    ]


def test_training_target_is_winsorized() -> None:
    target = pd.Series([*np.ones(98), 100.0, -100.0])

    clipped = md.winsorized(target)

    assert clipped.max() < 100 and clipped.min() > -100


def sample(rows: int = 400) -> pd.DataFrame:
    """Enchères hebdomadaires synthétiques (le mardi) dont la cible dépend d'une seule variable.

    Une enchère tombe exactement le 1er avril 2014, premier jour d'un trimestre de test.
    """
    rng = np.random.default_rng(42)
    days = pd.date_range("2010-01-05", periods=rows, freq="7D")
    frame = pd.DataFrame({name: rng.normal(size=rows) for name in ft.FEATURE_COLUMNS})
    frame["btc_mean3"] = 2.5 + 0.1 * rng.normal(size=rows)
    frame["btc_lag1"] = 2.5
    frame["btc"] = 2.5 + 0.5 * frame["vix"] + rng.normal(scale=0.1, size=rows)
    return frame.assign(
        auction_date=days,
        cusip=[f"C{i}" for i in range(rows)],
        family="bill",
        maturity_bucket="13W",
        status="completed",
    )


class Recorder:
    """Faux modèle : note les cibles de chaque fenêtre d'entraînement, prévoit 0."""

    targets: list[np.ndarray] = []
    names: list[str] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> Recorder:
        Recorder.targets.append(np.asarray(y).copy())
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x))


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[Recorder]:
    Recorder.targets, Recorder.names = [], []

    def fake(name: str) -> Recorder:
        Recorder.names.append(name)
        return Recorder()

    monkeypatch.setattr(md, "make_model", fake)
    return Recorder


def test_walk_forward_trains_only_on_auctions_before_each_quarter(
    recorder: type[Recorder],
) -> None:
    data = sample()
    assert QUARTER_START in set(data["auction_date"])  # le cas limite est bien présent

    predictions = md.walk_forward(data, md.Task("bill", "btc", "ridge", "toutes"))

    starts = sorted(predictions["fold"].unique())
    expected = [int((data["auction_date"] < start).sum()) for start in starts]
    assert [len(y) for y in recorder.targets] == expected  # strictement avant, jamais le jour même
    assert predictions["auction_date"].min() >= md.TEST_START
    assert (predictions["auction_date"] >= predictions["fold"]).all()
    # L'enchère du 1er avril est prévue dans le trimestre qui commence ce jour-là.
    same_day = predictions[predictions["auction_date"] == QUARTER_START]
    assert list(same_day["fold"]) == [QUARTER_START]


def test_changing_the_future_does_not_change_past_predictions() -> None:
    """Test anti-fuite le plus direct : on brouille tout après une date, cibles et variables.
    Avec un vrai modèle, les prévisions faites avant cette date doivent rester identiques."""
    data = sample()
    cut = pd.Timestamp("2015-07-01")
    future = data["auction_date"] >= cut
    scrambled = data.copy()
    columns = [*ft.FEATURE_COLUMNS, "btc"]
    rng = np.random.default_rng(0)
    scrambled.loc[future, columns] = rng.normal(scale=50, size=(int(future.sum()), len(columns)))
    task = md.Task("bill", "btc", "ridge", "toutes")

    before = md.walk_forward(data, task)
    after = md.walk_forward(scrambled, task)

    past = before["fold"] < cut
    assert past.sum() > 50
    pd.testing.assert_frame_equal(before[past], after[after["fold"] < cut])
    assert not np.allclose(before.loc[~past, "prediction"], after.loc[~past, "prediction"])


def test_an_outlier_in_a_test_feature_moves_only_its_own_prediction() -> None:
    """Écrêtage, imputation et standardisation sont appris sur l'entraînement seulement : une
    valeur extrême dans le trimestre testé ne change pas les prévisions des autres enchères."""
    data = sample()
    task = md.Task("bill", "btc", "ridge", "toutes")
    shocked = data.copy()
    shocked.loc[shocked["auction_date"] == QUARTER_START, "vix"] = 1000.0

    before = md.walk_forward(data, task)
    after = md.walk_forward(shocked, task)

    hit = before["auction_date"] == QUARTER_START
    # Jusqu'au trimestre du choc inclus ; ensuite, l'enchère choquée entre dans l'entraînement.
    until = before["fold"] <= QUARTER_START
    pd.testing.assert_frame_equal(before[until & ~hit], after[until & ~hit])
    assert (until & ~hit & (before["fold"] == QUARTER_START)).sum() >= 10
    assert before.loc[hit, "prediction"].item() != after.loc[hit, "prediction"].item()


def test_an_outlier_in_the_test_quarter_does_not_move_the_training_target(
    recorder: type[Recorder],
) -> None:
    """L'écrêtage de la cible est appris sur la fenêtre d'entraînement seulement."""
    data = sample()
    task = md.Task("bill", "btc", "ridge", "toutes")
    md.walk_forward(data, task)
    clean = list(recorder.targets)
    recorder.targets.clear()
    shocked = data.copy()
    shocked.loc[shocked["auction_date"] == QUARTER_START, "btc"] = 1000.0

    md.walk_forward(shocked, task)

    folds = sorted(md.walk_forward(data, task)["fold"].unique())
    index = folds.index(QUARTER_START)
    for before, after in zip(clean[: index + 1], recorder.targets[: index + 1], strict=True):
        np.testing.assert_array_equal(before, after)  # trimestres jusqu'au choc : intacts
    assert recorder.targets[index + 1].max() > clean[index + 1].max()  # ensuite, il est appris


def test_residual_variant_learns_the_gap_to_mean3_and_adds_it_back(
    recorder: type[Recorder],
) -> None:
    data = sample()

    predictions = md.walk_forward(data, md.Task("bill", "btc", "ridge_ecart", "toutes"))

    assert set(recorder.names) == {"ridge"}
    first = data[data["auction_date"] < md.TEST_START]
    expected = md.winsorized(first["btc"] - first["btc_mean3"]).to_numpy()
    np.testing.assert_allclose(recorder.targets[0], expected)
    # Le faux modèle prévoit un écart nul : la prévision finale est exactement mean3.
    np.testing.assert_allclose(predictions["prediction"], predictions["mean3"])
    assert set(predictions["model"]) == {"ridge_ecart"}


def test_constant_benchmark_uses_only_the_training_window(recorder: type[Recorder]) -> None:
    data = sample()
    predictions = md.walk_forward(data, md.Task("bill", "btc", "ridge", "toutes"))
    later = data.copy()
    later.loc[later["auction_date"] >= pd.Timestamp("2016-01-01"), "btc"] = 1e6

    first = md.add_constant_benchmark(predictions, data)
    second = md.add_constant_benchmark(predictions, later)

    fold = md.TEST_START
    past = data.loc[data["auction_date"] < fold, "btc"]
    expected = md.winsorized(past).mean()
    assert first.loc[first["fold"] == fold, "constant"].unique() == pytest.approx(expected)
    early = first["fold"] < pd.Timestamp("2016-01-01")
    np.testing.assert_allclose(first.loc[early, "constant"], second.loc[early, "constant"])


def test_a_real_model_learns_the_signal() -> None:
    data = sample()
    predictions = md.walk_forward(data, md.Task("bill", "btc", "ridge", "toutes"))

    result = md.score(md.add_constant_benchmark(predictions, data))

    assert result["r2_vs_mean3"] > 0.8
    assert result["r2_vs_constant"] > 0.8
    assert result["corr_surprise"] > 0.8


def benchmark_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "actual": [3.0, 1.0, 2.0, 2.4],
            "prediction": [2.5, 1.5, 2.0, 2.0],
            "mean3": [2.0, 2.0, 2.0, 2.0],
            "naive": [2.0, 2.0, 2.0, 2.0],
            "constant": [2.5, 2.5, 2.5, 2.5],
        }
    )


def test_score_against_the_benchmarks() -> None:
    result = md.score(benchmark_frame())

    # Erreurs du modèle : 0,25 + 0,25 + 0 + 0,16 ; de mean3 : 1 + 1 + 0 + 0,16.
    assert result["r2_vs_mean3"] == pytest.approx(1 - 0.66 / 2.16)
    # De la constante : 0,25 + 2,25 + 0,25 + 0,01.
    assert result["r2_vs_constant"] == pytest.approx(1 - 0.66 / 2.76)
    assert result["rmse"] == pytest.approx(np.sqrt(0.66 / 4))
    # 3e enchère : égale à mean3 ; 4e : prévue égale à mean3. Aucune des deux ne compte.
    assert result["direction"] == 1.0


def test_direction_counts_a_wrong_side() -> None:
    frame = benchmark_frame().assign(prediction=[2.5, 2.5, 2.0, 2.0])  # 2e enchère : mauvais côté

    assert md.score(frame)["direction"] == 0.5


def predictions_frame() -> pd.DataFrame:
    """Deux enchères par trimestre, deux trimestres par période ; « sans » plus loin du réel."""
    dates = pd.to_datetime(["2015-02-01", "2015-02-15", "2015-05-01", "2015-05-15"] * 2)
    dates = dates + pd.to_timedelta([0] * 4 + [6 * 365] * 4, unit="D")  # 2015 puis 2021
    rows = []
    for feature_set, shift in (("toutes", 0.0), ("sans_nouvelles_sources", 0.2)):
        for number, model in enumerate(md.MODEL_NAMES):
            noise = 0.05 * number  # les 5 modèles diffèrent un peu
            rows.append(
                pd.DataFrame(
                    {
                        "family": "bill",
                        "target": "btc",
                        "model": model,
                        "feature_set": feature_set,
                        "fold": dates.to_period("Q").start_time,
                        "cusip": [f"C{i}" for i in range(8)],
                        "auction_date": dates,
                        "maturity_bucket": "13W",
                        "actual": [3.0, 1.0] * 4,
                        "prediction": [2.6 - shift - noise, 1.4 + shift + noise] * 4,
                        "naive": [1.0, 3.0] * 4,
                        "mean3": [2.0, 2.0] * 4,
                        "constant": [2.1, 2.1] * 4,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_ensemble_is_the_average_of_the_five_models() -> None:
    predictions = predictions_frame()

    with_ensemble = md.add_ensemble(md.add_ensemble(predictions))  # deux fois : pas de doublon

    ensemble = with_ensemble[with_ensemble["model"] == md.ENSEMBLE]
    assert len(ensemble) == 16
    expected = predictions.groupby(["feature_set", "cusip"])["prediction"].mean()
    got = ensemble.set_index(["feature_set", "cusip"])["prediction"]
    pd.testing.assert_series_equal(got.sort_index(), expected.sort_index(), check_names=False)
    assert ensemble["constant"].eq(2.1).all()  # les benchmarks sont gardés


def test_bootstrap_interval_brackets_the_estimate_and_is_reproducible() -> None:
    frame = predictions_frame()
    frame = frame[(frame["model"] == "ridge") & (frame["feature_set"] == "toutes")].copy()
    frame.loc[frame["auction_date"].dt.year == 2021, "prediction"] = 2.0  # 2021 : égal à mean3

    low, high = md.bootstrap_r2(frame)

    assert low <= md.score(frame)["r2_vs_mean3"] <= high
    assert (low, high) == md.bootstrap_r2(frame)
    assert md.bootstrap_r2(frame, other=frame) == (0.0, 0.0)  # écart avec soi-même : nul


def test_r2_without_the_best_quarter() -> None:
    frame = benchmark_frame().assign(
        fold=pd.to_datetime(["2014-01-01", "2014-01-01", "2014-04-01", "2014-04-01"]),
        prediction=[3.0, 1.0, 2.5, 2.0],  # parfait au 1er trimestre, moins bon ensuite
        auction_date=pd.to_datetime(["2014-01-10", "2014-02-10", "2014-04-10", "2014-05-10"]),
    )

    # Sans le 1er trimestre, il reste : modèle 0,25 + 0,16 ; mean3 0 + 0,16.
    assert md.r2_without_best_quarter(frame) == pytest.approx(1 - 0.41 / 0.16)


def test_tables_include_benchmarks_periods_and_the_new_sources_experiment() -> None:
    predictions = md.add_ensemble(predictions_frame())

    metrics = md.metrics_table(predictions)
    comparison = md.new_sources_comparison(predictions)

    assert set(metrics["model"]) == {*md.MODEL_NAMES, md.ENSEMBLE, *md.BENCHMARKS}
    assert set(metrics["period"]) == set(md.PERIODS)
    reference = metrics[metrics["model"] == "mean3"]
    assert (reference["r2_vs_mean3"] == 0).all()
    assert (metrics["r2_ci_low"] <= metrics["r2_ci_high"]).all()
    assert set(comparison["model"]) == {*md.MODEL_NAMES, md.ENSEMBLE}
    assert (comparison["gain"] > 0).all()  # les prévisions « avec » sont plus proches
    assert md.best_models(metrics).shape[0] == 1


def test_best_model_is_chosen_on_2014_2019_only() -> None:
    rows = []
    for period, winner in (("2014-2019", "lasso"), ("2020-2026", "lightgbm")):
        for model in [*md.MODEL_NAMES, md.ENSEMBLE]:
            rows.append(
                {
                    "period": period,
                    "family": "bill",
                    "target": "btc",
                    "model": model,
                    "feature_set": "toutes",
                    "r2_vs_mean3": 0.3 if model == winner else 0.1,
                    "r2_vs_constant": 0.0,
                    "direction": 0.5,
                    "r2_ci_low": 0.0,
                    "r2_ci_high": 0.4,
                }
            )

    best = md.best_models(pd.DataFrame(rows))

    assert best.loc[0, "model"] == "lasso"  # pas LightGBM, meilleur sur la période jugée
    assert best.loc[0, "r2_selection"] == 0.3
    assert best.loc[0, "r2_evaluation"] == 0.1


def test_unknown_model_is_refused() -> None:
    with pytest.raises(ValueError, match="modèle inconnu"):
        md.make_model("svm")
