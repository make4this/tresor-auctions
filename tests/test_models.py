"""Tests des modèles et de la validation walk-forward (section 8 du plan)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tresor import features as ft
from tresor import models as md


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
    """Enchères hebdomadaires synthétiques dont la cible dépend d'une seule variable."""
    rng = np.random.default_rng(42)
    days = pd.date_range("2010-01-04", periods=rows, freq="7D")
    frame = pd.DataFrame({name: rng.normal(size=rows) for name in ft.FEATURE_COLUMNS})
    frame["btc_mean3"] = 2.5
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
    """Faux modèle : note la taille de chaque fenêtre d'entraînement."""

    sizes: list[int] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> Recorder:
        Recorder.sizes.append(len(x))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x))


def test_walk_forward_trains_only_on_auctions_before_each_quarter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = sample()
    Recorder.sizes = []
    monkeypatch.setattr(md, "make_model", lambda name: Recorder())

    predictions = md.walk_forward(data, md.Task("bill", "btc", "ridge", "toutes"))

    expected = [
        int((data["auction_date"] < start).sum()) for start in sorted(predictions["fold"].unique())
    ]
    assert Recorder.sizes == expected  # strictement avant le trimestre testé, jamais après
    assert predictions["auction_date"].min() >= md.TEST_START
    assert (predictions["auction_date"] >= predictions["fold"]).all()


def test_a_real_model_learns_the_signal() -> None:
    predictions = md.walk_forward(sample(), md.Task("bill", "btc", "ridge", "toutes"))

    assert md.score(predictions)["r2_vs_mean3"] > 0.8


def test_score_against_the_mean3_benchmark() -> None:
    frame = pd.DataFrame(
        {
            "actual": [3.0, 1.0, 2.0],
            "prediction": [2.5, 1.5, 2.0],
            "mean3": [2.0, 2.0, 2.0],
            "naive": [2.0, 2.0, 2.0],
        }
    )

    result = md.score(frame)

    # Erreurs du modèle : 0,25 + 0,25 + 0 ; de mean3 : 1 + 1 + 0.
    assert result["r2_vs_mean3"] == pytest.approx(1 - 0.5 / 2.0)
    assert result["direction"] == 1.0  # la 3e enchère, égale à mean3, ne compte pas
    assert result["rmse"] == pytest.approx(np.sqrt(0.5 / 3))


def test_tables_include_benchmarks_and_the_new_sources_experiment() -> None:
    rows = []
    for feature_set, shift in (("toutes", 0.0), ("sans_nouvelles_sources", 0.2)):
        for model in md.MODEL_NAMES:
            rows.append(
                pd.DataFrame(
                    {
                        "family": "bill",
                        "target": "btc",
                        "model": model,
                        "feature_set": feature_set,
                        "actual": [3.0, 1.0],
                        "prediction": [2.6 - shift, 1.4 + shift],  # « sans » : plus loin du réel
                        "naive": [1.0, 3.0],
                        "mean3": [2.0, 2.0],
                    }
                )
            )
    predictions = pd.concat(rows, ignore_index=True)

    metrics = md.metrics_table(predictions)
    comparison = md.new_sources_comparison(metrics)

    assert set(metrics["model"]) == {*md.MODEL_NAMES, *md.BENCHMARKS}
    assert metrics.loc[metrics["model"] == "mean3", "r2_vs_mean3"].iloc[0] == 0.0
    assert (comparison["gain"] > 0).all()  # les prévisions « avec » sont plus proches
    assert md.best_models(metrics).shape[0] == 1


def test_unknown_model_is_refused() -> None:
    with pytest.raises(ValueError, match="modèle inconnu"):
        md.make_model("svm")
