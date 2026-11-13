"""Tests des prévisions en direct et du journal figé (section 10 du plan)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline

from tresor import features as ft
from tresor import live
from tresor import models as md

AUCTION = pd.Timestamp("2026-11-18")


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def predictions(
    pred: float = 2.5, auction: pd.Timestamp = AUCTION, cusip: str = "A"
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cusip": [cusip],
            "auction_date": [auction],
            "target": "btc",
            "model": "moyenne_5",
            "family": "nominal_coupon",
            "maturity_bucket": "20Y",
            "pred": [pred],
            "bench_mean3": [2.61],
            "offering_amt": [16e9],
            "offering_estimated": [0],
        }
    )


def logged(pred: float = 2.5, at: str = "2026-11-10 23:30") -> pd.DataFrame:
    """Journal d'une seule prévision, faite le 10 novembre pour l'enchère du 18."""
    return live.update_log(pd.DataFrame(columns=live.LOG_COLUMNS), predictions(pred), utc(at))


# --- Le journal figé : test obligatoire du plan ----------------------------------------------


@pytest.mark.parametrize(
    "now",
    [
        "2026-11-18 05:30",  # 0 h 30 à New York, le jour de l'enchère
        "2026-11-18 14:00",  # 9 h à New York, avant le résultat de 13 h
        "2026-11-18 23:17",  # le soir, heure du workflow
        "2026-12-01 23:17",  # longtemps après
    ],
)
def test_a_row_is_frozen_from_the_auction_day_and_never_rewritten(now: str) -> None:
    log = logged(pred=2.5)

    updated = live.update_log(log, predictions(pred=9.9), utc(now))

    pd.testing.assert_frame_equal(updated, log)  # même prévision, même heure, au bit près


def test_a_prediction_is_refreshed_until_the_eve_of_the_auction() -> None:
    log = logged(pred=2.5)
    # 22 h à New York la veille de l'enchère (3 h UTC le jour même) : encore modifiable.
    now = utc("2026-11-18 03:00")

    updated = live.update_log(log, predictions(pred=9.9), now)

    assert updated["pred"].item() == 9.9
    assert updated["predicted_at"].item() == now


def test_no_prediction_is_created_on_the_auction_day() -> None:
    empty = pd.DataFrame(columns=live.LOG_COLUMNS)
    both = pd.concat(
        [
            predictions(auction=AUCTION),
            predictions(auction=AUCTION + pd.Timedelta(days=1), cusip="B"),
        ]
    )

    log = live.update_log(empty, both, utc("2026-11-18 14:00"))

    assert list(log["cusip"]) == ["B"]  # l'enchère du jour n'entre jamais dans le journal


def test_an_unchanged_prediction_keeps_its_original_time() -> None:
    log = logged(pred=2.5, at="2026-11-10 23:30")

    updated = live.update_log(log, predictions(pred=2.5), utc("2026-11-12 23:30"))

    assert updated["predicted_at"].item() == utc("2026-11-10 23:30")


def test_rows_missing_from_new_predictions_are_kept() -> None:
    log = logged(pred=2.5)  # l'enchère du 18 a disparu du calendrier : sa ligne reste

    updated = live.update_log(
        log, predictions(auction=AUCTION + pd.Timedelta(days=2), cusip="C"), utc("2026-11-12 12:00")
    )

    assert sorted(updated["cusip"]) == ["A", "C"]


def test_the_guard_refuses_a_changed_frozen_row() -> None:
    log = logged(pred=2.5)
    tampered = log.assign(pred=3.0)

    with pytest.raises(RuntimeError, match="figée"):
        live._check_frozen_unchanged(log, tampered, AUCTION)


def test_duplicated_keys_and_naive_times_are_refused() -> None:
    log = logged()
    with pytest.raises(ValueError, match="double"):
        live.update_log(pd.concat([log, log]), predictions(), utc("2026-11-12 12:00"))
    with pytest.raises(ValueError, match="fuseau"):
        live.update_log(log, predictions(), pd.Timestamp("2026-11-12 12:00"))


# --- Modèles finaux, prévisions et suivi ----------------------------------------------------


def feature_table(rows: int = 120) -> pd.DataFrame:
    """Enchères réalisées des deux familles, plus deux enchères à venir par famille."""
    rng = np.random.default_rng(3)
    frames = []
    for family, bucket in (("nominal_coupon", "10Y"), ("bill", "13W")):
        frame = pd.DataFrame({name: rng.normal(size=rows) for name in ft.FEATURE_COLUMNS})
        for target in ft.TARGETS:
            frame[f"{target}_mean3"] = 2.0
            frame[target] = 2.0 + frame["vix"] + rng.normal(scale=0.1, size=rows)
        frame = frame.assign(
            cusip=[f"{family}{i}" for i in range(rows)],
            auction_date=pd.date_range("2015-01-06", periods=rows, freq="7D"),
            family=family,
            maturity_bucket=bucket,
            status="completed",
            offering_amt=40e9,
            offering_estimated=0,
        )
        upcoming = frame.tail(2).copy()
        upcoming["cusip"] = [f"{family}U1", f"{family}U2"]
        upcoming["auction_date"] = [pd.Timestamp("2026-11-17"), pd.Timestamp("2026-11-19")]
        upcoming["status"] = "upcoming"
        upcoming["offering_estimated"] = [0, 1]
        upcoming[list(ft.TARGETS)] = np.nan
        # Erreur de données : une enchère « réalisée » datée dans le futur ne doit pas être prévue.
        future = frame.tail(1).assign(cusip=f"{family}X", auction_date=pd.Timestamp("2026-11-20"))
        frames += [frame, upcoming, future]
    return pd.concat(frames, ignore_index=True)


def fast_model(name: str) -> Pipeline:
    """Modèle rapide, différent pour chacun des 5 noms (sinon moyenne et médiane se confondent)."""
    return make_pipeline(SimpleImputer(), Ridge(alpha=10.0 ** md.MODEL_NAMES.index(name)))


@pytest.fixture
def trained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Modèles rapides entraînés et sauvegardés dans un dossier temporaire."""
    monkeypatch.setattr(md, "make_model", fast_model)
    clean = tmp_path / "clean"
    clean.mkdir()
    feature_table().to_parquet(clean / "features.parquet")
    models_dir = tmp_path / "models"
    sample = ft.model_sample(pd.read_parquet(clean / "features.parquet"))
    live.train_models(sample, models_dir, n_jobs=1)
    return tmp_path


def test_saved_models_are_reloaded_and_predict(trained: Path) -> None:
    models_dir = trained / "models"
    assert len(list(models_dir.glob("*.joblib"))) == 2 * 4 * 5
    metadata = (models_dir / live.METADATA_FILE).read_text()
    assert '"last_auction_date"' in metadata and "scikit-learn" in metadata

    features = pd.read_parquet(trained / "clean" / "features.parquet")
    rows = live.upcoming_rows(features, pd.Timestamp("2026-11-17"))
    result = live.predict_upcoming(rows, live.load_models(models_dir))

    assert set(result["cusip"]) == {"nominal_couponU2", "billU2"}  # le 17 est déjà aujourd'hui
    assert len(result) == 2 * 4 * len(live.LIVE_MODELS)
    one = result[(result["cusip"] == "billU2") & (result["target"] == "btc")].set_index("model")
    assert one.loc[md.ENSEMBLE, "pred"] == pytest.approx(
        one.loc[list(md.MODEL_NAMES), "pred"].mean()
    )
    assert one.loc[md.ENSEMBLE, "bench_mean3"] == 2.0
    assert set(result["offering_estimated"]) == {1}


def test_missing_models_ask_for_training(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--train"):
        live.load_models(tmp_path)


def test_run_predict_then_track_the_results(trained: Path) -> None:
    clean, output, models_dir = trained / "clean", trained / "output", trained / "models"

    first = live.run_predict(clean, output, models_dir, now=utc("2026-11-14 23:17"))
    assert set(first["cusip"]) == {"nominal_couponU1", "nominal_couponU2", "billU1", "billU2"}

    # Le 17 novembre au soir : l'enchère du 17 a eu lieu, son résultat est publié.
    features = pd.read_parquet(clean / "features.parquet")
    done = features["cusip"] == "billU1"
    features.loc[done, "status"] = "completed"
    features.loc[done, "btc"] = 3.0
    features.to_parquet(clean / "features.parquet")
    second = live.run_predict(clean, output, models_dir, now=utc("2026-11-17 23:17"))

    frozen = first[first["cusip"] == "billU1"].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        second[second["cusip"] == "billU1"].reset_index(drop=True), frozen
    )
    tracking = pd.read_parquet(output / live.TRACKING_FILE)
    row = tracking[
        (tracking["cusip"] == "billU1")
        & (tracking["target"] == "btc")
        & (tracking["model"] == md.ENSEMBLE)
    ]
    assert row["status"].item() == "realisee"
    assert row["error"].item() == pytest.approx(3.0 - row["pred"].item())
    assert row["bench_error"].item() == pytest.approx(1.0)
    pending = tracking[tracking["cusip"] == "billU2"]
    assert set(pending["status"]) == {"en_attente"}
