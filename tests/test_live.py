"""Tests des prévisions en direct et du journal figé (section 10 du plan)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from tests.test_models import sample as walk_forward_sample

from tresor import features as ft
from tresor import live
from tresor import models as md

AUCTION = pd.Timestamp("2026-11-18")


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def predictions(
    pred: float = 2.5,
    auction: pd.Timestamp = AUCTION,
    cusip: str = "A",
    offering_amt: float = 16e9,
    offering_estimated: int = 0,
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
            "offering_amt": [offering_amt],
            "offering_estimated": [offering_estimated],
        }
    )


EMPTY = live._empty(live.LOG_COLUMNS)


def logged(pred: float = 2.5, at: str = "2026-11-10 23:30") -> pd.DataFrame:
    """Journal d'une seule prévision, faite le 10 novembre pour l'enchère du 18."""
    return live.update_log(EMPTY, predictions(pred), utc(at))


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


@pytest.mark.parametrize("now", ["2026-11-18 03:00", "2026-11-18 04:30"])
def test_a_prediction_is_refreshed_until_the_eve_of_the_auction(now: str) -> None:
    log = logged(pred=2.5)  # 22 h et 23 h 30 à New York la veille (heure d'hiver, UTC-5)

    updated = live.update_log(log, predictions(pred=9.9), utc(now))

    assert updated["pred"].item() == 9.9
    assert updated["predicted_at"].item() == utc(now)


@pytest.mark.parametrize(
    ("now", "frozen"), [("2026-07-15 03:59", False), ("2026-07-15 04:00", True)]
)
def test_summer_time_moves_the_new_york_midnight(now: str, frozen: bool) -> None:
    auction = pd.Timestamp("2026-07-15")  # heure d'été : minuit à New York = 4 h UTC
    log = live.update_log(EMPTY, predictions(2.5, auction=auction), utc("2026-07-10 23:17"))

    updated = live.update_log(log, predictions(9.9, auction=auction), utc(now))

    assert updated["pred"].item() == (2.5 if frozen else 9.9)


def test_an_earlier_clock_is_refused() -> None:
    """Relancer « hier » ne doit pas rouvrir une ligne figée aujourd'hui."""
    tomorrow = AUCTION + pd.Timedelta(days=1)
    log = live.update_log(EMPTY, predictions(2.5), utc("2026-11-17 23:17"))
    log = live.update_log(
        log, predictions(2.5, cusip="B", auction=tomorrow), utc("2026-11-18 23:17")
    )

    with pytest.raises(ValueError, match="antérieur"):
        live.update_log(log, predictions(9.9), utc("2026-11-17 23:00"))


def test_no_prediction_is_created_on_the_auction_day() -> None:
    tomorrow = AUCTION + pd.Timedelta(days=1)
    both = pd.concat([predictions(auction=AUCTION), predictions(auction=tomorrow, cusip="B")])

    log = live.update_log(EMPTY, both, utc("2026-11-18 14:00"))

    assert list(log["cusip"]) == ["B"]  # l'enchère du jour n'entre jamais dans le journal


def test_an_evening_without_upcoming_auction_keeps_the_log() -> None:
    log = logged()
    none = live.predict_upcoming(pd.DataFrame(columns=["family"]), {})

    pd.testing.assert_frame_equal(live.update_log(log, none, utc("2026-11-18 23:17")), log)


def test_an_unchanged_prediction_is_kept_exactly() -> None:
    log = logged(pred=2.5, at="2026-11-10 23:30")

    updated = live.update_log(log, predictions(pred=2.5 + 1e-15), utc("2026-11-12 23:30"))

    pd.testing.assert_frame_equal(updated, log)  # même valeur au bit près et même heure


@pytest.mark.parametrize(
    "change", [{"pred": 2.5 + 1e-6}, {"offering_amt": 18e9}, {"offering_estimated": 1}]
)
def test_any_real_change_refreshes_the_prediction(change: dict[str, float]) -> None:
    log = logged(pred=2.5)
    now = utc("2026-11-12 23:30")

    updated = live.update_log(log, predictions(**{"pred": 2.5, **change}), now)  # type: ignore[arg-type]

    assert updated["predicted_at"].item() == now
    for name, value in change.items():
        assert updated[name].item() == value


def test_same_cusip_on_two_dates_are_two_rows() -> None:
    """Une réouverture garde son CUSIP : la date de l'enchère fait partie de la clé."""
    later = AUCTION + pd.Timedelta(days=7)
    first = pd.concat([predictions(2.5), predictions(2.6, auction=later)])
    log = live.update_log(EMPTY, first, utc("2026-11-10 23:17"))

    updated = live.update_log(log, predictions(2.9, auction=later), utc("2026-11-12 23:17"))

    assert updated.set_index("auction_date")["pred"].to_dict() == {AUCTION: 2.5, later: 2.9}


def test_rows_missing_from_new_predictions_are_kept() -> None:
    log = logged(pred=2.5)  # l'enchère du 18 a disparu du calendrier : sa ligne reste
    other = predictions(auction=AUCTION + pd.Timedelta(days=2), cusip="C")

    updated = live.update_log(log, other, utc("2026-11-12 12:00"))

    assert sorted(updated["cusip"]) == ["A", "C"]


def test_times_are_stored_in_utc_whatever_the_caller_zone() -> None:
    now = pd.Timestamp("2026-11-10 18:30", tz="America/New_York")

    log = live.update_log(EMPTY, predictions(), now)

    assert str(log["predicted_at"].dtype) == "datetime64[us, UTC]"
    assert log["predicted_at"].item() == utc("2026-11-10 23:30")


def test_a_round_trip_through_parquet_keeps_the_freeze_working(tmp_path: Path) -> None:
    path = tmp_path / live.LOG_FILE
    logged().to_parquet(path, index=False)
    log = live.read_log(path)

    updated = live.update_log(log, predictions(9.9), utc("2026-11-18 23:17").as_unit("ns"))

    pd.testing.assert_frame_equal(updated, log)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda f: f.assign(pred=3.0),
        lambda f: f.assign(offering_amt=1.0),
        lambda f: f.assign(predicted_at=utc("2026-11-11")),
        lambda f: f.assign(pred=f["pred"] + 1e-12),  # même un écart infime
        lambda f: f.assign(maturity_bucket="30Y"),
        lambda f: f.iloc[0:0],
    ],
)
def test_the_guard_refuses_any_change_to_a_frozen_row(tamper: object) -> None:
    log = logged(pred=2.5)

    with pytest.raises(RuntimeError, match="figée"):
        live.check_frozen_unchanged(log, tamper(log), AUCTION)  # type: ignore[operator]


def test_the_guard_refuses_a_frozen_row_added_after_the_fact() -> None:
    """Une prévision glissée dans le journal pour une enchère du jour ou passée, absente de
    l'ancien journal, est refusée, même datée d'avant l'enchère."""
    log = logged(pred=2.5)
    sneaked = live.update_log(
        EMPTY, predictions(2.5, cusip="B", auction=AUCTION), utc("2026-11-10 23:30")
    )

    with pytest.raises(RuntimeError, match="ajoutée après coup"):
        live.check_frozen_unchanged(log, pd.concat([log, sneaked]), AUCTION)
    # La veille de l'enchère, la même ligne n'est pas encore figée : elle peut entrer.
    live.check_frozen_unchanged(log, pd.concat([log, sneaked]), AUCTION - pd.Timedelta(days=1))


@pytest.mark.parametrize(
    ("predicted_at", "refused"),
    [
        ("2026-11-18 04:59", False),  # 23 h 59 à New York la veille : le 18 en UTC, accepté
        ("2026-11-18 05:00", True),  # minuit à New York, le jour de l'enchère
        ("2026-11-19 12:00", True),  # après l'enchère
        (None, True),  # date de prévision manquante
    ],
)
def test_the_guard_refuses_a_prediction_dated_on_its_auction_day(
    predicted_at: str | None, refused: bool
) -> None:
    """La date qui compte est celle de New York, pas celle d'UTC, même si la ligne n'est
    pas encore figée pour le contrôle."""
    log = logged(pred=2.5)
    when = utc(predicted_at) if predicted_at else pd.NaT
    late = log.assign(predicted_at=pd.Series([when], dtype="datetime64[us, UTC]"))
    before = AUCTION - pd.Timedelta(days=5)

    if refused:
        with pytest.raises(RuntimeError, match="datée du jour de son enchère"):
            live.check_frozen_unchanged(EMPTY, late, before)
    else:
        live.check_frozen_unchanged(EMPTY, late, before)


def test_bad_logs_and_naive_times_are_refused(tmp_path: Path) -> None:
    log = logged()
    with pytest.raises(ValueError, match="double"):
        live.update_log(pd.concat([log, log]), predictions(), utc("2026-11-12 12:00"))
    with pytest.raises(ValueError, match="fuseau"):
        live.update_log(log, predictions(), pd.Timestamp("2026-11-12 12:00"))
    for frame in (log.drop(columns="bench_mean3"), log.assign(extra=1)):
        frame.to_parquet(tmp_path / "bad.parquet", index=False)
        with pytest.raises(ValueError, match="colonnes"):
            live.read_log(tmp_path / "bad.parquet")


# --- Modèles finaux, prévisions et suivi ----------------------------------------------------

MEAN3 = {"btc": 2.0, "dealer_share": 0.3, "indirect_share": 0.6, "tail_bp": 1.5}


def feature_table(rows: int = 120) -> pd.DataFrame:
    """Enchères réalisées des deux familles, plus des enchères à venir et deux pièges."""
    rng = np.random.default_rng(3)
    frames = []
    for family, bucket, amount in (("nominal_coupon", "10Y", 42e9), ("bill", "13W", 80e9)):
        frame = pd.DataFrame({name: rng.normal(size=rows) for name in ft.FEATURE_COLUMNS})
        for target in ft.TARGETS:
            frame[f"{target}_mean3"] = MEAN3[target]  # une valeur différente par cible
            frame[target] = MEAN3[target] + frame["vix"] + rng.normal(scale=0.1, size=rows)
        frame = frame.assign(
            cusip=[f"{family}{i}" for i in range(rows)],
            auction_date=pd.date_range("2015-01-06", periods=rows, freq="7D"),
            family=family,
            maturity_bucket=bucket,
            status="completed",
            offering_amt=amount,
            offering_estimated=0,
        )
        upcoming = frame.tail(2).copy()
        upcoming["cusip"] = [f"{family}U1", f"{family}U2"]
        upcoming["auction_date"] = [pd.Timestamp("2026-11-17"), pd.Timestamp("2026-11-19")]
        upcoming["status"] = "upcoming"
        upcoming["offering_estimated"] = [0, 1]
        upcoming[list(ft.TARGETS)] = np.nan
        # Pièges : une enchère « réalisée » datée dans le futur (erreur de données), à ne pas
        # prévoir ; une enchère de 2008, hors de l'échantillon des modèles.
        future = frame.tail(1).assign(cusip=f"{family}X", auction_date=pd.Timestamp("2026-11-20"))
        old = frame.head(1).assign(cusip=f"{family}O", auction_date=pd.Timestamp("2008-06-03"))
        frames += [frame, upcoming, future, old]
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
    live.run_train(clean, tmp_path / "models", n_jobs=1)
    return tmp_path


def test_training_uses_completed_auctions_since_2009(trained: Path) -> None:
    metadata = json.loads((trained / "models" / live.METADATA_FILE).read_text())

    assert metadata["rows"]["bill/btc"] == 121  # 120 + la « réalisée » future ; pas celle de 2008
    assert metadata["versions"]["scikit-learn"] and metadata["sample_sha256"]
    assert len(list((trained / "models").glob("*.joblib"))) == 2 * 4 * 5
    assert not (trained / "models.part").exists()


def test_saved_models_are_reloaded_and_predict(trained: Path) -> None:
    features = pd.read_parquet(trained / "clean" / "features.parquet")
    rows = live.upcoming_rows(features, pd.Timestamp("2026-11-17"))

    result = live.predict_upcoming(rows, live.load_models(trained / "models"))

    assert set(result["cusip"]) == {"nominal_couponU2", "billU2"}  # le 17 est déjà aujourd'hui
    assert len(result) == 2 * 4 * len(live.LIVE_MODELS)
    one = result[(result["cusip"] == "billU2") & (result["target"] == "btc")].set_index("model")
    members = one.loc[list(md.MODEL_NAMES), "pred"]
    assert one.loc[md.ENSEMBLE, "pred"] == pytest.approx(members.mean())
    assert members.nunique() == 5
    ensemble = result[result["model"] == md.ENSEMBLE]
    for target, value in MEAN3.items():
        assert set(ensemble.loc[ensemble["target"] == target, "bench_mean3"]) == {value}
    assert set(result["offering_estimated"]) == {1}
    assert set(result.loc[result["family"] == "bill", "offering_amt"]) == {80e9}


def test_models_from_other_versions_or_features_are_refused(trained: Path) -> None:
    path = trained / "models" / live.METADATA_FILE
    original = json.loads(path.read_text())
    changes = (
        {"versions": {**original["versions"], "scikit-learn": "0.1"}},
        {"features": original["features"][::-1]},
    )
    for change in changes:
        path.write_text(json.dumps({**original, **change}))
        with pytest.raises(RuntimeError, match="--train"):
            live.load_models(trained / "models")


def test_missing_models_ask_for_training(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--train"):
        live.load_models(tmp_path)


def test_final_training_follows_the_walk_forward_recipe() -> None:
    """Entraîné sur les mêmes enchères, le modèle final prévoit comme le walk-forward."""
    data = walk_forward_sample()
    walk = md.walk_forward(data, md.Task("bill", "btc", "ridge", "toutes"))
    fold = walk["fold"].max()
    test = data[data["auction_date"] >= fold]
    noise = data.assign(family="nominal_coupon", btc=99.0)  # une autre famille : ignorée
    past = pd.concat([data[data["auction_date"] < fold], noise]).sample(frac=1, random_state=0)

    final = live._fit_final(past, "bill", "btc", "ridge")
    features = list(md.FEATURE_SETS["toutes"])

    np.testing.assert_array_equal(
        final.predict(test[features].to_numpy(dtype=float)),
        walk.loc[walk["fold"] == fold, "prediction"].to_numpy(),
    )


def test_lightgbm_is_saved_without_its_search_timings() -> None:
    final = live._fit_final(walk_forward_sample(rows=150), "bill", "btc", "lightgbm")

    assert not isinstance(final.steps[-1][1], md.GridSearchCV)


def test_run_predict_then_track_the_results(trained: Path) -> None:
    clean, output, models_dir = trained / "clean", trained / "output", trained / "models"

    # 22 h à New York le 14 novembre : déjà le 15 en UTC.
    first = live.run_predict(clean, output, models_dir, now=utc("2026-11-15 03:00"))
    assert set(first["cusip"]) == {"nominal_couponU1", "nominal_couponU2", "billU1", "billU2"}
    status = json.loads((output / live.STATUS_FILE).read_text())
    assert status["upcoming_auctions"] == 4 and status["today_new_york"] == "2026-11-14"

    # Le 18 novembre au soir : l'enchère de bills du 17 a eu lieu (tail non publié) ; celle des
    # notes du 17 a été reportée, sans résultat.
    features = pd.read_parquet(clean / "features.parquet")
    done = features["cusip"] == "billU1"
    features.loc[done, "status"] = "completed"
    features.loc[done, "btc"] = 3.0
    features.to_parquet(clean / "features.parquet")
    second = live.run_predict(clean, output, models_dir, now=utc("2026-11-18 23:17"))

    day = pd.Timestamp("2026-11-17")
    pd.testing.assert_frame_equal(
        second[second["auction_date"] == day].reset_index(drop=True),
        first[first["auction_date"] == day].reset_index(drop=True),
    )
    tracking = pd.read_parquet(output / live.TRACKING_FILE)
    ensemble = tracking[tracking["model"] == md.ENSEMBLE].set_index(["cusip", "target"])
    row = ensemble.loc[("billU1", "btc")]
    assert row["status"] == "realisee"
    assert row["error"] == pytest.approx(3.0 - row["pred"])
    assert row["bench_error"] == pytest.approx(1.0)
    assert ensemble.loc[("billU1", "tail_bp"), "status"] == "realisee"  # réalisée, sans tail
    assert np.isnan(ensemble.loc[("billU1", "tail_bp"), "error"])
    assert ensemble.loc[("nominal_couponU1", "btc"), "status"] == "sans_resultat"
    assert ensemble.loc[("billU2", "btc"), "status"] == "en_attente"


def test_a_lost_log_is_not_silently_restarted(trained: Path) -> None:
    clean, output, models_dir = trained / "clean", trained / "output", trained / "models"
    live.run_predict(clean, output, models_dir, now=utc("2026-11-14 23:17"))
    (output / live.LOG_FILE).unlink()

    with pytest.raises(FileNotFoundError, match="restaurer"):
        live.run_predict(clean, output, models_dir, now=utc("2026-11-15 23:17"))


def test_a_log_written_by_another_run_meanwhile_is_not_overwritten(
    trained: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean, output, models_dir = trained / "clean", trained / "output", trained / "models"
    live.run_predict(clean, output, models_dir, now=utc("2026-11-14 23:17"))
    original = live.predict_upcoming

    def concurrent(*args: object) -> pd.DataFrame:
        result = original(*args)  # type: ignore[arg-type]
        other = live.read_log(output / live.LOG_FILE).iloc[1:]
        other.to_parquet(output / live.LOG_FILE, index=False)  # l'autre exécution écrit
        return result

    monkeypatch.setattr(live, "predict_upcoming", concurrent)

    with pytest.raises(RuntimeError, match="autre exécution"):
        live.run_predict(clean, output, models_dir, now=utc("2026-11-15 23:17"))


def test_an_auction_of_the_day_waits_for_its_result() -> None:
    log = logged()
    features = pd.DataFrame(
        {
            "cusip": ["A"],
            "auction_date": [AUCTION],
            "status": ["upcoming"],
            **{target: [np.nan] for target in ft.TARGETS},
        }
    )

    today = live.live_tracking(log, features, AUCTION)
    tomorrow = live.live_tracking(log, features, AUCTION + pd.Timedelta(days=1))

    assert today["status"].item() == "en_attente"  # le résultat tombe à 13 h
    assert tomorrow["status"].item() == "sans_resultat"


def test_training_is_skipped_when_the_data_did_not_change(trained: Path) -> None:
    clean, models_dir = trained / "clean", trained / "models"
    before = json.loads((models_dir / live.METADATA_FILE).read_text())

    again = live.run_train(clean, models_dir, n_jobs=1)
    assert again["trained_at"] == before["trained_at"]  # rien n'a été refait

    features = pd.read_parquet(clean / "features.parquet")
    features.loc[0, "btc"] += 0.5  # une donnée d'entraînement change
    features.to_parquet(clean / "features.parquet")
    changed = live.run_train(clean, models_dir, n_jobs=1)
    assert changed["sample_sha256"] != before["sample_sha256"]
    forced = live.run_train(clean, models_dir, n_jobs=1, force=True)
    assert forced["trained_at"] != changed["trained_at"]
