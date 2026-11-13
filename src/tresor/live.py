"""Prévisions des enchères à venir et journal figé (phase 8, section 10 du plan).

- `train_models` : entraîne les 5 modèles sur tout l'historique, pour chaque famille et cible,
  et les sauvegarde (joblib) : l'application peut ainsi prévoir sans ré-entraîner.
- `predict_upcoming` : prévoit les enchères à venir avec les modèles sauvegardés.
- `update_log` : ajoute ou rafraîchit les prévisions du journal. À partir du jour de
  l'enchère (date de New York), une ligne est figée et n'est plus jamais réécrite.
- `live_tracking` : compare le journal aux résultats publiés.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline

from tresor import features as ft
from tresor import models as md
from tresor.config import CLEAN_DIR, MODELS_DIR, NEW_YORK_TZ, OUTPUT_DIR

logger = logging.getLogger(__name__)

FEATURE_SET = "toutes"  # le jeu de variables du modèle de référence de la phase 6
LIVE_MODELS = (*md.MODEL_NAMES, md.ENSEMBLE)
LOG_KEY = ["cusip", "auction_date", "target", "model"]
LOG_COLUMNS = [
    *LOG_KEY,
    "family",
    "maturity_bucket",
    "predicted_at",  # instant de la prévision, en UTC
    "pred",
    "bench_mean3",
    "offering_amt",
    "offering_estimated",  # 1 si le montant n'était pas encore annoncé
]
# Une prévision est réécrite seulement si l'une de ces valeurs change.
COMPARED = [
    "pred",
    "bench_mean3",
    "offering_amt",
    "offering_estimated",
    "family",
    "maturity_bucket",
]
LOG_FILE = "predictions_log.parquet"
TRACKING_FILE = "live_tracking.parquet"
METADATA_FILE = "metadata.json"


# --- Modèles finaux ------------------------------------------------------------------------


def _model_path(models_dir: Path, family: str, target: str, model: str) -> Path:
    return models_dir / f"{family}__{target}__{model}.joblib"


def _fit_final(sample: pd.DataFrame, family: str, target: str, model: str) -> Pipeline:
    """Même recette que le walk-forward, sur tout l'historique disponible."""
    data = sample[(sample["family"] == family) & sample[target].notna()]
    data = data.sort_values("auction_date", kind="stable")  # ordre utilisé par TimeSeriesSplit
    features = list(md.FEATURE_SETS[FEATURE_SET])
    return md.make_model(model).fit(
        data[features].to_numpy(dtype=float), md.winsorized(data[target]).to_numpy()
    )


def train_models(
    sample: pd.DataFrame, models_dir: Path = MODELS_DIR, n_jobs: int = -1
) -> dict[str, object]:
    """Entraîne et sauvegarde les 5 modèles par famille et cible ; renvoie les métadonnées."""
    tasks = [
        (family, target, model)
        for family in ft.MODELED_FAMILIES
        for target in ft.TARGETS
        for model in md.MODEL_NAMES
    ]
    fitted = Parallel(n_jobs=n_jobs)(delayed(_fit_final)(sample, *task) for task in tasks)
    models_dir.mkdir(parents=True, exist_ok=True)
    for task, pipeline in zip(tasks, fitted, strict=True):
        path = _model_path(models_dir, *task)
        temporary_path = path.with_name(f"{path.name}.part")
        joblib.dump(pipeline, temporary_path, compress=3)
        temporary_path.replace(path)
    import lightgbm
    import sklearn

    metadata: dict[str, object] = {
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "last_auction_date": sample["auction_date"].max().date().isoformat(),
        "rows": {
            f"{family}/{target}": int((sample["family"].eq(family) & sample[target].notna()).sum())
            for family in ft.MODELED_FAMILIES
            for target in ft.TARGETS
        },
        "features": list(md.FEATURE_SETS[FEATURE_SET]),
        "models": list(md.MODEL_NAMES),
        # Un modèle sauvegardé ne se relit de façon sûre qu'avec les mêmes versions.
        "versions": {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__},
    }
    (models_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_models(models_dir: Path = MODELS_DIR) -> dict[tuple[str, str, str], Pipeline]:
    """Relit les modèles sauvegardés, ou explique qu'il faut d'abord lancer --train."""
    loaded = {}
    for family in ft.MODELED_FAMILIES:
        for target in ft.TARGETS:
            for model in md.MODEL_NAMES:
                path = _model_path(models_dir, family, target, model)
                if not path.exists():
                    raise FileNotFoundError(f"modèle absent : {path.name} (lancer --train)")
                loaded[(family, target, model)] = joblib.load(path)
    return loaded


# --- Prévisions ----------------------------------------------------------------------------


def new_york_today(now: pd.Timestamp) -> pd.Timestamp:
    """Date du jour à New York (sans heure), pour un instant donné en UTC."""
    if now.tzinfo is None:
        raise ValueError("instant sans fuseau horaire : passer un instant en UTC")
    return now.tz_convert(NEW_YORK_TZ).normalize().tz_localize(None)


def upcoming_rows(features: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Enchères modélisées à venir, strictement après aujourd'hui (date de New York)."""
    return features[
        features["family"].isin(ft.MODELED_FAMILIES)
        & features["status"].eq("upcoming")
        & (features["auction_date"] > today)
    ]


def predict_upcoming(
    rows: pd.DataFrame, models: dict[tuple[str, str, str], Pipeline]
) -> pd.DataFrame:
    """Une ligne par enchère, cible et modèle (dont la moyenne des 5), au format du journal."""
    features = list(md.FEATURE_SETS[FEATURE_SET])
    parts = []
    for family, group in rows.groupby("family"):
        x = group[features].to_numpy(dtype=float)
        for target in ft.TARGETS:
            predictions = {
                model: models[(family, target, model)].predict(x) for model in md.MODEL_NAMES
            }
            predictions[md.ENSEMBLE] = np.mean(list(predictions.values()), axis=0)
            for model, values in predictions.items():
                parts.append(
                    pd.DataFrame(
                        {
                            "cusip": group["cusip"].to_numpy(),
                            "auction_date": group["auction_date"].to_numpy(),
                            "target": target,
                            "model": model,
                            "family": family,
                            "maturity_bucket": group["maturity_bucket"].to_numpy(),
                            "pred": values,
                            "bench_mean3": group[f"{target}_mean3"].to_numpy(),
                            "offering_amt": group["offering_amt"].to_numpy(dtype=float),
                            "offering_estimated": group["offering_estimated"].to_numpy(dtype=int),
                        }
                    )
                )
    if not parts:
        return pd.DataFrame(columns=[name for name in LOG_COLUMNS if name != "predicted_at"])
    return pd.concat(parts, ignore_index=True)


# --- Journal figé ----------------------------------------------------------------------------


def _check_log(log: pd.DataFrame) -> None:
    if log.duplicated(LOG_KEY).any():
        raise ValueError("journal des prévisions : clé en double")


def _same(old: pd.DataFrame, new: pd.DataFrame) -> np.ndarray:
    """Vrai si la prévision n'a pas changé (les valeurs manquantes sont égales entre elles)."""
    same = np.ones(len(old), dtype=bool)
    for column in COMPARED:
        a, b = old[column].to_numpy(), new[column].to_numpy()
        if column in ("pred", "bench_mean3", "offering_amt"):
            same &= np.isclose(a.astype(float), b.astype(float), rtol=0, atol=1e-12, equal_nan=True)
        else:
            same &= a == b
    return same


def update_log(log: pd.DataFrame, predictions: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Ajoute ou rafraîchit les prévisions, sans jamais toucher une ligne figée.

    Une prévision n'est écrite que pour une enchère postérieure à aujourd'hui (date de New
    York). Toute ligne du journal dont l'enchère est aujourd'hui ou passée est donc figée : sa
    clé, qui contient la date de l'enchère, ne peut pas correspondre à une nouvelle prévision.
    Une prévision inchangée garde sa date d'origine (`predicted_at`).
    """
    _check_log(log)
    today = new_york_today(now)
    fresh = predictions[predictions["auction_date"] > today].assign(predicted_at=now)
    fresh = fresh[LOG_COLUMNS].reset_index(drop=True)
    if log.empty:
        result = fresh
    else:
        previous = fresh[LOG_KEY].merge(log, on=LOG_KEY, how="left", indicator=True)
        known = (previous["_merge"] == "both").to_numpy()
        unchanged = known & _same(previous, fresh)
        fresh.loc[unchanged, "predicted_at"] = previous.loc[unchanged, "predicted_at"].to_numpy()
        replaced = log.set_index(LOG_KEY).index.isin(fresh.set_index(LOG_KEY).index)
        result = pd.concat([log[~replaced], fresh], ignore_index=True)
    result = result.sort_values(LOG_KEY, kind="stable").reset_index(drop=True)
    _check_log(result)
    _check_frozen_unchanged(log, result, today)
    return result


def _check_frozen_unchanged(log: pd.DataFrame, result: pd.DataFrame, today: pd.Timestamp) -> None:
    """Garde-fou : chaque ligne figée doit se retrouver à l'identique dans le nouveau journal."""
    if log.empty:
        return
    frozen = log[log["auction_date"] <= today]
    kept = frozen[LOG_KEY].merge(result, on=LOG_KEY, how="left")
    expected = frozen.sort_values(LOG_KEY, kind="stable").reset_index(drop=True)
    kept = kept[LOG_COLUMNS].sort_values(LOG_KEY, kind="stable").reset_index(drop=True)
    if not kept.equals(expected[LOG_COLUMNS]):
        raise RuntimeError("une ligne figée du journal a changé : écriture annulée")


def read_log(path: Path) -> pd.DataFrame:
    """Journal existant, ou journal vide au bon format."""
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.read_parquet(path)
    missing = set(LOG_COLUMNS) - set(log.columns)
    if missing:
        raise ValueError(f"journal des prévisions illisible : colonnes absentes {sorted(missing)}")
    return log[LOG_COLUMNS]


def live_tracking(log: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Journal et résultats publiés : erreur du modèle et de mean3, statut de chaque prévision."""
    done = features[features["status"].eq("completed")]
    results = done.melt(
        id_vars=["cusip", "auction_date"],
        value_vars=list(ft.TARGETS),
        var_name="target",
        value_name="actual",
    )
    tracked = log.merge(results, on=["cusip", "auction_date", "target"], how="left")
    tracked["status"] = np.where(tracked["actual"].notna(), "realisee", "en_attente")
    tracked["error"] = tracked["actual"] - tracked["pred"]
    tracked["bench_error"] = tracked["actual"] - tracked["bench_mean3"]
    return tracked


# --- Lancement -------------------------------------------------------------------------------


def run_train(clean_dir: Path = CLEAN_DIR, models_dir: Path = MODELS_DIR) -> dict[str, object]:
    """Étape --train : modèles finaux sur toutes les enchères réalisées depuis 2009."""
    sample = ft.model_sample(pd.read_parquet(clean_dir / "features.parquet"))
    metadata = train_models(sample, models_dir)
    logger.info("Modèles finaux entraînés jusqu'au %s", metadata["last_auction_date"])
    return metadata


def run_predict(
    clean_dir: Path = CLEAN_DIR,
    output_dir: Path = OUTPUT_DIR,
    models_dir: Path = MODELS_DIR,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Étape --predict : prévoit les enchères à venir, met à jour le journal et le suivi."""
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    features = pd.read_parquet(clean_dir / "features.parquet")
    rows = upcoming_rows(features, new_york_today(now))
    predictions = predict_upcoming(rows, load_models(models_dir))
    path = output_dir / LOG_FILE
    log = update_log(read_log(path), predictions, now)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.part")
    log.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)
    live_tracking(log, features).to_parquet(output_dir / TRACKING_FILE, index=False)
    logger.info("Prévisions : %d enchères à venir, %d lignes dans le journal",
                len(rows), len(log))  # fmt: skip
    return log
